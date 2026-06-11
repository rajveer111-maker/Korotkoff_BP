import h5py, numpy as np, os, pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------
# CONFIG  - edit these
# ---------------------------------------------
FILE_PATH  = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_below.h5'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_dashboard_v7b_b210.png'

RF_FREQ_HZ_FALLBACK = 915e6    # Hz  - your USRP TX centre frequency
FS_FALLBACK         = 10_000   # Hz  - your sample rate

# IQ channel wiring correction. 
# Run iq_diagnostic.py to find which one gives HR disp of 0.5-5 mm.
# Options: 'I+jQ', 'Q+jI', 'I-jQ', '-I+jQ'
IQ_MODE = '-I+jQ'   # - change after running diagnostic

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)

def sliding_kurtosis(x, w):
    return pd.Series(x).rolling(window=w).kurt().fillna(0).values

def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def cprint(msg, color='white'):
    codes = {'red':'\033[91m','green':'\033[92m','yellow':'\033[93m',
             'cyan':'\033[96m','white':'\033[0m'}
    print(f"{codes.get(color,'')}{msg}\033[0m")

def sanity(label, val, lo, hi, unit, abs_val=True):
    v = abs(val) if abs_val else val
    ok = lo <= v <= hi
    flag  = '[OK]' if ok else '[FAIL] OUT OF RANGE'
    color = 'green' if ok else 'red'
    cprint(f"  {label:<28}: {val:>10.4f} {unit}   {flag}  (expected {lo}-{hi})", color)
    return ok


# ---------------------------------------------
# IQ CHANNEL WIRING
# ---------------------------------------------
def apply_iq_mode(i_raw, q_raw, mode):
    modes = {
        'I+jQ' : i_raw  + 1j * q_raw,   # standard
        'Q+jI' : q_raw  + 1j * i_raw,   # channels swapped
        'I-jQ' : i_raw  - 1j * q_raw,   # Q inverted (USB vs LSB)
        '-I+jQ': -i_raw + 1j * q_raw,   # I inverted
    }
    if mode not in modes:
        cprint(f"  Unknown IQ_MODE '{mode}', defaulting to I+jQ", 'yellow')
        mode = 'I+jQ'
    cprint(f"  IQ mode: {mode}", 'cyan')
    return modes[mode]


# ---------------------------------------------
# IQ CONDITIONING (AD9361)
# ---------------------------------------------
def b210_iq_condition(iq_raw):
    i_c = iq_raw.real - iq_raw.real.mean()
    q_c = iq_raw.imag - iq_raw.imag.mean()

    p1 = np.mean(i_c**2)
    p2 = np.mean(q_c**2)
    p3 = np.mean(i_c * q_c)

    sin_phi = p3 / np.sqrt(p1 * p2 + 1e-20)
    cos_phi = np.sqrt(max(1.0 - sin_phi**2, 1e-10))
    alpha   = np.sqrt(p2 / (p1 + 1e-20))

    phase_err_deg = np.degrees(np.arcsin(np.clip(sin_phi, -1, 1)))
    amp_err_pct   = abs(alpha - 1.0) * 100

    # We allow up to 90 degrees to handle severe hardware port issues
    PHASE_LIMIT = 90.0   # degrees
    AMP_LIMIT   = 50.0   # percent (tightened slightly to focus on phase)

    if abs(phase_err_deg) < PHASE_LIMIT and amp_err_pct < 100.0:
        q_corr = (q_c - sin_phi * i_c) / (alpha * cos_phi + 1e-15)
        corrected = True
    else:
        q_corr = q_c
        corrected = False
        cprint(f"  [WARN] Imbalance too large for correction "
               f"(phase={phase_err_deg:.1f}deg, amp={amp_err_pct:.1f}%) - "
               f"check IQ_MODE setting", 'red')

    return i_c + 1j * q_corr, phase_err_deg, amp_err_pct, corrected


# ---------------------------------------------
# ROBUST PHASE EXTRACTION
# ---------------------------------------------
def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))

    hist, bins = np.histogram(dphi, bins=512)
    bin_w          = bins[1] - bins[0]
    carrier_offset = bins[np.argmax(hist)] + bin_w / 2
    dphi_c         = dphi - carrier_offset

    iqr_val  = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip_val = max(3.0 * iqr_val, 0.017)   # floor ~ 1 Hz at FS=10kHz
    dphi_c   = np.clip(dphi_c, -clip_val, clip_val)

    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    phase = signal.detrend(phase, type='linear')

    return phase, carrier_offset


# ---------------------------------------------
# ADC HEALTH CHECK
# ---------------------------------------------
def adc_health(i_raw, q_raw):
    cprint(\"\n-- ADC Health Check ---------------------------------\", 'cyan')
    for ch, x in [('I', i_raw), ('Q', q_raw)]:
        mx   = np.max(np.abs(x))
        std  = np.std(x)
        clip = mx > 0.95
        weak = std < 0.001
        status = ('[FAIL] CLIPPED' if clip else ('[WARN] VERY WEAK' if weak else '[OK] OK'))
        color  = 'red' if clip else ('yellow' if weak else 'green')
        cprint(f\"  {ch}: std={std:.5f}  |max|={mx:.5f}  {status}\", color)
        if clip:
            cprint(f\"     -> Reduce USRP RX gain to avoid saturation\", 'yellow')
        if weak:
            cprint(f\"     -> Increase USRP RX gain or check antenna\", 'yellow')


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def run():
    if not os.path.exists(FILE_PATH):
        cprint(\"File not found: \" + FILE_PATH, 'red'); return

    # -- Load ------------------------------------------------------
    with h5py.File(FILE_PATH, 'r') as f:
        data  = f['data'][:]
        attrs = dict(f.attrs)
        cprint(f\"\n  HDF5 attrs: {list(attrs.keys()) or '(none)'}\", 'cyan')

        def _attr(keys, fallback):
            for k in keys:
                if k in attrs:
                    cprint(f\"    '{k}' = {attrs[k]} (from HDF5)\", 'green')
                    return float(attrs[k])
            cprint(f\"    Not in HDF5 -> fallback {fallback}\", 'yellow')
            return float(fallback)

        RF_FREQ_HZ = _attr(['center_freq','rf_freq','fc','rx_freq','freq'], 
                            RF_FREQ_HZ_FALLBACK)
        FS         = _attr(['sample_rate','fs','samp_rate','rx_rate'], 
                            FS_FALLBACK)

    i_raw, q_raw = data[0, :], data[1, :]
    time = np.arange(len(i_raw)) / FS
    cprint(f\"\n  Samples={len(i_raw)}  Duration={time[-1]:.2f}s  FS={FS:.0f}Hz\", 'cyan')

    # -- Physical constants -----------------------------------------
    LAMBDA_MM = (299_792_458.0 / RF_FREQ_HZ) * 1000
    SCALE     = LAMBDA_MM / (4 * np.pi)

    cprint(f\"\n-- USRP B210 ----------------------------------------\", 'cyan')
    cprint(f\"  RF freq  : {RF_FREQ_HZ/1e9:.4f} GHz\", 'cyan')
    cprint(f\"  lambda   : {LAMBDA_MM:.2f} mm\", 'cyan')
    cprint(f\"  Scale    : {SCALE:.4f} mm/rad\", 'cyan')
    cprint(f\"  FS       : {FS:.0f} Hz\", 'cyan')

    # -- ADC health -------------------------------------------------
    adc_health(i_raw, q_raw)

    # -- IQ wiring + DC removal + imbalance correction --------------
    iq_wired = apply_iq_mode(i_raw, q_raw, IQ_MODE)
    iq, phase_err_deg, amp_err_pct, corrected = b210_iq_condition(iq_wired)
    mag_raw = np.abs(iq)

    cprint(f\"\n-- AD9361 IQ Imbalance ------------------------------\", 'cyan')
    pe_col = 'green' if abs(phase_err_deg) < 5 else ('yellow' if abs(phase_err_deg) < 20 else 'red')
    ae_col = 'green' if amp_err_pct < 5       else ('yellow' if amp_err_pct < 25       else 'red')
    cprint(f\"  Phase imbalance : {phase_err_deg:.3f} deg  \"
           f\"({'corrected' if corrected else 'NOT corrected - fix IQ_MODE'})\", pe_col)
    cprint(f\"  Amp imbalance   : {amp_err_pct:.2f}%  \"
           f\"({'corrected' if corrected else 'NOT corrected - fix IQ_MODE'})\", ae_col)

    # -- Robust phase -----------------------------------------------
    phase_clean, carrier_offset = robust_phase(iq)
    co_hz = carrier_offset * FS / (2 * np.pi)

    # -- 50 Hz notch ------------------------------------------------
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag      = signal.filtfilt(b50, a50, mag_raw)

    # -- Bandpass filters -------------------------------------------
    sos_hr   = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
    sos_koro = signal.butter(4, [10,  49 ], btype='band', fs=FS, output='sos')

    phase_hr   = signal.sosfiltfilt(sos_hr,   phase_clean)
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)

    disp_hr    = phase_hr   * SCALE
    disp_koro  = phase_koro * SCALE
    vel_hr     = np.append(np.diff(disp_hr)   * FS, 0)
    vel_koro   = np.append(np.diff(disp_koro) * FS, 0)
    
    # -- Korotkoff window detection ---------------------------------
    sm_energy = (pd.Series(sliding_rms(vel_koro, int(FS*0.3))**2)
                   .rolling(window=int(FS*2), center=True)
                   .mean().fillna(0).values)
    T_SKIP = 5
    vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS))
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0               and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    on_s, off_s = time[si], time[ei]
    dur = off_s - on_s

    # -- Improved Beat detection ------------------------------------
    # Use disp_hr (Phase) for much cleaner beat detection
    t_stable = disp_hr[int(10*FS):int(20*FS)] if len(disp_hr) > int(20*FS) else disp_hr
    pth = np.std(t_stable) * 0.8
    # Search for dips (negative peaks) for heartbeat
    peaks, _ = signal.find_peaks(-disp_hr, distance=int(FS*0.5), prominence=pth)
    
    if len(peaks) > 1:
        iv = np.diff(time[peaks]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_bpm_t = 60.0/np.median(viv) if len(viv)>0 else 0
    else: hr_bpm_t = 0
    
    if on_s < off_s:
        iz = slice(int(on_s*FS), int(off_s*FS))
        f_hr,p_hr = signal.welch(disp_hr[iz], fs=FS, nperseg=min(len(disp_hr[iz]),int(FS*5)))
    else:
        f_hr,p_hr = signal.welch(disp_hr, fs=FS, nperseg=int(FS*10))
    hm = (f_hr>=0.5)&(f_hr<=3.0)
    hr_pk = f_hr[hm][np.argmax(p_hr[hm])] if np.any(hm) else 0
    hr_bpm_f = hr_pk*60

    # -- SNR --------------------------------------------------------
    snr_db = 0
    if on_s < off_s:
        io,ie = int(on_s*FS), int(off_s*FS)
        av = vel_koro[io:ie]; ns = min(ie+int(2*FS), len(vel_koro)-len(av))
        nv = vel_koro[max(0,ns):max(0,ns)+len(av)]
        f_a,p_a = signal.welch(av, fs=FS, nperseg=min(1024,len(av)))
        f_n,p_n = signal.welch(nv, fs=FS, nperseg=min(1024,len(av)))
        km = (f_a>=10)&(f_a<=49)
        if np.any(km) and np.mean(p_n[km])>0: snr_db = 10*np.log10(np.mean(p_a[km])/np.mean(p_n[km]))

    # -- Y-axis limits ----------------------------------------------
    if on_s < off_s:
        io,ie = int(on_s*FS), int(off_s*FS)
        ph_hr_lim    = max(0.05, np.percentile(np.abs(phase_hr[io:ie]),   99.5)*1.5)
        ph_koro_lim  = max(0.01, np.percentile(np.abs(phase_koro[io:ie]), 99.5)*1.5)
        vel_koro_lim = max(10,   np.percentile(np.abs(vel_koro[io:ie]),   99.5)*1.5)
    else: ph_hr_lim, ph_koro_lim, vel_koro_lim = 1.0, 0.2, 200
    disp_hr_lim, disp_koro_lim = ph_hr_lim * SCALE, ph_koro_lim * SCALE

    # ==================== PLOT ====================
    fig, axes = plt.subplots(8, 2, figsize=(22, 40))
    plt.subplots_adjust(hspace=0.55)
    yw = dict(color='yellow', alpha=0.2)
    def kspan(ax): 
        if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')

    # 1. Magnitude
    ax = axes[0,0]
    ax.plot(time, mag, 'b', alpha=0.6); kspan(ax)
    mid = mag[int(T_SKIP*FS):int((time[-1]-T_SKIP)*FS)]; ym = np.percentile(mid, [1,99])
    ax.set_ylim(max(0, ym[0]-0.2*(ym[1]-ym[0])), ym[1]+0.3*(ym[1]-ym[0]))
    ax.set_title(f'1. Magnitude  [IQ_MODE={IQ_MODE}]'); ax.set_ylabel('a.u.')

    # 1b. Phase
    ax = axes[0,1]
    ax.plot(time, phase_koro, 'darkgreen', alpha=0.8, label='Koro Phase')
    ax.plot(time, phase_hr, 'green', alpha=0.4, label='HR Phase'); kspan(ax)
    ax.set_ylim(-max(ph_hr_lim, ph_koro_lim), max(ph_hr_lim, ph_koro_lim))
    ax.set_title('1b. Phase'); ax.legend(fontsize=7)

    # 2a. Disp HR
    ax = axes[1,0]
    ax.plot(time, disp_hr, 'red'); kspan(ax); ax.set_ylim(-disp_hr_lim, disp_hr_lim)
    ax.set_title('2a. Displacement HR'); ax.set_ylabel('mm')

    # 2b. Disp Koro
    ax = axes[1,1]
    ax.plot(time, disp_koro, 'darkred'); kspan(ax); ax.set_ylim(-disp_koro_lim, disp_koro_lim)
    ax.set_title('2b. Displacement Koro'); ax.set_ylabel('mm')

    # 3. Velocity HR & Beats
    ax = axes[2,0]
    vhn = vel_hr/(np.max(np.abs(vel_hr))+1e-9)
    ax.plot(time, vhn, 'blue', alpha=0.7)
    ax.plot(time[peaks], vhn[peaks], 'ro', ms=5, label=f'Beats ({len(peaks)})')
    ax.set_ylim(-1.1,1.1); ax.set_title('3a. Velocity HR - Normalised'); ax.legend(fontsize=7)

    # 3b. Velocity Koro
    ax = axes[2,1]
    ax.plot(time, vel_koro, 'darkred'); kspan(ax); ax.set_ylim(-vel_koro_lim, vel_koro_lim)
    ax.set_title('3b. Velocity Koro'); ax.set_ylabel('mm/s')

    # 4. STFT
    for col,(sig_,lbl_,fb,cm_) in enumerate([(disp_hr,'4a. STFT HR',(0,5),'viridis'),(disp_koro,'4b. STFT Koro',(8,60),'plasma')]):
        f_,t_,Z_ = signal.stft(sig_, fs=FS, nperseg=4096, noverlap=3840)
        P_ = 10*np.log10(np.abs(Z_)**2+1e-20); m_ = (f_>=fb[0])&(f_<=fb[1]); va,vb = np.percentile(P_[m_],[30,99.5])
        ax = axes[3,col]; im = ax.pcolormesh(t_, f_[m_], P_[m_], shading='gouraud', cmap=cm_, vmin=va, vmax=vb)
        ax.set_ylim(*fb); ax.set_title(lbl_); plt.colorbar(im, ax=ax)

    # 5. Kurtosis
    axes[4,0].plot(time, sliding_kurtosis(mag, int(FS*0.5)), 'purple'); axes[4,0].set_title('5a. Mag Kurtosis')
    axes[4,1].plot(time, sliding_kurtosis(vel_koro, int(FS*0.1)), 'purple'); axes[4,1].set_title('5b. Vel Koro Kurtosis')

    # 6. Energy
    axes[5,0].plot(time, sm_energy, 'teal'); axes[5,0].axhline(eth, color='r', ls='--'); axes[5,0].set_title('6a. Koro Energy Envelope')
    axes[5,1].plot(time, sliding_rms(vel_koro, int(FS*0.1)), 'darkred'); axes[5,1].set_title('6b. Vel Jitter RMS')

    # 7. TKEO
    axes[6,0].plot(time, calc_tkeo(vel_koro), 'teal'); axes[6,0].set_title('7a. TKEO Energy')
    axes[6,1].plot(time, np.abs(np.diff(np.append(vel_koro,0)))*FS, 'm'); axes[6,1].set_title('7b. Vel Change Rate')

    # 8. OVERLAYS & ROBUST SCALING
    vz = slice(int(10*FS), int((time[-1]-10)*FS))
    hr_s = np.percentile(np.abs(disp_hr[vz]), 95) + 1e-9
    ko_s = np.percentile(np.abs(vel_koro[vz]), 95) + 1e-9
    zhr, zvl = disp_hr/hr_s, vel_koro/ko_s

    ax = axes[7,0]
    ax.plot(time, zhr, 'k', lw=2.5, label='Heartbeat')
    ax.plot(time, zvl, 'r', alpha=0.5, label='Koro Velocity')
    ax.plot(time[peaks], zhr[peaks], 'ro', ms=4, label='Beats')
    kspan(ax); ax.set_ylim(-3,3); ax.set_title('8a. Full Overlay (Robust Scaling)'); ax.legend(fontsize=7)

    ax = axes[7,1]
    if on_s < off_s:
        zi, ze = max(0, int(on_s*FS)-int(1.5*FS)), min(len(time), int(off_s*FS)+int(1.5*FS))
        zt, lzhr = time[zi:ze], disp_hr[zi:ze]/(np.percentile(np.abs(disp_hr[zi:ze]), 98)+1e-9)
        lzko = vel_koro[zi:ze]/(np.percentile(np.abs(vel_koro[zi:ze]), 98)+1e-9)
        ax.plot(zt, lzhr, 'k', lw=3, label='Heartbeat')
        ax.plot(zt, lzko, 'm', alpha=0.7, label='Koro Velocity')
        zp = peaks[(peaks>=zi)&(peaks<ze)]
        ax.plot(time[zp], lzhr[zp-zi], 'ro', ms=6, label='Beats')
        ax.axvspan(on_s, off_s, **yw); ax.set_xlim(zt[0], zt[-1]); ax.set_title('8b. Zoomed Detail')
    else: ax.set_title('8b. Zoom (No window)')
    ax.set_ylim(-2.5,2.5); ax.legend(fontsize=7)

    for ax in axes.flat: ax.set_xlabel('Time (s)')
    plt.suptitle(f\"Korotkoff Dashboard v7b - USRP B210  |  RF={RF_FREQ_HZ/1e9:.3f}GHz  lambda={LAMBDA_MM:.1f}mm\", fontsize=14, fontweight='bold', y=0.93)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')

    cprint(f\"\n  Plot saved -> {OUTPUT_IMG}\", 'green')
    cprint(f\"  Koro Window  : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)\", 'white')
    cprint(f\"  HR (BPM)     : {hr_bpm_t:.1f} (time) / {hr_bpm_f:.1f} (freq)\", 'white')
    cprint(f\"  SNR          : {snr_db:.1f} dB\", 'green' if snr_db>5 else 'yellow')

if __name__ == '__main__':
    run()
