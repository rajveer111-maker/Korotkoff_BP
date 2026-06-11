"""
Korotkoff Window Consensus Detector v2
=======================================
Uses 6 independent detection methods with SUSTAINED-ENERGY detection
(not peak-based) to avoid locking onto transient spikes.

Constraints:
  - Onset  > 10 s from start
  - Offset < 10 s before end  
  - Duration ~ 5-15 s (target ~10 s)
"""
import h5py, numpy as np, os, pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, welch, stft, medfilt
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────
FILE_PATH  = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_consensus_may15.png'
FS_FALLBACK = 10_000
FC_HZ       = 0.9e9
IQ_MODE     = '-I+jQ'

# Constraints
MIN_ONSET_S   = 10.0
MIN_TAIL_S    = 10.0
MIN_DUR_S     = 5.0
MAX_DUR_S     = 18.0

# ── HELPERS ─────────────────────────────────────────────────────
C = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE     = LAMBDA_MM / (4 * np.pi)

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True).mean().fillna(0).values)

def sliding_kurt(x, w):
    return pd.Series(x).rolling(w, center=True).kurt().fillna(0).values

def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def smooth(x, w):
    k = max(1, w)
    return np.convolve(x, np.ones(k)/k, mode='same')

def apply_iq_mode(i, q, mode):
    modes = {'I+jQ': i+1j*q, 'Q+jI': q+1j*i, 'I-jQ': i-1j*q, '-I+jQ': -i+1j*q}
    return modes.get(mode, i+1j*q)

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sin_phi = p3 / np.sqrt(p1*p2 + 1e-20)
    cos_phi = np.sqrt(max(1 - sin_phi**2, 1e-10))
    alpha = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sin_phi,-1,1)))) < 90:
        qc2 = (qc - sin_phi*ic) / (alpha*cos_phi + 1e-15)
    else:
        qc2 = qc
    return ic + 1j*qc2

def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3*iqr, 0.017)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return signal.detrend(phase, type='linear')


def find_sustained_window(curve, time, fs, rec_dur, min_dur=5.0, max_dur=18.0):
    """
    Find the best SUSTAINED high-energy window using a sliding-window
    scoring approach. This avoids transient spike artifacts.
    """
    search_start = int(MIN_ONSET_S * fs)
    search_end   = int((rec_dur - MIN_TAIL_S) * fs)
    if search_end <= search_start + int(min_dur * fs):
        return None

    # Remove impulsive spikes: use large median filter
    spike_win = max(3, int(fs * 0.5)) | 1  # must be odd
    curve_clean = medfilt(curve, kernel_size=min(spike_win, len(curve) if len(curve) % 2 == 1 else len(curve)-1))
    curve_clean = smooth(curve_clean, int(fs * 1.0))

    # Slide a window of varying duration and score by TOTAL energy × duration preference
    # Total energy (sum) naturally favors longer windows over short spikes
    # Gaussian preference centered at 10s with sigma=3s strongly penalizes non-~10s windows
    best_score = -1
    best_on = best_off = 0
    TARGET_DUR = 10.0
    DUR_SIGMA  = 3.0  # Gaussian width in seconds
    
    for dur_test in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        win_samples = int(dur_test * fs)
        dur_weight = np.exp(-0.5 * ((dur_test - TARGET_DUR) / DUR_SIGMA)**2)
        for start in range(search_start, search_end - win_samples, int(fs * 0.25)):
            end = start + win_samples
            if end > search_end:
                break
            window_total = np.sum(curve_clean[start:end])
            score = window_total * dur_weight
            if score > best_score:
                best_score = score
                best_on = time[start]
                best_off = time[min(end, len(time)-1)]

    dur = best_off - best_on
    if dur < 2.0:
        return None
    return {'onset': best_on, 'offset': best_off, 'duration': dur}


def passes_constraints(w, rec_dur):
    if w is None:
        return False
    return (w['onset'] >= MIN_ONSET_S and
            w['offset'] <= rec_dur - MIN_TAIL_S and
            MIN_DUR_S <= w['duration'] <= MAX_DUR_S)


# ── MAIN ────────────────────────────────────────────────────────
def run():
    with h5py.File(FILE_PATH, 'r') as f:
        data = f['data'][:]
        attrs = dict(f.attrs)
    fs = float(attrs.get('sample_rate', attrs.get('fs', FS_FALLBACK)))
    i_raw, q_raw = data[0,:], data[1,:]
    N = len(i_raw)
    time = np.arange(N) / fs
    rec_dur = time[-1]
    print(f"Loaded: {N} samples, {rec_dur:.1f}s, fs={fs:.0f} Hz")

    # IQ conditioning
    iq = b210_iq_condition(apply_iq_mode(i_raw, q_raw, IQ_MODE))
    phase = robust_phase(iq)

    # Notch 50 Hz
    b50, a50 = signal.iirnotch(50.0, 30, fs)

    # Koro band velocity
    sos_koro = butter(4, [10, 49], btype='band', fs=fs, output='sos')
    phase_koro = sosfiltfilt(sos_koro, phase)
    vel_koro = np.append(np.diff(phase_koro)*fs, 0) * SCALE

    # HR band
    sos_hr = butter(4, [0.5, 3.0], btype='band', fs=fs, output='sos')
    disp_hr = sosfiltfilt(sos_hr, phase) * SCALE

    # Magnitude
    mag = signal.filtfilt(b50, a50, np.abs(iq))

    # ── BUILD 6 ENERGY CURVES ──────────────────────────────────
    # M1: Velocity RMS energy
    m1_curve = sliding_rms(vel_koro, int(fs*0.5))**2

    # M2: TKEO energy
    m2_curve = np.abs(calc_tkeo(vel_koro))

    # M3: Sliding kurtosis
    m3_curve = np.clip(sliding_kurt(vel_koro, int(fs*1.0)), 0, None)

    # M4: Hilbert envelope
    m4_curve = np.abs(hilbert(vel_koro))

    # M5: Spectral band-power ratio (sliding PSD)
    win_spec = int(fs * 2.0); step = int(fs * 0.5)
    n_steps = max(1, (N - win_spec) // step)
    bp_ratio = np.zeros(N)
    for idx in range(n_steps):
        s = idx * step; e = s + win_spec
        seg = vel_koro[s:e]
        ff, pp = welch(seg, fs=fs, nperseg=min(1024, len(seg)))
        km = (ff >= 10) & (ff <= 49)
        nm = ((ff >= 2) & (ff < 10)) | ((ff > 49) & (ff <= 80))
        sp = np.mean(pp[km]) if np.any(km) else 1e-20
        np_ = np.mean(pp[nm]) if np.any(nm) else 1e-20
        bp_ratio[s:e] = np.maximum(bp_ratio[s:e], sp / (np_ + 1e-20))
    m5_curve = bp_ratio

    # M6: STFT sub-band integrated energy
    nperseg_s = 2048
    f_stft, t_stft, Zxx = stft(vel_koro, fs=fs, nperseg=nperseg_s, noverlap=nperseg_s*3//4)
    P_stft = np.abs(Zxx)**2
    koro_mask = (f_stft >= 10) & (f_stft <= 49)
    stft_energy = np.mean(P_stft[koro_mask, :], axis=0)
    m6_curve = np.interp(time, t_stft, stft_energy)

    # ── DETECT SUSTAINED WINDOWS ──────────────────────────────
    curves = {
        'M1_VelEnergy':  m1_curve, 'M2_TKEO':       m2_curve,
        'M3_Kurtosis':   m3_curve, 'M4_HilbertEnv': m4_curve,
        'M5_BandPower':  m5_curve, 'M6_STFT':       m6_curve,
    }
    methods = {}
    for name, curve in curves.items():
        print(f"  Detecting {name}...", end=' ')
        w = find_sustained_window(curve, time, fs, rec_dur)
        methods[name] = w
        if w:
            ok = passes_constraints(w, rec_dur)
            print(f"onset={w['onset']:.2f}s  offset={w['offset']:.2f}s  dur={w['duration']:.1f}s  [{'PASS' if ok else 'FAIL'}]")
        else:
            print("No window found [FAIL]")

    # ── CONSENSUS ──────────────────────────────────────────────
    valid = {k: v for k, v in methods.items() if passes_constraints(v, rec_dur)}
    if valid:
        final_on  = float(np.median([v['onset']  for v in valid.values()]))
        final_off = float(np.median([v['offset'] for v in valid.values()]))
    elif methods:
        # Fallback: use all methods that returned something
        has = {k: v for k, v in methods.items() if v is not None}
        if has:
            final_on  = float(np.median([v['onset']  for v in has.values()]))
            final_off = float(np.median([v['offset'] for v in has.values()]))
        else:
            final_on, final_off = 15.0, 25.0
    else:
        final_on, final_off = 15.0, 25.0
    
    final_dur = final_off - final_on
    n_agree = len(valid)

    print(f"\n{'='*60}")
    print(f"  CONSENSUS KOROTKOFF WINDOW")
    print(f"  Onset  : {final_on:.2f} s")
    print(f"  Offset : {final_off:.2f} s")
    print(f"  Duration: {final_dur:.1f} s")
    print(f"  Methods agreeing: {n_agree} / {len(methods)}")
    print(f"{'='*60}")

    # ── HR & SNR ───────────────────────────────────────────────
    io, ie = int(final_on*fs), int(final_off*fs)
    t_stable = disp_hr[int(10*fs):int(20*fs)] if len(disp_hr) > int(20*fs) else disp_hr
    pth = np.std(t_stable) * 0.8
    peaks, _ = signal.find_peaks(-disp_hr, distance=int(fs*0.5), prominence=pth)
    if len(peaks) > 1:
        iv = np.diff(time[peaks]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_bpm = 60.0/np.median(viv) if len(viv) > 0 else 0
    else:
        hr_bpm = 0

    snr_db = 0
    if io < ie:
        av = vel_koro[io:ie]
        nv = vel_koro[:min(int(10*fs), len(vel_koro))]
        npsg = min(1024, len(av), len(nv))
        if npsg > 16:
            fa, pa = welch(av, fs=fs, nperseg=npsg)
            fn, pn = welch(nv, fs=fs, nperseg=npsg)
            km = (fa >= 10) & (fa <= 49)
            if np.any(km) and np.mean(pn[km]) > 0:
                snr_db = 10*np.log10(np.mean(pa[km])/np.mean(pn[km]))

    # ── PLOT (10-panel dashboard) ──────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(24, 32))
    plt.subplots_adjust(hspace=0.50, wspace=0.25)
    yw = dict(color='gold', alpha=0.25)
    def kspan(ax):
        ax.axvspan(final_on, final_off, **yw, label=f'Consensus {final_on:.1f}-{final_off:.1f}s')

    m_colors = {'M1_VelEnergy':'blue','M2_TKEO':'red','M3_Kurtosis':'purple',
                'M4_HilbertEnv':'green','M5_BandPower':'orange','M6_STFT':'cyan'}

    # 1. Velocity + method windows
    ax = axes[0,0]
    ax.plot(time, vel_koro, 'gray', alpha=0.5, lw=0.4)
    for mk, mv in methods.items():
        if mv:
            ok = mk in valid
            ax.axvspan(mv['onset'], mv['offset'], alpha=0.08 if not ok else 0.15,
                       color=m_colors[mk], label=f"{mk} {'✓' if ok else '✗'}")
    kspan(ax)
    ax.set_title('1. Velocity Koro + All Method Windows', fontweight='bold')
    ax.set_ylabel('mm/s'); ax.legend(fontsize=6, ncol=2)

    # 2. Normalised curves
    ax = axes[0,1]
    for (lbl, c), col in zip(curves.items(), m_colors.values()):
        # Smooth & normalize for display
        cs = smooth(c, int(fs*2.0))
        cn = cs / (np.max(cs) + 1e-20)
        ax.plot(time, cn, color=col, alpha=0.7, lw=1.2, label=lbl.split('_')[0]+'_'+lbl.split('_')[1] if '_' in lbl else lbl)
    ax.axvline(final_on, color='k', ls='--', lw=2, label=f'Onset {final_on:.1f}s')
    ax.axvline(final_off, color='k', ls='-.', lw=2, label=f'Offset {final_off:.1f}s')
    ax.set_title('2. Normalised Detection Curves', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # 3. STFT Spectrogram
    ax = axes[1,0]
    P_db = 10*np.log10(P_stft + 1e-20)
    fm = (f_stft >= 5) & (f_stft <= 60)
    va, vb = np.percentile(P_db[fm], [20, 99])
    im = ax.pcolormesh(t_stft, f_stft[fm], P_db[fm], shading='gouraud', cmap='magma', vmin=va, vmax=vb)
    ax.axvline(final_on, color='lime', ls='--', lw=2); ax.axvline(final_off, color='lime', ls='--', lw=2)
    ax.set_title('3. STFT Spectrogram (5-60 Hz)', fontweight='bold'); ax.set_ylabel('Hz')
    plt.colorbar(im, ax=ax, label='dB')

    # 4. STFT HR
    ax = axes[1,1]
    f2, t2, Z2 = stft(disp_hr, fs=fs, nperseg=4096, noverlap=3840)
    P2 = 10*np.log10(np.abs(Z2)**2+1e-20); fmh = (f2>=0)&(f2<=5)
    va2, vb2 = np.percentile(P2[fmh], [30, 99])
    im2 = ax.pcolormesh(t2, f2[fmh], P2[fmh], shading='gouraud', cmap='viridis', vmin=va2, vmax=vb2)
    kspan(ax); ax.set_title('4. STFT HR (0-5 Hz)', fontweight='bold'); ax.set_ylabel('Hz')
    plt.colorbar(im2, ax=ax, label='dB')

    # 5. Displacement HR + beats
    ax = axes[2,0]
    ax.plot(time, disp_hr, 'firebrick', lw=0.8)
    ax.plot(time[peaks], disp_hr[peaks], 'bo', ms=4, label=f'Beats ({len(peaks)})')
    kspan(ax)
    ax.set_title(f'5. Displacement HR — {hr_bpm:.0f} BPM', fontweight='bold')
    ax.set_ylabel('mm'); ax.legend(fontsize=7)

    # 6. Koro zoomed
    ax = axes[2,1]
    pad = int(3*fs)
    zi, ze = max(0, io-pad), min(N, ie+pad)
    ax.plot(time[zi:ze], vel_koro[zi:ze], 'purple', lw=0.5)
    ax.axvspan(final_on, final_off, **yw)
    ax.set_title('6. Zoomed Koro Velocity', fontweight='bold'); ax.set_ylabel('mm/s')

    # 7. Full overlay
    ax = axes[3,0]
    vz = slice(int(10*fs), int((rec_dur-10)*fs))
    hr_s = np.percentile(np.abs(disp_hr[vz]), 95) + 1e-9
    ko_s = np.percentile(np.abs(vel_koro[vz]), 95) + 1e-9
    ax.plot(time, disp_hr/hr_s, 'k', lw=1.5, label='Heartbeat')
    ax.plot(time, vel_koro/ko_s, 'r', alpha=0.5, lw=0.6, label='Koro Vel')
    kspan(ax); ax.set_ylim(-3,3)
    ax.set_title('7. Full Overlay', fontweight='bold'); ax.legend(fontsize=7)

    # 8. Zoomed overlay
    ax = axes[3,1]
    zt = time[zi:ze]
    lhr = disp_hr[zi:ze]/(np.percentile(np.abs(disp_hr[zi:ze]),98)+1e-9)
    lko = vel_koro[zi:ze]/(np.percentile(np.abs(vel_koro[zi:ze]),98)+1e-9)
    ax.plot(zt, lhr, 'k', lw=2, label='Heartbeat')
    ax.plot(zt, lko, 'm', alpha=0.6, label='Koro Vel')
    zp = peaks[(peaks>=zi)&(peaks<ze)]
    if len(zp) > 0:
        ax.plot(time[zp], lhr[zp-zi], 'ro', ms=5, label='Beats')
    ax.axvspan(final_on, final_off, **yw); ax.set_ylim(-2.5,2.5)
    ax.set_title('8. Zoomed Detail', fontweight='bold'); ax.legend(fontsize=7)

    # 9. Method durations bar
    ax = axes[4,0]
    labels, colors_bar, durations = [], [], []
    for mk, mv in methods.items():
        labels.append(mk.replace('_','\n'))
        if mv and passes_constraints(mv, rec_dur):
            colors_bar.append('limegreen'); durations.append(mv['duration'])
        elif mv:
            colors_bar.append('salmon'); durations.append(mv['duration'])
        else:
            colors_bar.append('lightgray'); durations.append(0)
    ax.bar(labels, durations, color=colors_bar, edgecolor='black')
    ax.axhline(10, color='blue', ls='--', label='Target 10s')
    ax.set_title('9. Method Durations', fontweight='bold'); ax.set_ylabel('Duration (s)'); ax.legend(fontsize=7)

    # 10. Summary
    ax = axes[4,1]; ax.axis('off')
    lines = [
        f"KOROTKOFF CONSENSUS REPORT",
        f"{'='*50}",
        f"Recording   : {os.path.basename(FILE_PATH)}",
        f"Duration    : {rec_dur:.1f} s",
        f"Sample Rate : {fs:.0f} Hz",
        f"",
        f"CONSENSUS WINDOW:",
        f"  Onset     : {final_on:.2f} s",
        f"  Offset    : {final_off:.2f} s",
        f"  Duration  : {final_dur:.1f} s",
        f"  Methods OK: {n_agree} / {len(methods)}",
        f"",
        f"VALIDATION:",
        f"  Heart Rate: {hr_bpm:.0f} BPM",
        f"  Koro SNR  : {snr_db:.1f} dB",
        f"  Onset>10s : {'YES' if final_on>=10 else 'NO'}",
        f"  Off<{rec_dur-10:.0f}s  : {'YES' if final_off<=rec_dur-10 else 'NO'}",
        f"  Dur 5-18s : {'YES' if MIN_DUR_S<=final_dur<=MAX_DUR_S else 'NO'}",
        f"{'='*50}",
    ]
    status = 'ALL CONSTRAINTS MET' if (n_agree >= 3 and final_on >= 10 and 
             final_off <= rec_dur-10 and MIN_DUR_S <= final_dur <= MAX_DUR_S) else 'CHECK MANUALLY'
    lines.append(f"  STATUS: {status}")
    ax.text(0.05, 0.95, '\n'.join(lines), fontsize=12, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    for a in axes.flat: a.set_xlabel('Time (s)')
    fig.suptitle(f'Korotkoff Consensus v2 — {os.path.basename(FILE_PATH)}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\nDashboard saved -> {OUTPUT_IMG}")

if __name__ == '__main__':
    run()
