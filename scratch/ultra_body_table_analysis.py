"""
Ultra Acousto-RF: Detailed Body vs Table Analysis
Magnitude AND Phase — All recordings
Saves to: data_new/Ultra/ultra_detailed_analysis/
"""
import h5py, os
import numpy as np
from scipy import signal, stats
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

F0 = 100.714        # ultrasound carrier fundamental (Hz)
ACTIVE_T = (4.0, 8.0)   # active US window seconds

# ── RMG displacement conversion (same as RMG paper) ──────────────────────────
# Carrier: 900 MHz  =>  lambda = c/fc = 299792458/900e6 = 0.33310 m = 333.10 mm
# Round-trip backscatter: d(t) [mm] = phi(t) [rad] * lambda/(4*pi)
# d(t) [um] = phi(t) [rad] * lambda/(4*pi) * 1000
FC   = 900e6                            # Hz
LAMB = 299792458 / FC * 1000           # wavelength in mm  = 333.10 mm
SCALE_MM = LAMB / (4 * np.pi)          # rad -> mm  ~= 26.53 mm/rad
SCALE_UM = SCALE_MM * 1000             # rad -> um  ~= 26525 um/rad

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':10.5,'axes.titleweight':'bold',
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'#E0E0E0','grid.linewidth':0.5,
})

# ── helpers ──────────────────────────────────────────────────────────────────
def load(name):
    with h5py.File(os.path.join(ULTRA, f'{name}.h5'),'r') as f:
        d = f['data'][:]
    return d[0], d[1]   # I, Q

def detect_fs(n_samples, known_dur=None):
    """Estimate FS from known durations or fall back to 45kHz."""
    if known_dur: return int(n_samples / known_dur)
    return 45000

def active_mask(t): return (t >= ACTIVE_T[0]) & (t <= ACTIVE_T[1])

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def lpf(x, hi, fs, order=4):
    sos = butter(order, hi, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def fit_circle(i, q):
    A = np.column_stack([i, q, np.ones_like(i)])
    B = -(i**2 + q**2)
    r,*_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

def phase_diff(i, q):
    """Arc-tangent differential phase (radians) — same method as RMG paper."""
    iq  = i + 1j*q
    dp  = np.angle(iq[1:] * np.conj(iq[:-1]))
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp  = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def phase_to_disp_um(phi):
    """Convert accumulated phase [rad] -> tissue displacement [um].
    Uses RMG paper formula: d = phi * lambda/(4*pi), output in micrometres."""
    return phi * SCALE_UM

def phase_to_disp_mm(phi):
    """Convert accumulated phase [rad] -> tissue displacement [mm]."""
    return phi * SCALE_MM

def mag_env(i, q, fs):
    sos = butter(4, 300., btype='low', fs=fs, output='sos')
    return np.abs(sosfiltfilt(sos, i + 1j*q))

def psd_band(f, pxx, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(pxx[m], f[m]))

def bw_20db(f, pxx, fc):
    """Bandwidth at -20dB below peak near fc."""
    m = (f >= fc-20) & (f <= fc+20)
    fp, pp = f[m], pxx[m]
    pk = np.max(pp)
    thresh = pk * 0.01  # -20 dB
    above = fp[pp >= thresh]
    return float(above[-1] - above[0]) if len(above) > 1 else 0.0

def spectral_entropy(pxx):
    pn = pxx / (np.sum(pxx) + 1e-20)
    return float(-np.sum(pn * np.log(pn + 1e-12)))

def features_from_seg(i_seg, q_seg, fs):
    phi_rad  = phase_diff(i_seg, q_seg)          # radians
    disp_um  = phase_to_disp_um(phi_rad)          # micrometres (RMG conversion)
    mag      = mag_env(i_seg, q_seg, fs)          # V
    results  = {}

    # Phase displacement features (in micrometres)
    for sig, name, unit in [(disp_um,'Disp_um','um'), (mag,'Magnitude','V')]:
        results[f'{name}_RMS_{unit}']   = float(np.sqrt(np.mean(sig**2)))
        results[f'{name}_Std_{unit}']   = float(np.std(sig))
        results[f'{name}_Kurt']         = float(stats.kurtosis(sig, fisher=False))
        results[f'{name}_Skew']         = float(stats.skew(sig))
        results[f'{name}_P2P_{unit}']   = float(np.percentile(sig,95)-np.percentile(sig,5))
        # PSD
        nperseg = min(len(sig), int(fs*1.0))
        f_p, pxx = welch(sig, fs=fs, nperseg=nperseg)
        results[f'{name}_PSD_HB']   = psd_band(f_p, pxx, 0.1, 3.0)   # heartbeat band
        results[f'{name}_PSD_resp'] = psd_band(f_p, pxx, 0.1, 0.6)   # respiration
        results[f'{name}_PSD_hr']   = psd_band(f_p, pxx, 0.6, 3.0)   # heart rate
        results[f'{name}_SpEnt']    = spectral_entropy(pxx)
        for harmonic in [1, 2, 3, 4]:
            fc = harmonic * F0
            if fc < fs/2:
                results[f'{name}_BW20dB_{harmonic}f0'] = bw_20db(f_p, pxx, fc)

    # Also keep raw-phase RMS for ratio comparisons
    results['Phase_rad_RMS'] = float(np.sqrt(np.mean(phi_rad**2)))
    results['Phase_Kurt']    = float(stats.kurtosis(phi_rad, fisher=False))
    nperseg2 = min(len(phi_rad), int(fs*1.0))
    fp2, pxx2 = welch(phi_rad, fs=fs, nperseg=nperseg2)
    results['Phase_PSD_0.1_3Hz'] = psd_band(fp2, pxx2, 0.1, 3.0)
    for harmonic in [1, 2, 3, 4]:
        fc = harmonic * F0
        if fc < fs/2:
            results[f'Phase_BW20dB_{harmonic}f0'] = bw_20db(fp2, pxx2, fc)
    results['Magnitude_RMS'] = float(np.sqrt(np.mean(mag**2)))
    results['Magnitude_Kurt']= float(stats.kurtosis(mag, fisher=False))
    fm, pm = welch(mag, fs=fs, nperseg=nperseg2)
    results['Magnitude_PSD_0.1_3Hz'] = psd_band(fm, pm, 0.1, 3.0)
    return results

# ── PAIRS ────────────────────────────────────────────────────────────────────
PAIRS = [
    ('ultra_rftable1', 'ultra_rfbody1',  'Pair 1'),
    ('ultra_rftable2', 'ultra_rfbody2',  'Pair 2'),
    ('ultra_rftable3', 'ultra_rfbody3',  'Pair 3'),
    ('ultra_rftable4', 'ultra_rfbody01', 'Pair 4'),
]

all_rows = []
summary  = []

for tbl_name, body_name, pair_label in PAIRS:
    print(f"\n{'='*55}\n{pair_label}: {tbl_name} vs {body_name}\n{'='*55}")

    I_t, Q_t = load(tbl_name)
    I_b, Q_b = load(body_name)

    # Estimate FS from known ~10s recordings
    fs_t = detect_fs(len(I_t), 10.0)
    fs_b = detect_fs(len(I_b), 10.0)
    # Use minimum FS for consistency
    fs = min(fs_t, fs_b, 50000)

    t_t = np.arange(len(I_t)) / fs
    t_b = np.arange(len(I_b)) / fs

    # Active window mask
    ma_t = active_mask(t_t)
    ma_b = active_mask(t_b)

    # Correct for DC offset
    xc_t, yc_t = fit_circle(I_t[ma_t], Q_t[ma_t])
    xc_b, yc_b = fit_circle(I_b[ma_b], Q_b[ma_b])
    Ic_t, Qc_t = I_t - xc_t, Q_t - yc_t
    Ic_b, Qc_b = I_b - xc_b, Q_b - yc_b

    # Full signals
    phi_t = phase_diff(Ic_t, Qc_t)
    phi_b = phase_diff(Ic_b, Qc_b)
    mag_t = mag_env(Ic_t, Qc_t, fs)
    mag_b = mag_env(Ic_b, Qc_b, fs)

    # phase_diff uses np.insert(np.cumsum(...),0,0) so output is length N (same as input)
    phi_ta = phi_t[ma_t]
    phi_ba = phi_b[ma_b]
    mag_ta = mag_t[ma_t]
    mag_ba = mag_b[ma_b]
    t_ta   = t_t[ma_t]
    t_ba   = t_b[ma_b]

    # Physiological displacement (0.8-2.5 Hz) in MICROMETRES — RMG paper formula
    # d [um] = phi [rad] * lambda/(4*pi) * 1000  |  lambda=333.10mm at 900MHz
    phi_hb_t = bpf(phi_t, 0.8, 2.5, fs) * SCALE_UM   # um
    phi_hb_b = bpf(phi_b, 0.8, 2.5, fs) * SCALE_UM   # um
    # Korotkoff velocity band (30-180 Hz) displacement in um/s
    phi_vk_t = np.append(np.diff(bpf(phi_t, 30, 180, fs)) * fs, 0) * SCALE_UM
    phi_vk_b = np.append(np.diff(bpf(phi_b, 30, 180, fs)) * fs, 0) * SCALE_UM

    # PSD of full phase (to see harmonic comb)
    nperseg = min(len(phi_t), int(fs*2.0))
    f_t, pxx_t = welch(phi_t, fs=fs, nperseg=nperseg)
    f_b, pxx_b = welch(phi_b, fs=fs, nperseg=nperseg)

    # PSD of active-window only
    f_ta, pxx_ta = welch(phi_ta, fs=fs, nperseg=min(len(phi_ta), int(fs*1.0)))
    f_ba, pxx_ba = welch(phi_ba, fs=fs, nperseg=min(len(phi_ba), int(fs*1.0)))

    # Features
    feat_t = features_from_seg(Ic_t[ma_t], Qc_t[ma_t], fs)
    feat_b = features_from_seg(Ic_b[ma_b], Qc_b[ma_b], fs)

    # Print key metrics
    for key in ['Phase_RMS','Phase_Kurt','Phase_PSD_0.1_3Hz','Phase_BW20dB_1f0',
                'Magnitude_RMS','Magnitude_Kurt','Magnitude_PSD_0.1_3Hz']:
        vt = feat_t.get(key, 0); vb = feat_b.get(key, 0)
        ratio = vb/(vt+1e-20)
        print(f"  {key:<30} Table={vt:.4e}  Body={vb:.4e}  Ratio={ratio:.2f}x")

    # Store CSV
    row_t = {'pair': pair_label, 'condition':'Table', 'file': tbl_name}
    row_t.update(feat_t)
    row_b = {'pair': pair_label, 'condition':'Body',  'file': body_name}
    row_b.update(feat_b)
    all_rows += [row_t, row_b]

    # Summary for report
    summary.append({
        'pair': pair_label, 'table': tbl_name, 'body': body_name,
        'Phase_RMS_ratio':  feat_b.get('Phase_RMS',0)   / (feat_t.get('Phase_RMS',1e-20)),
        'Phase_Kurt_Body':  feat_b.get('Phase_Kurt',0),
        'Phase_Kurt_Table': feat_t.get('Phase_Kurt',0),
        'Mag_RMS_ratio':    feat_b.get('Magnitude_RMS',0)/(feat_t.get('Magnitude_RMS',1e-20)),
        'Phase_PSD_HB_ratio': feat_b.get('Phase_PSD_0.1_3Hz',0)/(feat_t.get('Phase_PSD_0.1_3Hz',1e-20)),
        'BW_1f0_Table': feat_t.get('Phase_BW20dB_1f0',0),
        'BW_1f0_Body':  feat_b.get('Phase_BW20dB_1f0',0),
    })

    # ── FIGURE: 8 rows × 2 cols ───────────────────────────────────────────
    CT='#2C3E50'; CB='#C0392B'; CM='#1A6FC4'; CK='#F39C12'
    fig = plt.figure(figsize=(22, 30), dpi=300, facecolor='#FFFFFF')
    gs  = gridspec.GridSpec(8, 2, hspace=0.52, wspace=0.28,
                            left=0.07, right=0.97, top=0.96, bottom=0.02)

    def ann(ax, title, xl='Time (s)', yl=''):
        ax.set_title(title, pad=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        for v in [ACTIVE_T[0], ACTIVE_T[1]]:
            ax.axvline(v, color=CK, lw=1.0, ls='--', zorder=3)

    # ROW 0: Raw I/Q waveform (first 2s)
    for col,(IC,QC,t,clr,lbl) in enumerate([(Ic_t,Qc_t,t_t,CT,'Table'),(Ic_b,Qc_b,t_b,CB,'Body')]):
        ax=fig.add_subplot(gs[0,col])
        m2=(t<=2.0)
        ax.plot(t[m2],IC[m2],color=clr,lw=0.5,alpha=0.8,label='I channel')
        ax.plot(t[m2],QC[m2],color=clr,lw=0.5,alpha=0.5,ls='--',label='Q channel')
        ax.set_title(f'({"A" if col==0 else "B"}) {lbl} — Raw I/Q Signal [0-2s zoom]')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (V)')
        ax.legend(frameon=False)

    # ROW 1: Magnitude envelope (full)
    for col,(mg,t,clr,lbl) in enumerate([(mag_t,t_t,CT,'Table'),(mag_b,t_b,CB,'Body')]):
        ax=fig.add_subplot(gs[1,col])
        ax.fill_between(t,0,mg,color=clr,alpha=0.35)
        ax.plot(t,mg,color=clr,lw=0.5)
        ann(ax,f'({"C" if col==0 else "D"}) {lbl} — RF Magnitude Envelope [full]','Time (s)','Amplitude (V)')
        ax.text(ACTIVE_T[0]+0.1,np.max(mg)*0.92,'Active US window',color=CK,fontsize=8)

    # ROW 2: Phase->Displacement (full) in micrometres
    for col,(ph,t,clr,lbl) in enumerate([(phi_t,t_t,CT,'Table'),(phi_b,t_b,CB,'Body')]):
        ax=fig.add_subplot(gs[2,col])
        disp_full = ph * SCALE_UM          # radians -> micrometres
        ax.plot(t, disp_full, color=clr, lw=0.5, alpha=0.85)
        ann(ax, f'({"E" if col==0 else "F"}) {lbl} — Tissue Displacement [full recording]\n'
                f'd(t) = phi(t) x lambda/(4pi)   [lambda=333mm @ 900MHz]',
            'Time (s)', 'Displacement (um)')
        peak = float(np.percentile(np.abs(disp_full), 99))
        ax.text(0.02, 0.92, f'P99={peak:.1f} um', transform=ax.transAxes,
                fontsize=9, color=clr)

    # ROW 3: Heartbeat displacement (0.8-2.5 Hz) — active window in micrometres
    for col,(hb,t,clr,lbl) in enumerate([(phi_hb_t,t_t,CT,'Table'),(phi_hb_b,t_b,CB,'Body')]):
        ax=fig.add_subplot(gs[3,col])
        ma=(t>=ACTIVE_T[0])&(t<=ACTIVE_T[1])
        ax.plot(t[ma], hb[ma], color=clr, lw=1.2)
        ax.set_title(f'({"G" if col==0 else "H"}) {lbl} — Arterial Displacement 0.8-2.5 Hz [active window]\n'
                     f'Expected: ~500-1000 um for arterial pulse (per RMG paper)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement (um)')
        rms = float(np.sqrt(np.mean(hb[ma]**2)))
        p2p = float(np.percentile(hb[ma],95) - np.percentile(hb[ma],5))
        ax.text(0.02, 0.92, f'RMS={rms:.1f} um   P2P={p2p:.1f} um',
                transform=ax.transAxes, fontsize=9, color=clr)

    # ROW 4: PSD comparison Phase (log y)
    ax=fig.add_subplot(gs[4,:])
    mf=(f_ta>=0.5)&(f_ta<=5.0)
    ax.semilogy(f_ta[mf],pxx_ta[mf],color=CT,lw=1.1,label=f'Table (active window)')
    ax.semilogy(f_ba[mf],pxx_ba[mf],color=CB,lw=1.1,ls='--',label=f'Body (active window)')
    ax.fill_between(f_ba[mf],pxx_ta[mf],pxx_ba[mf],
                    where=pxx_ba[mf]>pxx_ta[mf],color=CB,alpha=0.2,label='Body excess (heartbeat band)')
    ax.set_title('(I) Phase PSD — 0.5-5 Hz [Heartbeat & Respiration Band]  Table vs Body')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (log scale)')
    ax.legend(frameon=False)

    # ROW 5: PSD — Harmonic region (50-550 Hz)
    ax=fig.add_subplot(gs[5,:])
    mf2=(f_t>=50)&(f_t<=550)
    ax.semilogy(f_t[mf2],pxx_t[mf2],color=CT,lw=0.9,label='Table Phase PSD')
    ax.semilogy(f_b[mf2],pxx_b[mf2],color=CB,lw=0.9,ls='--',label='Body Phase PSD')
    for n in [1,2,3,4,5]:
        fc=n*F0
        if fc<np.max(f_t[mf2]):
            ax.axvline(fc,color=CK,lw=0.8,ls=':',alpha=0.8)
            ax.text(fc+1,ax.get_ylim()[1]*0.6,f'{n}f₀\n{fc:.0f}Hz',fontsize=7,color=CK)
    ax.set_title('(J) Phase PSD — 50-550 Hz [Harmonic Comb Region]  Table vs Body')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (log scale)')
    ax.legend(frameon=False)

    # ROW 6: Spectral broadening zoom around 1f0
    ax=fig.add_subplot(gs[6,0])
    fc1=F0
    mf3=(f_t>=fc1-5)&(f_t<=fc1+5)
    if np.sum(mf3)>2:
        ax.plot(f_t[mf3],10*np.log10(pxx_t[mf3]+1e-20),color=CT,lw=1.1,label='Table')
        ax.plot(f_b[mf3],10*np.log10(pxx_b[mf3]+1e-20),color=CB,lw=1.1,ls='--',label='Body')
        ax.fill_between(f_b[mf3],10*np.log10(pxx_t[mf3]+1e-20),
                        10*np.log10(pxx_b[mf3]+1e-20),
                        where=pxx_b[mf3]>pxx_t[mf3],color=CB,alpha=0.25,label='Body broadening')
    ax.set_title(f'(K) Phase PSD Zoom — 1f₀ ({fc1:.1f} Hz) ±5 Hz\nLine Broadening = Physiological Modulation')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (dB)')
    ax.legend(frameon=False)

    # Broadening zoom around 4f0
    ax=fig.add_subplot(gs[6,1])
    fc4=4*F0
    mf4=(f_t>=fc4-5)&(f_t<=fc4+5)
    if np.sum(mf4)>2:
        ax.plot(f_t[mf4],10*np.log10(pxx_t[mf4]+1e-20),color=CT,lw=1.1,label='Table')
        ax.plot(f_b[mf4],10*np.log10(pxx_b[mf4]+1e-20),color=CB,lw=1.1,ls='--',label='Body')
        ax.fill_between(f_b[mf4],10*np.log10(pxx_t[mf4]+1e-20),
                        10*np.log10(pxx_b[mf4]+1e-20),
                        where=pxx_b[mf4]>pxx_t[mf4],color=CB,alpha=0.25,label='Body broadening')
    ax.set_title(f'(L) Phase PSD Zoom — 4f₀ ({fc4:.1f} Hz) ±5 Hz\nHigher Harmonic Amplification')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (dB)')
    ax.legend(frameon=False)

    # ROW 7: Feature comparison bar chart — Table vs Body
    feat_keys  = ['Phase_RMS','Phase_Kurt','Phase_PSD_0.1_3Hz',
                  'Magnitude_RMS','Magnitude_Kurt','Magnitude_PSD_0.1_3Hz']
    feat_labs  = ['Phase\nRMS','Phase\nKurt','Phase\nHB PSD',
                  'Mag\nRMS','Mag\nKurt','Mag\nHB PSD']
    vt_arr = np.array([feat_t.get(k,0) for k in feat_keys])
    vb_arr = np.array([feat_b.get(k,0) for k in feat_keys])
    mx_arr = np.maximum(np.abs(vt_arr), np.abs(vb_arr)) + 1e-20
    vt_n   = vt_arr/mx_arr; vb_n = vb_arr/mx_arr

    ax=fig.add_subplot(gs[7,:])
    x=np.arange(len(feat_keys))
    ax.bar(x-0.2,vt_n,0.38,label='Table (control)',color=CT,alpha=0.8)
    ax.bar(x+0.2,vb_n,0.38,label='Body (active)',  color=CB,alpha=0.8)
    for i,(vt,vb) in enumerate(zip(vt_arr,vb_arr)):
        r=vb/(vt+1e-20)
        ax.text(i+0.2,vb_n[i]+0.02,f'{r:.1f}x',ha='center',fontsize=8,color=CB,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(feat_labs)
    ax.set_title('(M) Feature Comparison: Table vs Body — All Metrics (normalised to max)\nAnnotation = Body/Table ratio')
    ax.set_ylabel('Relative Value'); ax.legend(frameon=False)

    fig.suptitle(f"Acousto-RF Ultra Analysis: {pair_label}\n"
                 f"{tbl_name} (Table/Control)  vs  {body_name} (Body/Active)\n"
                 f"Active US Window: {ACTIVE_T[0]}–{ACTIVE_T[1]} s  |  f₀={F0} Hz",
                 fontsize=13, fontweight='bold', y=0.978)
    outf=os.path.join(OUT,f'ultra_detailed_{pair_label.replace(" ","")}.png')
    plt.savefig(outf,dpi=300,facecolor='#FFFFFF',bbox_inches='tight')
    print(f"  Saved: {outf}")
    plt.close()

# ── Summary figure: all 4 pairs side by side ─────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(24, 14), dpi=300, facecolor='#FFFFFF')
plt.rcParams['axes.titlesize']=9
titles=['Phase RMS Ratio\n(Body/Table)',
        'Phase HB PSD Ratio\n(Body/Table, 0.1-3Hz)',
        'Phase Kurtosis\n(Body blue, Table grey)',
        'Mag RMS Ratio\n(Body/Table)']
vals  =[['Phase_RMS_ratio'],['Phase_PSD_HB_ratio'],
        ['Phase_Kurt_Body','Phase_Kurt_Table'],['Mag_RMS_ratio']]

for pi, row in enumerate(summary):
    CB='#C0392B'; CT='#2C3E50'
    # Phase RMS ratio
    axes[0,pi].bar(['Table','Body'],[1.0, row['Phase_RMS_ratio']],
                   color=[CT,CB],alpha=0.8)
    axes[0,pi].set_title(f"{row['pair']}\nPhase RMS Ratio = {row['Phase_RMS_ratio']:.1f}x")
    axes[0,pi].set_ylabel('Ratio (Table=1)')
    # HB PSD ratio
    axes[1,pi].bar(['Table','Body'],[1.0, row['Phase_PSD_HB_ratio']],
                   color=[CT,CB],alpha=0.8)
    axes[1,pi].set_title(f"HB Band PSD Ratio = {row['Phase_PSD_HB_ratio']:.1f}x")
    axes[1,pi].set_ylabel('Ratio (Table=1)')
    # Spectral BW
    axes[2,pi].bar(['Table BW\n1f0','Body BW\n1f0'],
                   [row['BW_1f0_Table'],row['BW_1f0_Body']],
                   color=[CT,CB],alpha=0.8)
    axes[2,pi].set_title(f"BW@-20dB 1f0\nT={row['BW_1f0_Table']:.2f}Hz  B={row['BW_1f0_Body']:.2f}Hz")
    axes[2,pi].set_ylabel('Bandwidth (Hz)')

plt.suptitle("Acousto-RF Summary: 4 Body vs Table Pairs\nPhase & Magnitude Changes in Active Ultrasound Window",
             fontsize=14,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'ultra_summary_all_pairs.png'),
            dpi=300,facecolor='#FFFFFF',bbox_inches='tight')
plt.close()
print(f"Summary figure saved.")

# ── CSV ──────────────────────────────────────────────────────────────────────
if all_rows:
    keys = ['pair','condition','file'] + \
           [k for k in all_rows[0] if k not in ('pair','condition','file')]
    with open(os.path.join(OUT,'ultra_features.csv'),'w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(all_rows)
    print(f"CSV saved: {os.path.join(OUT,'ultra_features.csv')}")

print("\nAll done.")
