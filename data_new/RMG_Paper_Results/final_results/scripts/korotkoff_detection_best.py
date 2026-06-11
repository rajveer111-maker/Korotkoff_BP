"""
Best-Practice Korotkoff Detection Analysis
RF Magnitude + Phase vs Stethoscope Ground Truth
5x2 diagnostic figure per subject at 300 DPI
"""
import h5py, os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, find_peaks, welch
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'font.weight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9,  'lines.linewidth': 1.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.6,
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'

# ── HELPERS ──────────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def lpf(x, hi, fs, order=4):
    sos = butter(order, hi, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=35):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

def tkeo(x):
    out = np.zeros_like(x)
    out[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    out[0] = out[1]; out[-1] = out[-2]
    return np.maximum(out, 0)

def smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(x, np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, *_ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def norm01(x):
    xm, xM = np.min(x), np.max(x)
    return (x - xm) / (xM - xm + 1e-12)

def snr_db(env, mask_sig, mask_bas):
    peak  = np.max(env[mask_sig])
    noise = np.mean(env[mask_bas]) + 1e-12
    return 10*np.log10(peak/noise)

# ── DETECTION: Adaptive TKEO peak picking ────────────────────────────────────
def detect_korotkoff_beats(env_n, t, k_on, k_off, min_gap=0.5):
    """
    Pick Korotkoff beats in normalised TKEO envelope.
    Adaptive threshold = 55th percentile of Korotkoff window energy.
    Returns beat times and indices.
    """
    mask = (t >= k_on) & (t <= k_off)
    env_k = env_n.copy(); env_k[~mask] = 0
    thr  = np.percentile(env_k[mask], 55)   # top 45% energy = true Korotkoff clicks
    dist = max(1, int(min_gap / (t[1]-t[0])))
    peaks, _ = find_peaks(env_k, height=thr, distance=dist, prominence=thr*0.6)
    return t[peaks], peaks

# ─────────────────────────────────────────────────────────────────────────────
def run(sub_select):
    cfg = {
        1: dict(
            name='Subject 1 (Prof. Kan)', rec='Rec 06',
            rf=os.path.join(BASE,'Sub_1_Prof_kan','Rec_6.h5'),
            wav=os.path.join(BASE,'Sub_1_Prof_kan','sthethoscope_rec06.wav'),
            out=os.path.join(BASE,'korotkoff_detection_Sub1.png'),
            k_on=27.75, k_off=43.50, defl=18.3, t_max=52.0, lag=1.7083,
            notches=[100.71, 201.43, 302.14, 402.86],
        ),
        2: dict(
            name='Subject 2 (Rajveer)', rec='Rec 04',
            rf=os.path.join(BASE,'Sub_2_Rajveer','Rec_4.h5'),
            wav=os.path.join(BASE,'Sub_2_Rajveer','sthethoscope_rec04.wav'),
            out=os.path.join(BASE,'korotkoff_detection_Sub2.png'),
            k_on=27.375, k_off=42.00, defl=18.6, t_max=51.0, lag=2.6042,
            notches=[50.0, 64.0, 100.6, 201.2],
        ),
    }[sub_select]

    k_on, k_off, defl, t_max, lag = cfg['k_on'], cfg['k_off'], cfg['defl'], cfg['t_max'], cfg['lag']
    notches = cfg['notches']

    FS_RF  = 10_000
    DEC    = 10
    fs     = FS_RF // DEC          # 1 000 Hz working rate
    FC     = 0.9e9
    SCALE  = (299_792_458.0/FC*1000) / (4*np.pi)   # rad → mm

    t_clean_start = defl + 3.0
    t_clean_end   = k_off + 1.5

    # ── 1. LOAD & CONDITION RF ────────────────────────────────────────────────
    print(f"\n[{cfg['name']}] Loading RF …")
    with h5py.File(cfg['rf'], 'r') as f:
        raw = f['data'][:]
    i_raw, q_raw = -raw[0], raw[1]

    xc, yc     = fit_circle(i_raw, q_raw)
    i_c, q_c   = i_raw - xc, q_raw - yc

    # ── 2. BEST PREPROCESSING ─────────────────────────────────────────────────
    #  Phase: robust arc-tan differential + IQR clip + cumsum + linear detrend
    phi_raw = robust_phase(i_c, q_c)

    #  Magnitude: LP 300 Hz then decimate (avoids alias before envelope)
    sos_lp = butter(4, 300., btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))

    #  Notch BOTH channels at every harmonic
    phi_c = phi_raw.copy()
    mag_c = mag_raw.copy()
    for f0 in notches:
        phi_c = notch(phi_c, f0, FS_RF)
        mag_c = notch(mag_c, f0, FS_RF)

    # ── 3. DUAL-BAND EXTRACTION ───────────────────────────────────────────────
    #  3a. Heartbeat compliance (0.4-3 Hz) – arterial wall displacement
    phi_disp = decimate(bpf(phi_c, 0.4, 3.0, FS_RF), DEC, ftype='fir') * SCALE
    mag_disp = decimate(bpf(mag_c, 0.4, 3.0, FS_RF), DEC, ftype='fir') * (SCALE / (np.mean(mag_raw)+1e-9))
    t_ds     = np.arange(len(phi_disp)) / fs

    #  3b. Korotkoff vibration (30-180 Hz) – click / snap velocity
    phi_vk = np.append(np.diff(bpf(phi_c, 30, 180, FS_RF)) * FS_RF, 0) * SCALE
    mag_vk = np.append(np.diff(bpf(mag_c, 30, 180, FS_RF)) * FS_RF, 0)
    t_rf   = np.arange(len(phi_vk)) / FS_RF

    # Zero outside deflation window
    for arr in [phi_vk, mag_vk]:
        arr[(t_rf < t_clean_start) | (t_rf > t_clean_end)] = 0.0
    for arr in [phi_disp, mag_disp]:
        arr[(t_ds < t_clean_start) | (t_ds > t_clean_end)] = 0.0

    #  3c. TKEO energy envelopes (short 0.1s window for temporal resolution)
    mag_tkeo = smooth(tkeo(mag_vk), 0.10, FS_RF)
    phi_tkeo = smooth(tkeo(phi_vk), 0.10, FS_RF)

    #  Decimate envelopes to 1 kHz
    mag_tkeo_ds = decimate(mag_tkeo, DEC, ftype='fir')
    phi_tkeo_ds = decimate(phi_tkeo, DEC, ftype='fir')

    #  3d. Fused envelope: RMS of normalised Mag+Phase TKEO (sensor fusion)
    mask_k  = (t_ds >= k_on) & (t_ds <= k_off)
    mask_b  = (t_ds >= 22.0) & (t_ds <= k_on - 2.0)
    def nrm(e): return (e-np.percentile(e[mask_b],5)) / (np.max(e[mask_k])+1e-12)
    mag_n = np.clip(nrm(mag_tkeo_ds), 0, None)
    phi_n = np.clip(nrm(phi_tkeo_ds), 0, None)
    fused = np.sqrt((mag_n**2 + phi_n**2) / 2)   # L2 fusion

    # ── 4. STETHOSCOPE GT ─────────────────────────────────────────────────────
    print(f"[{cfg['name']}] Loading stethoscope …")
    fs_a, audio = wavfile.read(cfg['wav'])
    audio = audio.astype(np.float64) / 32768.
    if audio.ndim > 1: audio = audio.mean(axis=1)

    audio_bp  = bpf(audio, 50., 1000., fs_a)
    audio_env = smooth(tkeo(audio_bp), 0.10, fs_a)
    t_a = np.arange(len(audio_env)) / fs_a + lag
    audio_env[(t_a < t_clean_start) | (t_a > t_clean_end)] = 0.0
    mask_ka = (t_a >= k_on) & (t_a <= k_off)
    mask_ba = (t_a >= 22.0) & (t_a <= k_on - 2.0)
    steth_n = np.clip(nrm.__class__                         # inline norm
                      and (lambda e,m1,m2:
                           np.clip((e-np.percentile(e[m2],5))/(np.max(e[m1])+1e-12),0,None)
                          )(audio_env, mask_ka, mask_ba),
                 0, None)
    # simpler:
    b_s = np.percentile(audio_env[mask_ba], 5)
    steth_n = np.clip((audio_env - b_s)/(np.max(audio_env[mask_ka])+1e-12), 0, None)

    # Raw waveform (for panel)
    audio_n = audio_bp / (np.max(np.abs(audio_bp[mask_ka]))+1e-10)
    audio_n[(t_a < t_clean_start) | (t_a > t_clean_end)] = 0.

    # ── 5. BEAT DETECTION ─────────────────────────────────────────────────────
    beats_mag,   idx_m = detect_korotkoff_beats(mag_n,   t_ds, k_on, k_off)
    beats_phi,   idx_p = detect_korotkoff_beats(phi_n,   t_ds, k_on, k_off)
    beats_fused, idx_f = detect_korotkoff_beats(fused,   t_ds, k_on, k_off)
    beats_gt,    _     = detect_korotkoff_beats(steth_n, t_a,  k_on, k_off)

    hr_fused = 60./np.mean(np.diff(beats_fused)) if len(beats_fused)>1 else 0
    hr_gt    = 60./np.mean(np.diff(beats_gt))    if len(beats_gt)>1    else 0

    snr_mag  = snr_db(mag_n,  mask_k, mask_b)
    snr_phi  = snr_db(phi_n,  mask_k, mask_b)
    snr_fus  = snr_db(fused,  mask_k, mask_b)

    print(f"  SNR Mag={snr_mag:+.1f} dB  Phi={snr_phi:+.1f} dB  Fused={snr_fus:+.1f} dB")
    print(f"  Beats  Mag={len(beats_mag)}  Phi={len(beats_phi)}  Fused={len(beats_fused)}  GT={len(beats_gt)}")
    print(f"  HR     Fused={hr_fused:.1f} BPM  GT={hr_gt:.1f} BPM")

    # ── 6. FIGURE (5 rows × 2 cols → use 5-row layout) ───────────────────────
    fig = plt.figure(figsize=(22, 26), dpi=300, facecolor='#FFFFFF')
    gs  = fig.add_gridspec(5, 2, hspace=0.52, wspace=0.25,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    CK   = '#F39C12'; CKFILL='#FEF9EC'
    CM   = '#1A6FC4'; CP='#C0392B'; CF='#1B7F4E'; CGT='#2980B9'

    def shade(ax):
        ax.axvspan(k_on, k_off, color=CKFILL, alpha=0.85, zorder=0)
        ax.axvline(k_on,  color=CK, lw=1.2, ls='--', zorder=3, label='_')
        ax.axvline(k_off, color=CK, lw=1.2, ls='--', zorder=3, label='_')
        ax.axvline(defl,  color='#888', lw=0.8, ls=':', zorder=2)

    def label(ax, title, xl='Time (s)', yl='Amplitude'):
        ax.set_title(title, pad=6)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    xl = (max(0,k_on-8), min(t_max,k_off+3))  # zoom window

    # ---------- ROW 0: RAW WAVEFORMS (RF mag velocity vs Steth audio) ---------
    ax = fig.add_subplot(gs[0, 0])
    shade(ax)
    mv = mag_vk / (np.max(np.abs(mag_vk))+1e-10)
    ax.plot(t_rf[::10], mv[::10], color=CM, lw=0.4, alpha=0.6, label='RF Mag velocity (30-180 Hz)')
    ax.set_xlim(xl); ax.set_ylim(-1.3, 1.3)
    label(ax, '(A) RF Magnitude — Korotkoff Raw Velocity (30–180 Hz)', yl='Norm. Amplitude')
    ax.legend(loc='upper left', frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    shade(ax)
    pv = phi_vk / (np.max(np.abs(phi_vk))+1e-10)
    ax.plot(t_rf[::10], pv[::10], color=CP, lw=0.4, alpha=0.6, label='RF Phase velocity (30-180 Hz)')
    ax.set_xlim(xl); ax.set_ylim(-1.3, 1.3)
    label(ax, '(B) RF Phase — Korotkoff Raw Velocity (30–180 Hz)', yl='Norm. Amplitude')
    ax.legend(loc='upper left', frameon=False)

    # ---------- ROW 1: COMPLIANCE PULSES (0.4-3 Hz) ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    shade(ax)
    ax.plot(t_ds, mag_disp, color=CM, lw=1.1, alpha=0.85, label='Mag compliance pulse')
    ax.set_xlim(xl)
    label(ax, f'(C) RF Magnitude — Arterial Compliance (0.4–3 Hz)', yl='Displacement (a.u.)')
    ax.legend(loc='upper left', frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    shade(ax)
    ax.plot(t_ds, phi_disp, color=CP, lw=1.1, alpha=0.85, label='Phase compliance pulse')
    ax.set_xlim(xl)
    label(ax, f'(D) RF Phase — Arterial Compliance (0.4–3 Hz)', yl='Displacement (mm)')
    ax.legend(loc='upper left', frameon=False)

    # ---------- ROW 2: TKEO ENERGY ENVELOPES + beat marks ----------------------
    ax = fig.add_subplot(gs[2, 0])
    shade(ax)
    ax.fill_between(t_ds, mag_n, alpha=0.25, color=CM)
    ax.plot(t_ds, mag_n, color=CM, lw=1.0, label=f'Mag TKEO env (SNR={snr_mag:+.1f} dB)')
    ax.plot(t_ds, phi_n, color=CP, lw=1.0, ls='--', label=f'Phase TKEO env (SNR={snr_phi:+.1f} dB)')
    ax.plot(t_a,  steth_n, color=CGT, lw=0.9, ls=':', alpha=0.8, label='GT steth TKEO')
    for bt in beats_mag: ax.axvline(bt, color=CM, lw=0.7, alpha=0.5)
    for bt in beats_phi: ax.axvline(bt, color=CP, lw=0.7, alpha=0.5, ls='--')
    ax.set_xlim(xl); ax.set_ylim(-0.05, 1.35)
    label(ax, '(E) Normalised TKEO Envelopes — Mag & Phase vs GT', yl='Norm. Energy')
    ax.legend(loc='upper left', frameon=False, ncol=2)

    ax = fig.add_subplot(gs[2, 1])
    shade(ax)
    ax.fill_between(t_ds, fused, alpha=0.3, color=CF)
    ax.plot(t_ds, fused,  color=CF,  lw=1.2, label=f'Fused (Mag+Phase) SNR={snr_fus:+.1f} dB')
    ax.plot(t_a,  steth_n,color=CGT, lw=1.0, ls='--', label='GT Steth TKEO')
    for bt in beats_fused: ax.axvline(bt, color=CF,  lw=1.0, alpha=0.7)
    for bt in beats_gt:    ax.axvline(bt, color=CGT, lw=0.8, alpha=0.6, ls='--')
    ax.set_xlim(xl); ax.set_ylim(-0.05, 1.35)
    label(ax, f'(F) Fused RF vs GT — Beats RF={len(beats_fused)} GT={len(beats_gt)}',
          yl='Norm. Energy')
    ax.legend(loc='upper left', frameon=False)

    # ---------- ROW 3: DETECTED BEAT TIMELINE (raster) -------------------------
    ax = fig.add_subplot(gs[3, 0])
    shade(ax)
    for i,bt in enumerate(beats_mag):   ax.vlines(bt, 2.6,3.4, color=CM,  lw=1.5)
    for i,bt in enumerate(beats_phi):   ax.vlines(bt, 1.6,2.4, color=CP,  lw=1.5)
    for i,bt in enumerate(beats_fused): ax.vlines(bt, 0.6,1.4, color=CF,  lw=1.5)
    for i,bt in enumerate(beats_gt):    ax.vlines(bt,-0.4,0.4, color=CGT, lw=1.5)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(['GT Steth','Fused','Phase','Mag'])
    ax.set_xlim(xl); ax.set_ylim(-0.8,4.0)
    label(ax,'(G) Korotkoff Beat Detection Raster — RF vs GT','Time (s)','Channel')
    ax.grid(axis='x', alpha=0.4); ax.grid(axis='y', alpha=0)

    # ---------- ROW 3 RIGHT: Beat interval comparison --------------------------
    ax = fig.add_subplot(gs[3, 1])
    if len(beats_fused) > 1 and len(beats_gt) > 1:
        ibi_f = np.diff(beats_fused)*1000
        ibi_g = np.diff(beats_gt)*1000
        n = min(len(ibi_f), len(ibi_g))
        ax.plot(range(1,n+1), ibi_f[:n], 'o-', color=CF,  ms=5, label='RF Fused IBI')
        ax.plot(range(1,n+1), ibi_g[:n], 's--',color=CGT, ms=5, label='GT Steth IBI')
    ax.set_xlabel('Beat Index'); ax.set_ylabel('Inter-beat Interval (ms)')
    label(ax,'(H) Inter-Beat Interval: RF Fused vs GT Stethoscope')
    ax.legend(loc='upper right', frameon=False)

    # ---------- ROW 4: PSD KOROTKOFF REGION ------------------------------------
    nperseg = int(FS_RF * 1.0)
    mk_rf = (t_rf >= k_on) & (t_rf <= k_off)
    mb_rf = (t_rf >= k_off+1.5) & (t_rf <= t_max)

    ax = fig.add_subplot(gs[4, 0])
    f_m, Pk_m = welch(mag_vk[mk_rf], fs=FS_RF, nperseg=nperseg)
    f_m, Pb_m = welch(mag_vk[mb_rf], fs=FS_RF, nperseg=nperseg)
    mf = (f_m>=10)&(f_m<=200)
    ax.plot(f_m[mf], 10*np.log10(Pk_m[mf]+1e-20), color=CM, lw=1.1, label='Korotkoff window')
    ax.plot(f_m[mf], 10*np.log10(Pb_m[mf]+1e-20), color='#777', lw=1.0, ls='--', label='Baseline')
    ax.fill_between(f_m[mf],10*np.log10(Pb_m[mf]+1e-20),10*np.log10(Pk_m[mf]+1e-20),
                    alpha=0.15, color=CM)
    label(ax,'(I) RF Magnitude PSD — Korotkoff vs Baseline','Frequency (Hz)','PSD (dB)')
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[4, 1])
    f_p, Pk_p = welch(phi_vk[mk_rf], fs=FS_RF, nperseg=nperseg)
    f_p, Pb_p = welch(phi_vk[mb_rf], fs=FS_RF, nperseg=nperseg)
    pf = (f_p>=10)&(f_p<=200)
    ax.plot(f_p[pf], 10*np.log10(Pk_p[pf]+1e-20), color=CP, lw=1.1, label='Korotkoff window')
    ax.plot(f_p[pf], 10*np.log10(Pb_p[pf]+1e-20), color='#777', lw=1.0, ls='--', label='Baseline')
    ax.fill_between(f_p[pf],10*np.log10(Pb_p[pf]+1e-20),10*np.log10(Pk_p[pf]+1e-20),
                    alpha=0.15, color=CP)
    label(ax,'(J) RF Phase PSD — Korotkoff vs Baseline','Frequency (Hz)','PSD (dB)')
    ax.legend(frameon=False)

    # ── SUPER TITLE ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Best-Practice Korotkoff Detection | {cfg['name']} | {cfg['rec']}\n"
        f"RF Fused: {len(beats_fused)} beats @ {hr_fused:.1f} BPM   "
        f"GT Steth: {len(beats_gt)} beats @ {hr_gt:.1f} BPM   "
        f"Korotkoff window: {k_on:.2f}–{k_off:.2f} s ({k_off-k_on:.1f} s)",
        fontsize=15, fontweight='bold', y=0.975
    )

    plt.savefig(cfg['out'], dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    print(f"  Saved: {cfg['out']}")
    plt.close()


run(1)
run(2)
print("\nAll done.")
