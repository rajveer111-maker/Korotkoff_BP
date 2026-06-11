"""
Best-Session RF Korotkoff Analysis — Publication Figure
========================================================
Automatically selects the highest-SNR recording from each subject,
then produces a comprehensive 5-row × 2-column (10-panel) comparison
figure at 300 DPI with fully labelled axes.

Subjects:
    Sub_1_Prof_kan  (10 recordings)
    Sub_2_Rajveer   (10 recordings)
"""

import h5py, os, glob, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
from scipy.fft import next_fast_len
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'best_session_analysis.png')

SUBJECTS = {
    'Sub_1\n(Prof. Kan)': os.path.join(BASE, 'Sub_1_Prof_kan'),
    'Sub_2\n(Rajveer)':   os.path.join(BASE, 'Sub_2_Rajveer'),
}

FS     = 10_000
FC     = 0.9e9
LAMBDA = (299_792_458.0 / FC) * 1000   # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)          # 26.51 mm/rad
KORO_DUR         = 17.5   # s
STETH_OFFSET     = 3.5    # s after deflation onset

COLORS = ['#00FFFF', '#FF6B6B']   # cyan for Sub1, coral for Sub2
BGFIG  = '#0d1117'
BGAX   = '#161b22'
WHT    = '#F8F8F2'
GOLD   = '#FFD700'
LIME   = '#39FF14'
PURP   = '#BD93F9'

# ── PROCESSING HELPERS ─────────────────────────────────────────────
def iq_condition(iq):
    """
    Fits a circle to the raw IQ constellation to estimate and subtract
    the true clutter DC center (xc, yc), preventing massive phase-wrap artifacts
    during micro-vibration phase unwrapping.
    """
    x, y = iq.real, iq.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    return (x - xc) + 1j * (y - yc)


def robust_phase(iq):
    dp  = np.angle(iq[1:] * np.conj(iq[:-1]))
    h,b = np.histogram(dp, bins=512)
    co  = b[np.argmax(h)] + (b[1]-b[0])/2
    dc  = dp - co
    iqr = np.percentile(dc,75) - np.percentile(dc,25)
    dc  = np.clip(dc, -max(3*iqr,0.017), max(3*iqr,0.017))
    return signal.detrend(np.insert(np.cumsum(dc),0,0.0), type='linear')

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def detect_defl(vk, t, lo=18.0, hi=35, fb=20.0):
    sl,sh = int(lo*FS), int(min(hi*FS, len(vk)))
    if sh <= sl+FS: return fb
    tr  = smooth(np.abs(vk), int(FS*2))
    dt  = np.diff(tr[sl:sh])
    dts = smooth(np.abs(dt), max(1, int(FS*0.5)))
    if dts.max() < 1e-12: return fb
    td  = t[sl + np.argmax(dts)]
    return float(td) if lo<=td<=hi else fb


def process_rec(h5path):
    with h5py.File(h5path,'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0,:], data[1,:]
    N = len(i_raw)
    t = np.arange(N) / FS

    iq  = iq_condition(-i_raw + 1j*q_raw)
    ph  = robust_phase(iq)

    # Korotkoff velocity  10–50 Hz (Clinical RMG fundamental band)
    sos_k = butter(4, [10, 50], btype='band', fs=FS, output='sos')
    pk    = sosfiltfilt(sos_k, ph)
    vk    = np.append(np.diff(pk)*FS, 0) * SCALE   # mm/s



    # Heartbeat displacement  0.4–3 Hz
    sos_h = butter(4, [0.4,3.0], btype='band', fs=FS, output='sos')
    dh    = sosfiltfilt(sos_h, ph) * SCALE          # mm

    # Korotkoff window
    defl  = detect_defl(vk, t)
    k_on  = defl + STETH_OFFSET
    k_on  = max(k_on, 20.0) # Ensure Korotkoff window starts after 20.0 seconds (post-inflation)
    k_off = min(k_on + KORO_DUR, t[-1] - 2.0)


    mask_k    = (t >= k_on)  & (t <= k_off)
    mask_base = (t >= t[-1] - 7.0) & (t <= t[-1] - 2.0) # True quiet post-deflation baseline

    vk_k    = vk[mask_k]
    vk_base = vk[mask_base]
    rms_k   = float(np.sqrt(np.mean(vk_k**2)))    if len(vk_k)>0    else 0.0
    rms_b   = float(np.sqrt(np.mean(vk_base**2))) if len(vk_base)>0 else 1e-6
    snr     = rms_k / (rms_b + 1e-20)
    peak_k  = float(np.max(np.abs(vk_k)))          if len(vk_k)>0    else 0.0

    # Heart rate
    dh_seg = dh[mask_k] if mask_k.any() else dh[mask_base]
    pks,_  = signal.find_peaks(-dh_seg, distance=int(FS*0.5),
                                prominence=np.std(dh_seg)*0.5)
    if len(pks)>1:
        iv  = np.diff(t[mask_k][pks] if mask_k.any() else t[mask_base][pks])
        viv = iv[(iv>0.4)&(iv<1.5)]
        hr  = float(60/np.median(viv)) if len(viv)>0 else 0.0
    else:
        hr = 0.0

    # PSD
    if len(vk_k) > FS:
        f_p, p_p = welch(vk_k, fs=FS, nperseg=min(len(vk_k), int(FS*2)))
        _,   p_b = welch(vk_base, fs=FS, nperseg=min(len(vk_base), int(FS*2)))
    else:
        f_p=np.array([0]); p_p=np.array([0]); p_b=np.array([0])

    # Spectrogram (downsampled to 600 Hz)
    ds_fs = 600
    vk_ds  = signal.resample_poly(vk, up=ds_fs, down=FS)
    t_ds   = np.arange(len(vk_ds)) / ds_fs
    nps    = min(len(vk_ds)//4, int(ds_fs*0.15))
    f_sg,t_sg,Sxx = signal.spectrogram(vk_ds, fs=ds_fs, window='hann',
                                        nperseg=nps, noverlap=nps*7//8, nfft=1024)
    P_db = 10*np.log10(np.sqrt(np.abs(Sxx))+1e-20)

    return dict(
        t=t, vk=vk, dh=dh, ph=ph,
        defl=defl, k_on=k_on, k_off=k_off,
        rms_k=rms_k, rms_b=rms_b, peak_k=peak_k, snr=snr, hr=hr,
        f_psd=f_p, p_psd=p_p, p_base=p_b,
        f_sg=f_sg, t_sg=t_sg, P_db=P_db,
        rec_dur=t[-1], N=N
    )

# ── SELECT BEST SESSION PER SUBJECT ────────────────────────────────
best = {}   # subj_label -> (rec_name, result)
for subj_label, subj_dir in SUBJECTS.items():
    h5_files = sorted(
        glob.glob(os.path.join(subj_dir, 'Rec_*.h5')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
    )
    print(f"\nScanning {subj_label.replace(chr(10),' ')} ...")
    best_snr, best_res, best_name = -1, None, ''
    for h5f in h5_files:
        rname = os.path.splitext(os.path.basename(h5f))[0]
        res   = process_rec(h5f)
        print(f"  {rname}: SNR={res['snr']:.2f}x  RMS={res['rms_k']:.0f} mm/s  HR={res['hr']:.1f} BPM")
        if res['snr'] > best_snr:
            best_snr, best_res, best_name = res['snr'], res, rname
    best[subj_label] = (best_name, best_res)
    print(f"  --> Best: {best_name}  (SNR={best_snr:.2f}x)")

# ── PLOT ───────────────────────────────────────────────────────────
print("\nGenerating publication figure ...")
fig = plt.figure(figsize=(22, 30), dpi=300)
fig.patch.set_facecolor(BGFIG)
gs  = gridspec.GridSpec(5, 2, figure=fig,
                         hspace=0.52, wspace=0.28,
                         left=0.07, right=0.96,
                         top=0.95, bottom=0.04)

subj_labels = list(best.keys())

def styled_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BGAX)
    ax.set_title(title, color=WHT, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, color=WHT, fontsize=10, labelpad=5)
    ax.set_ylabel(ylabel, color=WHT, fontsize=10, labelpad=5)
    ax.tick_params(colors=WHT, labelsize=9, length=4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.6, alpha=0.8, which='both')
    ax.minorticks_on()
    ax.tick_params(which='minor', colors='#30363d', length=2)
    return ax

TBOX = dict(boxstyle='round,pad=0.4', facecolor='#21262d', alpha=0.90, lw=0)

for col, (slabel, color) in enumerate(zip(subj_labels, COLORS)):
    rname, res = best[slabel]
    subj_short = slabel.replace('\n', ' ')
    t   = res['t']
    ds  = max(1, len(t)//8000)
    t_p = t[::ds];  vk_p = res['vk'][::ds];  dh_p = res['dh'][::ds]

    kspan = dict(color=GOLD, alpha=0.15)

    # ── ROW 0: Korotkoff velocity waveform ──────────────────────────
    ax = fig.add_subplot(gs[0, col])
    styled_ax(ax,
              f'{subj_short} — {rname}  |  Korotkoff Velocity (10–50 Hz)',
              xlabel='Time (s)',
              ylabel='Micro-velocity  $v_k(t)$  [mm/s]')
    ax.plot(t_p, vk_p, color=color, lw=0.45, alpha=0.85,
            label='$v_k(t) = \\Delta\\phi_{BP}(t)\\cdot f_s\\cdot$SCALE')
    ax.axvspan(res['k_on'], res['k_off'], **kspan,
               label=f"Korotkoff window\n{res['k_on']:.1f}–{res['k_off']:.1f} s  ({KORO_DUR:.0f} s)")
    ax.axvline(res['defl'], color=LIME, ls=':', lw=1.5,
               label=f"Cuff deflation start  ({res['defl']:.1f} s)")
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT,
              loc='upper right', framealpha=0.85)


    # Stats annotation
    ax.text(0.01, 0.98,
            f"Peak $|v_k|$ = {res['peak_k']:.0f} mm/s\n"
            f"RMS  $v_k$  = {res['rms_k']:.0f} mm/s  (Koro window)\n"
            f"RMS  $v_k$  = {res['rms_b']:.0f} mm/s  (baseline)\n"
            f"SNR = {res['snr']:.2f}×",
            transform=ax.transAxes, fontsize=8.5, va='top',
            color=color, family='monospace', bbox=TBOX)

    # ── ROW 1: Heartbeat displacement ───────────────────────────────
    ax = fig.add_subplot(gs[1, col])
    styled_ax(ax,
              f'{subj_short} — {rname}  |  Tissue Displacement, Heartbeat Band (0.4–3 Hz)',
              xlabel='Time (s)',
              ylabel='Displacement  $d(t) = \\phi_h(t)\\cdot$SCALE  [mm]')
    ax.plot(t_p, dh_p, color=color, lw=0.55, alpha=0.9,
            label='$d(t)$  (heartbeat band 0.4–3 Hz)')
    ax.axvspan(res['k_on'], res['k_off'], **kspan, label='Korotkoff window')
    ax.fill_between(t_p, dh_p, alpha=0.12, color=color)
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT, loc='upper right')
    ax.text(0.01, 0.98,
            f"Heart Rate = {res['hr']:.1f} BPM\n"
            f"$\\lambda$ = {LAMBDA:.1f} mm  |  SCALE = {SCALE:.2f} mm/rad",
            transform=ax.transAxes, fontsize=8.5, va='top',
            color=color, family='monospace', bbox=TBOX)

    # ── ROW 2: Power Spectral Density ───────────────────────────────
    ax = fig.add_subplot(gs[2, col])
    styled_ax(ax,
              f'{subj_short} — {rname}  |  PSD of Korotkoff Velocity',
              xlabel='Frequency  [Hz]',
              ylabel='Power Spectral Density  [dB/Hz]')
    fm = res['f_psd'] <= 100
    if fm.any() and len(res['p_psd'])>1:
        ax.plot(res['f_psd'][fm],
                10*np.log10(res['p_psd'][fm]+1e-20),
                color=color, lw=1.8,
                label=f"Korotkoff window ({res['k_on']:.1f}–{res['k_off']:.1f} s)")
        ax.plot(res['f_psd'][fm],
                10*np.log10(res['p_base'][fm]+1e-20),
                color=PURP, lw=1.2, ls='--', alpha=0.7,
                label=f"Quiet baseline ({res['rec_dur']-7.0:.1f}–{res['rec_dur']-2.0:.1f} s)")
    ax.axvspan(10, 50, color=GOLD, alpha=0.08,
               label='Korotkoff band (10–50 Hz)')
    ax.set_xlim([0, 100])
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT, loc='upper right')

    # ── ROW 3: Spectrogram ──────────────────────────────────────────
    ax = fig.add_subplot(gs[3, col])
    styled_ax(ax,
              f'{subj_short} — {rname}  |  Time–Frequency Spectrogram (10–50 Hz)',
              xlabel='Time  [s]',
              ylabel='Frequency  [Hz]')
    fm_sg = (res['f_sg'] >= 10) & (res['f_sg'] <= 50)
    P_show = res['P_db'][fm_sg]
    va, vb = np.percentile(P_show, [15, 99])
    im = ax.imshow(P_show,
                   extent=[res['t_sg'][0], res['t_sg'][-1],
                           res['f_sg'][fm_sg][0], res['f_sg'][fm_sg][-1]],
                   aspect='auto', origin='lower',
                   cmap='magma', vmin=va, vmax=vb,
                   interpolation='bilinear')
    ax.axvline(res['k_on'],  color=GOLD, ls='--', lw=2.0,
               label=f"Koro onset  {res['k_on']:.1f} s")
    ax.axvline(res['k_off'], color=GOLD, ls='--', lw=2.0,
               label=f"Koro offset {res['k_off']:.1f} s")
    ax.axvline(res['defl'],  color=LIME, ls=':', lw=1.5,
               label=f"Deflation start {res['defl']:.1f} s")
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT,
              loc='upper right', framealpha=0.85)


    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label('Amplitude Spectral Density  [dB]', color=WHT, fontsize=8)
    cb.ax.tick_params(colors=WHT, labelsize=7)

    # ── ROW 4: Statistical summary card ─────────────────────────────
    ax = fig.add_subplot(gs[4, col])
    ax.set_facecolor(BGAX); ax.axis('off')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.set_title(f'{subj_short} — {rname}  |  Measurement Summary',
                 color=WHT, fontsize=11, fontweight='bold', pad=6)

    card = [
        f"Subject         : {subj_short}",
        f"Recording       : {rname}  (best SNR)",
        f"File            : {rname}.h5",
        f"Recording duration : {res['rec_dur']:.1f} s",
        f"Samples         : {res['N']:,}  @ {FS} Hz",
        "",
        "─── Signal Parameters ──────────────────",
        f"Carrier freq fc    : {FC/1e9:.2f} GHz",
        f"Wavelength lambda  : {LAMBDA:.2f} mm",
        f"Scale factor       : {SCALE:.4f} mm/rad",
        f"  (= lambda / 4*pi)",
        "",
        "─── Velocity Derivation ────────────────",
        "  pk[n] = BPF(phi, 10–50 Hz)",
        "  vk[n] = diff(pk)[n] x fs x SCALE",
        f"        = diff(pk)[n] x {FS} x {SCALE:.2f}",

        "",
        "─── Korotkoff Window ───────────────────",
        f"  Cuff deflation start : {res['defl']:.2f} s",
        f"  Korotkoff onset      : {res['k_on']:.2f} s",
        f"  Korotkoff offset     : {res['k_off']:.2f} s",
        f"  Duration             : {res['k_off']-res['k_on']:.2f} s",

        "",
        "─── Velocity Metrics ───────────────────",
        f"  Peak |vk| (Koro win) : {res['peak_k']:.1f} mm/s",
        f"  RMS  |vk| (Koro win) : {res['rms_k']:.1f} mm/s",
        f"  RMS  |vk| (baseline) : {res['rms_b']:.1f} mm/s  (quiet)",
        f"  SNR  Koro / Baseline : {res['snr']:.2f} x",

        "",
        "─── Physiological Metrics ──────────────",
        f"  Heart Rate (detected): {res['hr']:.1f} BPM",
    ]
    ax.text(0.04, 0.97, '\n'.join(card),
            transform=ax.transAxes, fontsize=9,
            family='monospace', color=WHT, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                      alpha=0.95, lw=0))

# ── COLUMN HEADERS ─────────────────────────────────────────────────
fig.text(0.27, 0.965, f"Sub_1  (Prof. Kan)  —  Best Recording: {best[subj_labels[0]][0]}",
         ha='center', color=COLORS[0], fontsize=13, fontweight='bold')
fig.text(0.73, 0.965, f"Sub_2  (Rajveer)  —  Best Recording: {best[subj_labels[1]][0]}",
         ha='center', color=COLORS[1], fontsize=13, fontweight='bold')

fig.suptitle(
    'RF Radar Radiomyography (RMG)  —  Best-Session Korotkoff Analysis\n'
    'USRP B210 @ 0.9 GHz  |  10-50 Hz Korotkoff Band  |  Adaptive Deflation Detection',
    color=WHT, fontsize=14, fontweight='bold', y=0.990)



plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor=BGFIG)
print(f"\nFigure saved -> {OUT}")
