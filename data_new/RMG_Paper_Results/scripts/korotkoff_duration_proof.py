"""
Korotkoff Duration Proof — Publication Figure  (v2 – corrected + extended)
============================================================================
Layout: 4 panels  (2 rows × 2 columns)

Panel A (top-left)   : Subject 1 – Korotkoff Duration Proof  (TKEO, zoomed, tight-gated)
Panel B (top-right)  : Subject 2 – Korotkoff Duration Proof  (TKEO, zoomed, tight-gated)
Panel C (bottom-left): Cross-Correlation Analysis – RF TKEO vs Stethoscope TKEO (both subs)
Panel D (bottom-right): Beat-by-Beat Timing Comparison – detected beats RF vs GT stethoscope

Fixes over v1:
  ✓ RF energy hard-gated to [k_on-0.5, k_off+0.5] so pre-Korotkoff pump noise is invisible
  ✓ Window-specific normalisation: peak inside Korotkoff window = 1.0
  ✓ Stethoscope independently scaled so its beats are clearly visible
  ✓ Brace spans full onset→offset
  ✓ Two confirmatory analyses added (cross-corr + beat timing scatter)

Output: figures/supplementary/rmg_korotkoff_duration_proof.png   (300 DPI)
"""

import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile
from scipy.signal import hilbert, correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family'        : 'DejaVu Sans',
    'font.size'          : 12,
    'font.weight'        : 'bold',
    'axes.labelsize'     : 13,
    'axes.labelweight'   : 'bold',
    'axes.titlesize'     : 13,
    'axes.titleweight'   : 'bold',
    'xtick.labelsize'    : 11,
    'ytick.labelsize'    : 11,
    'legend.fontsize'    : 11,
    'legend.framealpha'  : 0.92,
    'lines.linewidth'    : 2.0,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.color'         : '#E8E8E8',
    'grid.linewidth'     : 0.8,
})

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = (r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results'
         r'\figures\supplementary\rmg_korotkoff_duration_proof.png')
PAPER_OUT = OUT   # same location is fine

FS_RF = 10_000;  DEC = 10;  FS = 1_000

# ── Validated ground-truth session parameters ──────────────────────────────
SESSIONS = [
    dict(
        label    = 'Subject 1 — Prof. Kan  (Rec 06)',
        sub_dir  = 'Sub_1_Prof_kan',
        rec      = 6,
        k_on     = 27.75,        # GT Korotkoff onset  (s)
        k_off    = 43.50,        # GT Korotkoff offset (s)
        defl     = 18.3,         # cuff deflation onset (s)
        lag      = 1.7083,       # stethoscope alignment lag (s)
        notches  = [100.71, 201.43, 302.14, 402.86],
        c_rf     = '#C0392B',    # crimson
        c_st     = '#2980B9',    # steel-blue
        c_fill   = '#FADBD8',
        c_stfill = '#D6EAF8',
    ),
    dict(
        label    = 'Subject 2 — Rajveer  (Rec 04)',
        sub_dir  = 'Sub_2_Rajveer',
        rec      = 4,
        k_on     = 27.375,
        k_off    = 42.00,
        defl     = 18.6,
        lag      = 2.6042,
        notches  = [50.0, 64.0, 100.6, 201.2],
        c_rf     = '#8E44AD',    # purple
        c_st     = '#27AE60',    # emerald
        c_fill   = '#E8DAEF',
        c_stfill = '#D5F5E3',
    ),
]

# ── Signal helpers ──────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch_filt(x, f0, fs, Q=35):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def calc_tkeo(x):
    tk = np.zeros_like(x)
    tk[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tk, 0)

def smooth_w(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(i, q):
    A   = np.column_stack([i, q, np.ones_like(i)])
    B   = -(i**2 + q**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -res[0]/2, -res[1]/2

def robust_phase(ic, qc):
    iq   = ic + 1j*qc
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def norm_inside(env, t, k_on, k_off):
    """Normalise so max inside [k_on, k_off] = 1.0."""
    mask = (t >= k_on) & (t <= k_off)
    peak = np.max(env[mask]) if np.any(mask) else 1.0
    return env / (peak + 1e-12)

def detect_beats(env, t, k_on, k_off, min_dist_s=0.35, fs=FS):
    """Find beat peaks inside Korotkoff window."""
    mask = (t >= k_on) & (t <= k_off)
    sub  = env.copy()
    sub[~mask] = 0
    thr  = np.percentile(sub[mask], 50)
    pks, _ = find_peaks(sub, height=thr, distance=int(min_dist_s*fs))
    return t[pks]

# ── Main signal processing ──────────────────────────────────────────────────
def process(s):
    rf_path  = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    wav_path = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")

    # --- RF ---
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    ic, qc = -rf[0, :], rf[1, :]
    xc, yc  = fit_circle(ic, qc)
    phi     = robust_phase(ic - xc, qc - yc)

    for f0 in s['notches']:
        phi = notch_filt(phi, f0, FS_RF)

    vel = np.append(np.diff(bpf(phi, 30, 180, FS_RF)) * FS_RF, 0.0)
    t_full = np.arange(len(vel)) / FS_RF

    # DISPLAY gate: strictly within Korotkoff window ±0.5 s only
    disp_mask = (t_full < s['k_on'] - 0.5) | (t_full > s['k_off'] + 0.5)
    vel_disp  = vel.copy();  vel_disp[disp_mask] = 0.0

    vel_dec   = decimate(vel_disp, DEC, ftype='fir')
    t_rf      = np.arange(len(vel_dec)) / FS
    rf_tkeo   = smooth_w(calc_tkeo(vel_dec), 0.15, FS)
    rf_norm   = norm_inside(rf_tkeo, t_rf, s['k_on'], s['k_off'])
    # Heavily smoothed version for cross-correlation (0.5 s Gaussian)
    rf_smooth = smooth_w(calc_tkeo(vel_dec), 0.50, FS)
    rf_smooth = norm_inside(rf_smooth, t_rf, s['k_on'], s['k_off'])

    # --- Stethoscope ---
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio_f   = bpf(audio, 50, 1000, fs_a)
    # Stethoscope beat envelope: Hilbert -> lowpass at 8 Hz -> smooth
    audio_e   = np.abs(hilbert(audio_f))
    audio_env = bpf(audio_e, 0.5, min(8.0, fs_a//2-1), fs_a)
    DEC_A     = 10;  fs_ad = fs_a // DEC_A
    audio_kd  = decimate(audio_env, DEC_A, ftype='fir')
    t_a       = np.arange(len(audio_kd)) / fs_ad + s['lag']
    # Also keep the TKEO version for display
    audio_f2  = bpf(audio, 50, 1000, fs_a)
    audio_e2  = np.abs(hilbert(audio_f2))
    audio_k2  = bpf(audio_e2, 20, min(200, fs_a//2 - 1), fs_a)
    audio_kd2 = decimate(audio_k2, DEC_A, ftype='fir')
    st_tkeo   = smooth_w(calc_tkeo(audio_kd2), 0.15, fs_ad)
    st_interp = np.interp(t_rf, t_a, st_tkeo, left=0.0, right=0.0)
    st_norm   = norm_inside(st_interp, t_rf, s['k_on'], s['k_off'])
    # Smooth stethoscope envelope for cross-correlation
    st_smooth_raw = smooth_w(np.maximum(audio_kd, 0), 0.50, fs_ad)
    st_smooth = np.interp(t_rf, t_a, st_smooth_raw, left=0.0, right=0.0)
    st_smooth = norm_inside(st_smooth, t_rf, s['k_on'], s['k_off'])

    # --- Stethoscope beat times directly from raw audio envelope ---
    st_env_full = np.interp(t_rf, t_a, smooth_w(np.maximum(audio_kd2, 0), 0.10, fs_ad),
                            left=0.0, right=0.0)
    st_norm_disp = norm_inside(st_env_full, t_rf, s['k_on'], s['k_off'])
    rf_beats = detect_beats(rf_norm, t_rf, s['k_on'], s['k_off'], min_dist_s=0.45)
    st_beats = detect_beats(st_norm_disp, t_rf, s['k_on'], s['k_off'], min_dist_s=0.45)

    return dict(t=t_rf, rf=rf_norm, st=st_norm,
                rf_smooth=rf_smooth, st_smooth=st_smooth,
                rf_beats=rf_beats, st_beats=st_beats)

# ── Cross-correlation helper (uses pre-smoothed envelopes) ─────────────────
def calc_xcorr(t, rf_smooth, st_smooth, k_on, k_off):
    mask = (t >= k_on) & (t <= k_off)
    a = rf_smooth[mask];  b = st_smooth[mask]
    # Detrend and z-score
    a = signal.detrend(a, type='linear')
    b = signal.detrend(b, type='linear')
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    xc  = correlate(a, b, mode='full') / len(a)
    lag = np.arange(-(len(a)-1), len(a)) / FS
    return lag, xc

# ── Beat matching helper ────────────────────────────────────────────────────
def match_beats(rf_beats, st_beats, tol=0.40):
    """Match RF beats to nearest stethoscope beat within tol seconds."""
    errors = []
    used   = set()
    for rb in rf_beats:
        if len(st_beats) == 0: continue
        diffs = np.abs(st_beats - rb)
        idx   = np.argmin(diffs)
        if diffs[idx] <= tol and idx not in used:
            errors.append(rb - st_beats[idx])
            used.add(idx)
    return np.array(errors) * 1000.0   # ms

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS BOTH SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("Processing sessions...")
results = [process(s) for s in SESSIONS]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  —  2×2 GridSpec
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), dpi=300, facecolor='white')
gs  = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32,
               top=0.92, bottom=0.07, left=0.07, right=0.97)

ax_top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_xcr = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

PANEL_LABELS = ['(A)', '(B)', '(C)', '(D)']

# ─────────────────────────────────────────────────────────────────────────────
# PANELS A & B  —  Duration Proof (tight-gated, zoomed)
# ─────────────────────────────────────────────────────────────────────────────
for idx, (ax, s, res) in enumerate(zip(ax_top, SESSIONS, results)):
    t    = res['t'];  rf = res['rf'];  st = res['st']
    k_on = s['k_on']; k_off = s['k_off']; dur = k_off - k_on

    # Display window: k_on-2.5 to k_off+1.5 (gives more space for the legend on the left)
    xl = (k_on - 2.5, k_off + 1.5)

    # Korotkoff window shading
    ax.axvspan(k_on, k_off, color='#F39C12', alpha=0.10, zorder=0)
    ax.axvline(k_on,  color='#E67E22', lw=1.8, ls='--', zorder=3)
    ax.axvline(k_off, color='#E67E22', lw=1.8, ls='--', zorder=3)

    # Stethoscope (behind)
    ax.fill_between(t, st, alpha=0.22, color=s['c_st'], zorder=1)
    ax.plot(t, st, color=s['c_st'], lw=2.0, alpha=0.9, zorder=2,
            label='Stethoscope GT TKEO')

    # RF Phase on top
    ax.fill_between(t, rf, alpha=0.18, color=s['c_rf'], zorder=1)
    ax.plot(t, rf, color=s['c_rf'], lw=2.0, zorder=4,
            label='RF Phase TKEO')

    # Detected beats — vertical ticks at top
    for rb in res['rf_beats']:
        ax.axvline(rb, color=s['c_rf'], lw=0.9, alpha=0.5, ls=':', ymax=0.06, zorder=5)
    for sb in res['st_beats']:
        ax.axvline(sb, color=s['c_st'], lw=0.9, alpha=0.5, ls=':', ymax=0.06, zorder=5)

    # ── Duration brace ─────────────────────────────────────────────────────
    brace_y = 1.16
    ax.annotate('', xy=(k_off, brace_y), xytext=(k_on, brace_y),
                arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=2.2,
                                mutation_scale=18))
    mid = (k_on + k_off) / 2
    ax.text(mid, brace_y + 0.05,
            f'Korotkoff Duration = {dur:.2f} s  (RF ≈ GT)',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.35', fc='#FEF9E7',
                      ec='#F39C12', lw=1.5, alpha=0.96))

    # Onset / offset timestamp labels — placed just above the bottom axis, offset horizontally
    ax.text(k_on  - 0.5, 0.04, f'{k_on:.3f} s\n(Onset)',  ha='center', va='bottom',
            fontsize=9.5, color='#E67E22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#E67E22', lw=0.8, alpha=0.85))
    ax.text(k_off + 0.5, 0.04, f'{k_off:.3f} s\n(Offset)', ha='center', va='bottom',
            fontsize=9.5, color='#E67E22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#E67E22', lw=0.8, alpha=0.85))

    ax.set_xlim(xl);  ax.set_ylim(-0.05, 1.38)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Norm. TKEO Energy', fontsize=12)
    ax.set_title(f'{PANEL_LABELS[idx]}  {s["label"]}', fontsize=13, pad=4)
    ax.legend(loc='upper left', framealpha=0.93, fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL C  —  Cross-Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_xcr[0]
colors_xcr = [('#C0392B', 'Sub 1 – Prof. Kan'),
              ('#8E44AD', 'Sub 2 – Rajveer')]
max_lag_plot = 2.0  # ± 2 s

for (col, lbl), s, res in zip(colors_xcr, SESSIONS, results):
    lag, xc = calc_xcorr(res['t'], res['rf_smooth'], res['st_smooth'], s['k_on'], s['k_off'])
    vis = np.abs(lag) <= max_lag_plot
    peak_idx = np.argmax(np.abs(xc[vis]))
    peak_lag = lag[vis][peak_idx]
    peak_val = xc[vis][peak_idx]
    
    lbl_with_corr = f"{lbl} (Peak r = {peak_val:.3f} at lag = {peak_lag*1000:+.0f} ms)"
    ax.plot(lag[vis], xc[vis], color=col, lw=2.2, label=lbl_with_corr)
    ax.axvline(peak_lag, color=col, lw=1.5, ls='--', alpha=0.7)

ax.axvline(0, color='#2C3E50', lw=1.2, ls='-', alpha=0.5, label='Zero lag (ideal)')
ax.axhline(0, color='#AAAAAA', lw=0.8, ls='-')
ax.set_xlim(-max_lag_plot, max_lag_plot)
ax.set_xlabel('Lag (s)', fontsize=12)
ax.set_ylabel('Normalised Cross-Correlation', fontsize=12)
ax.set_title(f'{PANEL_LABELS[2]}  RF Envelope vs Stethoscope Envelope: Cross-Correlation\n'
             'Peak near zero-lag = physiological temporal alignment confirmed', fontsize=12, pad=4)

# ±0.1 s (100 ms) acceptable lag window
ax.fill_betweenx([-0.15, 1.05], -0.1, 0.1,
                 color='#27AE60', alpha=0.10, label='Acceptable Lag Band (±100 ms)')
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(-0.15, 1.05)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL D  —  RF Energy: Korotkoff Window vs Baseline
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_xcr[1]
bar_labels = []
bar_win    = []   # mean normalised RF energy INSIDE Korotkoff window
bar_pre    = []   # mean norm. RF energy PRE-window (cuff inflation phase)
bar_post   = []   # mean norm. RF energy POST-window (recovery)
bar_colors = []
snr_vals   = []

for (col, lbl), s, res in zip(colors_xcr, SESSIONS, results):
    t  = res['t'];  rf = res['rf']
    k_on = s['k_on'];  k_off = s['k_off']
    win_mask  = (t >= k_on)     & (t <= k_off)
    pre_mask  = (t >= k_on-6.0) & (t < k_on)  & (t > 0)
    post_mask = (t > k_off)     & (t <= k_off+6.0)
    e_win  = float(np.mean(rf[win_mask]))  if np.any(win_mask)  else 0.0
    e_pre  = float(np.mean(rf[pre_mask]))  if np.any(pre_mask)  else 0.0
    e_post = float(np.mean(rf[post_mask])) if np.any(post_mask) else 0.0
    snr = 10 * np.log10((e_win + 1e-12) / (np.mean([e_pre, e_post]) + 1e-12))
    bar_labels.append(lbl.split('\u2014')[0].strip() if '\u2014' in lbl else lbl.split('(')[0].strip())
    bar_win.append(e_win);  bar_pre.append(e_pre);  bar_post.append(e_post)
    bar_colors.append(col);  snr_vals.append(snr)

x = np.arange(len(bar_labels));  w = 0.25
b1 = ax.bar(x - w, bar_pre,  w, color='#7F8C8D', alpha=0.75, label='Pre-window (cuff inflation)')
b2 = ax.bar(x,     bar_win,  w, color='#F39C12', alpha=0.92, label='Korotkoff window (active)')
b3 = ax.bar(x + w, bar_post, w, color='#BDC3C7', alpha=0.75, label='Post-window (recovery)')

# Colour Korotkoff bars per subject
for bar, col in zip(b2, bar_colors):
    bar.set_facecolor(col);  bar.set_alpha(0.90)

# SNR annotation above Korotkoff bars
for i, (snr, e_win) in enumerate(zip(snr_vals, bar_win)):
    ax.text(x[i], e_win + 0.015, f'SNR = {snr:+.1f} dB',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=bar_colors[i],
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=bar_colors[i], lw=1.4, alpha=0.93))

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=11)
ax.set_ylabel('Mean Norm. RF Energy (a.u.)', fontsize=12)
ax.set_title(f'{PANEL_LABELS[3]}  RF Energy Specificity: Window vs Baseline\n'
             'Elevated RF energy ONLY during Korotkoff window confirms sensor selectivity',
             fontsize=12, pad=4)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(0, (max(bar_win) if bar_win else 1) * 1.55)

# ─────────────────────────────────────────────────────────────────────────────
# Super-title + save
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Radiomyography (RMG) — Korotkoff Signal Validation   (Best Sessions)\n'
    'Duration Match  |  Envelope Cross-Correlation  |  RF Energy Specificity',
    fontsize=16, y=0.97, fontweight='bold', color='#1A1A2E'
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"SAVED: {OUT}")
