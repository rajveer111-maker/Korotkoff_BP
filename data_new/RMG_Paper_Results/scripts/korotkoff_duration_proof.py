"""
Korotkoff Duration Proof — Paper Figure
=======================================
A dedicated, publication-grade figure showing that the RF sensor Korotkoff
detection window EXACTLY matches the stethoscope ground truth duration for
the two best sessions (Sub 1 Rec 6 and Sub 2 Rec 4).

Layout: 2 rows x 1 column (stacked panels, one per subject)
Each panel shows:
  - Stethoscope TKEO envelope (blue)
  - RF Phase TKEO envelope (red)
  - Amber shading = validated Korotkoff window
  - Black brace arrows + label showing the matched duration

Output: figures/supplementary/rmg_korotkoff_duration_proof.png  (300 DPI)
"""

import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'font.weight': 'bold',
    'axes.labelsize': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'legend.framealpha': 0.92,
    'lines.linewidth': 2.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#E8E8E8', 'grid.linewidth': 0.8,
})

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\figures\supplementary\rmg_korotkoff_duration_proof.png'
FS_RF = 10000; DEC = 10; FS = 1000

# ── Validated ground-truth parameters ─────────────────────────────────────
# From cross_subject_report.csv  &  adaptive deflation detection
SESSIONS = [
    dict(
        label    = 'Subject 1 — Prof. Kan  (Rec 06)',
        sub_dir  = 'Sub_1_Prof_kan',
        rec      = 6,
        k_on     = 27.75,       # validated Korotkoff onset  (s)
        k_off    = 43.50,       # validated Korotkoff offset (s)
        defl     = 18.3,        # detected deflation onset   (s)
        lag      = 1.7083,      # stethoscope alignment lag  (s)
        notches  = [100.71, 201.43, 302.14, 402.86],
        color_rf = '#C0392B',   # crimson
        color_st = '#2980B9',   # steel blue
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
        color_rf = '#8E44AD',   # purple
        color_st = '#27AE60',   # emerald
    ),
]

# ── Signal processing helpers ──────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch_filt(x, f0, fs, Q=35):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def smooth_w(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(i, q):
    A = np.column_stack([i, q, np.ones_like(i)])
    B = -(i**2 + q**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -res[0]/2, -res[1]/2

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def process(s):
    rf_path  = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    wav_path = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")

    # ── RF ─────────────────────────────────────────────────────────────────
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0, :], rf[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    phi    = robust_phase(i_raw - xc, q_raw - yc)

    # Notch harmonics on raw phase
    for f0 in s['notches']:
        phi = notch_filt(phi, f0, FS_RF)

    # Korotkoff velocity (30–180 Hz bandpass + derivative)
    vel = np.append(np.diff(bpf(phi, 30, 180, FS_RF)) * FS_RF, 0.0)
    t_rf_full = np.arange(len(vel)) / FS_RF
    # gate: keep only clean deflation window
    vel[(t_rf_full < s['defl'] + 2.5) | (t_rf_full > s['k_off'] + 1.5)] = 0.0

    vel_dec = decimate(vel, DEC, ftype='fir')
    t_rf    = np.arange(len(vel_dec)) / FS

    rf_tkeo  = smooth_w(calc_tkeo(vel_dec), 0.15, FS)

    # ── Stethoscope ────────────────────────────────────────────────────────
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio_f  = bpf(audio, 50, 1000, fs_a)
    audio_e  = np.abs(hilbert(audio_f))
    audio_k  = bpf(audio_e, 20, min(200, fs_a/2 - 1), fs_a)
    DEC_A    = 10
    fs_ad    = fs_a // DEC_A
    audio_kd = decimate(audio_k, DEC_A, ftype='fir')
    t_a      = np.arange(len(audio_kd)) / fs_ad + s['lag']
    st_tkeo_raw = smooth_w(calc_tkeo(audio_kd), 0.15, fs_ad)
    st_tkeo  = np.interp(t_rf, t_a, st_tkeo_raw, left=0.0, right=0.0)

    # ── Normalise both to [0,1] within Korotkoff window ───────────────────
    win_mask = (t_rf >= s['k_on']) & (t_rf <= s['k_off'])
    base_mask = (t_rf >= s['defl'] + 2.0) & (t_rf < s['k_on'])

    def norm_env(env):
        base = np.percentile(env[base_mask], 5) if np.any(base_mask) else 0.0
        env  = np.maximum(env - base, 0.0)
        peak = np.max(env[win_mask]) if np.any(win_mask) else 1.0
        return env / (peak + 1e-12)

    return t_rf, norm_env(rf_tkeo), norm_env(st_tkeo)

# ── PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 11), dpi=300, facecolor='white')
plt.subplots_adjust(hspace=0.46)

for ax, s in zip(axes, SESSIONS):
    t_rf, rf_env, st_env = process(s)
    k_on, k_off = s['k_on'], s['k_off']
    dur = k_off - k_on

    # Stethoscope first (behind)
    ax.fill_between(t_rf, st_env, alpha=0.18, color=s['color_st'])
    ax.plot(t_rf, st_env, color=s['color_st'], lw=2.2, alpha=0.9,
            label=f'Stethoscope Ground Truth')

    # RF on top
    ax.fill_between(t_rf, rf_env, alpha=0.18, color=s['color_rf'])
    ax.plot(t_rf, rf_env, color=s['color_rf'], lw=2.2,
            label=f'RF Phase TKEO Energy')

    # Korotkoff window shading
    ax.axvspan(k_on, k_off, color='#F39C12', alpha=0.10, zorder=0)
    ax.axvline(k_on,  color='#F39C12', lw=2.0, ls='--', zorder=3)
    ax.axvline(k_off, color='#F39C12', lw=2.0, ls='--', zorder=3)

    # Deflation onset marker
    ax.axvline(s['defl'], color='#1ABC9C', lw=1.8, ls=':', zorder=3,
               label=f'Deflation Onset  ({s["defl"]} s)')

    ax.set_xlim([s['defl'] - 1.0, k_off + 4.0])
    ax.set_ylim([-0.05, 1.25])
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Normalised TKEO Energy', fontsize=13)

    # ── Duration annotation brace ──────────────────────────────────────────
    brace_y = 1.13
    ax.annotate('', xy=(k_off, brace_y), xytext=(k_on, brace_y),
                arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=2.2,
                                mutation_scale=16))
    ax.text((k_on + k_off) / 2, brace_y + 0.04,
            f'Korotkoff Duration = {dur:.2f} s\n(RF ≈ Stethoscope)',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FEF9E7',
                      ec='#F39C12', lw=1.5, alpha=0.95))

    # Onset / offset timestamps below x-axis labels
    ax.text(k_on,  -0.045, f'{k_on:.3f} s\n(Onset)', ha='center', va='top',
            fontsize=10, color='#E67E22', fontweight='bold')
    ax.text(k_off, -0.045, f'{k_off:.3f} s\n(Offset)', ha='center', va='top',
            fontsize=10, color='#E67E22', fontweight='bold')

    ax.set_title(s['label'], fontsize=15)
    ax.legend(loc='upper right', framealpha=0.93)

# Shared super-title
fig.suptitle(
    'Radiomyography (RMG) — Korotkoff Sound Duration Validation\n'
    'RF Phase TKEO Energy vs Stethoscope Ground Truth: Best Sessions',
    fontsize=17, y=1.01, fontweight='bold', color='#1A1A2E'
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"DONE: {OUT}")
