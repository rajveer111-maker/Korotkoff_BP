"""
Multi-Algorithm Validation of Korotkoff Duration
=================================================
Shows RF Phase Velocity vs Stethoscope Ground Truth for the two best sessions
using 4 energy-extraction algorithms (TKEO, RMS, Shannon, Hilbert).

FIX: Tight gating — RF zeroed outside [k_on-1, k_off+1]; normalization is
     strictly inside the Korotkoff window so pre-onset noise cannot inflate
     the envelope.
"""

import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert
from scipy.signal import filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'font.weight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'legend.fontsize': 11, 'lines.linewidth': 2.0,
    'axes.grid': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT1 = os.path.join(BASE, 'multi_algo_best_both_subjects.png')
OUT2 = r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\figures\diagnostic\multi_algo_best_both_subjects.png'
FS_RF = 10000; DEC = 10; FS = 1000
CP = '#C0392B'   # crimson  — RF
CS = '#2980B9'   # blue     — Stethoscope

# ── helpers ──────────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=35):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def calc_rms(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.maximum(fftconvolve(x**2, np.ones(k)/k, mode='same'), 0))

def calc_shannon(x):
    xn = x / (np.max(np.abs(x)) + 1e-10)
    return np.maximum(-(xn**2) * np.log(xn**2 + 1e-10), 0)

def calc_analytic(x):
    return np.abs(hilbert(x))

def smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -res[0]/2, -res[1]/2

# ── Best Session from Each Subject ────────────────────────────────────────────
# (sub_dir, sub_name, rec_idx, k_on, k_off, lag, defl, notches)
sessions = [
    ('Sub_1_Prof_kan', 'Subject 1 (Prof. Kan)', 6,
     27.75, 43.50, 1.7083, 18.3, [100.71, 201.43, 302.14, 402.86]),
    ('Sub_2_Rajveer',  'Subject 2 (Rajveer)',   4,
     27.375, 42.00, 2.6042, 18.6, [50.0, 64.0, 100.6, 201.2])
]

algo_titles = [
    'Teager-Kaiser Energy Operator (TKEO)',
    'Root Mean Square (RMS) Energy',
    'Shannon Energy',
    'Absolute Analytic Envelope (Hilbert)',
]

fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=300, facecolor='white')

for col_idx, (sub_dir, sub_name, rec_idx, k_on, k_off, lag, defl, notches) in enumerate(sessions):
    rf_path  = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    print(f"Processing {sub_name} Rec {rec_idx}...")

    # ── RF phase extraction ──────────────────────────────────────────────────
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0, :], rf[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    phi = robust_phase(i_raw - xc, q_raw - yc)

    # Notch harmonics on full-rate phase
    for freq in notches:
        phi = notch(phi, freq, FS_RF)

    # Korotkoff velocity (30–180 Hz bandpass + derivative)
    vel = np.append(np.diff(bpf(phi, 30, 180, FS_RF)) * FS_RF, 0.0)

    # ── TIGHT gating: zero everything OUTSIDE [k_on-1.5 … k_off+1.5] ───────
    # This removes all valve/pump transients that inflate the pre-Korotkoff baseline
    t_rf_full = np.arange(len(vel)) / FS_RF
    gate_lo = k_on  - 1.5
    gate_hi = k_off + 1.5
    vel[(t_rf_full < gate_lo) | (t_rf_full > gate_hi)] = 0.0

    vel_dec = decimate(vel, DEC, ftype='fir')
    t_rf    = np.arange(len(vel_dec)) / FS

    # ── Stethoscope ──────────────────────────────────────────────────────────
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio_f   = bpf(audio, 50, 1000, fs_a)
    audio_k   = bpf(np.abs(audio_f), 20, min(200, (fs_a/2) - 1), fs_a)
    DEC_A = 10; fs_ad = fs_a // DEC_A
    audio_k_d = decimate(audio_k, DEC_A, ftype='fir')

    # Gate stethoscope the same way (aligned timeline)
    t_a_full = (np.arange(len(audio_k_d)) / fs_ad) + lag
    audio_k_d[(t_a_full < gate_lo) | (t_a_full > gate_hi)] = 0.0

    # ── Compute 4 energy envelopes ───────────────────────────────────────────
    r_tkeo = smooth(calc_tkeo(vel_dec),        1.2, FS)
    s_tkeo = smooth(calc_tkeo(audio_k_d),      1.2, fs_ad)
    r_rms  = smooth(calc_rms(vel_dec,  0.3, FS),     0.8, FS)
    s_rms  = smooth(calc_rms(audio_k_d, 0.3, fs_ad), 0.8, fs_ad)
    r_shan = smooth(calc_shannon(vel_dec),     1.2, FS)
    s_shan = smooth(calc_shannon(audio_k_d),   1.2, fs_ad)
    r_hilb = smooth(calc_analytic(vel_dec),    1.2, FS)
    s_hilb = smooth(calc_analytic(audio_k_d),  1.2, fs_ad)

    # Resample steth envelopes to RF timeline
    t_a = (np.arange(len(s_tkeo)) / fs_ad) + lag
    s_tkeo = np.interp(t_rf, t_a, s_tkeo, left=0.0, right=0.0)

    t_a = (np.arange(len(s_rms)) / fs_ad) + lag
    s_rms  = np.interp(t_rf, t_a, s_rms,  left=0.0, right=0.0)

    t_a = (np.arange(len(s_shan)) / fs_ad) + lag
    s_shan = np.interp(t_rf, t_a, s_shan, left=0.0, right=0.0)

    t_a = (np.arange(len(s_hilb)) / fs_ad) + lag
    s_hilb = np.interp(t_rf, t_a, s_hilb, left=0.0, right=0.0)

    approaches = [
        (r_tkeo, s_tkeo),
        (r_rms,  s_rms),
        (r_shan, s_shan),
        (r_hilb, s_hilb),
    ]

    # ── Normalization: scale to peak INSIDE Korotkoff window ─────────────────
    win_mask = (t_rf >= k_on) & (t_rf <= k_off)

    def norm(env):
        peak = np.max(env[win_mask]) if np.any(win_mask) and np.max(env[win_mask]) > 0 else 1.0
        return env / peak

    # ── Plot ─────────────────────────────────────────────────────────────────
    for row_idx, (r_env, s_env) in enumerate(approaches):
        ax = axes[row_idx, col_idx]

        r_n = norm(r_env)
        s_n = norm(s_env)

        ax.fill_between(t_rf, s_n, alpha=0.15, color=CS)
        ax.plot(t_rf, s_n, color=CS, lw=2.2, alpha=0.9,
                label='Stethoscope Acoustic (GT)')
        ax.fill_between(t_rf, r_n, alpha=0.15, color=CP)
        ax.plot(t_rf, r_n, color=CP, lw=2.2, alpha=0.85,
                label='RF Phase Velocity')

        # Korotkoff window markers
        ax.axvspan(k_on, k_off, color='#F39C12', alpha=0.10, zorder=0)
        ax.axvline(k_on,  color='#F39C12', lw=2.0, ls='--', zorder=2)
        ax.axvline(k_off, color='#F39C12', lw=2.0, ls='--', zorder=2)

        # Duration annotation on first row only
        if row_idx == 0:
            dur = k_off - k_on
            ax.annotate('', xy=(k_off, 1.05), xytext=(k_on, 1.05),
                        arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=2.0,
                                        mutation_scale=14))
            ax.text((k_on + k_off)/2, 1.08,
                    f'Korotkoff: {dur:.2f} s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.25', fc='#FEF9E7',
                              ec='#F39C12', lw=1.2, alpha=0.95))

        ax.set_title(f"{sub_name}\n{algo_titles[row_idx]}", pad=6)
        ax.set_xlim([k_on - 3, k_off + 3])
        ax.set_ylim([-0.05, 1.20])

        if row_idx == 3:
            ax.set_xlabel('Time (s)', fontsize=13)
        if col_idx == 0:
            ax.set_ylabel('Normalised Energy', fontsize=13)
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper left', fontsize=10)

fig.suptitle(
    'Multi-Algorithm Validation of Korotkoff Duration: RF Phase vs Stethoscope\n'
    'Comparing 4 Signal Processing Techniques — Best Sessions of Both Subjects',
    fontsize=19, y=0.995, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.01, 1, 0.975])
plt.subplots_adjust(hspace=0.45, wspace=0.18)

plt.savefig(OUT1, dpi=300, bbox_inches='tight', facecolor='white')
os.makedirs(os.path.dirname(OUT2), exist_ok=True)
plt.savefig(OUT2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"DONE: {OUT1}")
print(f"DONE: {OUT2}")
