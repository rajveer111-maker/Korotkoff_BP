"""
Upgraded Academic Cohort Validation Dashboard
=============================================
Layout: 4 rows x 2 columns
- Left Column: Subject 1 (Prof. Kan) -> Session 1, Session 3, Session 6, and Grand Average (7 sessions)
- Right Column: Subject 2 (Rajveer)  -> Session 2, Session 4, Session 6, and Grand Average (6 sessions)
- Zoom Range: 22s to 45s (zoomed in to show detailed heartbeat alignment)
- Normalization: Conducted inside the [22, 45]s window
"""
import h5py
import numpy as np
from scipy import signal
from scipy.signal import (butter, sosfiltfilt, hilbert, decimate, detrend,
                           iirnotch, filtfilt, fftconvolve)
from scipy.io import wavfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = os.path.join(BASE, 'cohort_dual_modality_validation.png')
FS_RF = 10000; DEC = 10; FS = 1000
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

# ── DSP Helpers ──────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    k = max(1, int(w * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def process_session(sub_dir, rec_idx):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
    if not os.path.exists(rf_path) or not os.path.exists(wav_path):
        return None, None, None
        
    # RF
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    phi = robust_phase(i_c, q_c)
    phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    vel_hi = np.append(np.diff(bpf(phi_clean, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    
    vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
    rf_env = smooth(np.maximum(vel_dec, 0), 1.5, FS)
    t_rf = np.arange(len(rf_env)) / FS

    # Audio
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    
    st_bp = bpf(audio, 30, 1000, fs_a)
    st_hilb = np.abs(hilbert(st_bp))
    st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
    st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)
    
    steth_env = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)
    # Normalize in the zoomed window with baseline subtraction
    mask = (t_rf >= 22.0) & (t_rf <= 45.0)
    if np.any(mask):
        r_base = np.percentile(rf_env[mask], 5)
        rf_clean = np.maximum(rf_env - r_base, 0)
        r_norm = rf_clean / (np.max(rf_clean[mask]) + 1e-12)
        
        s_base = np.percentile(steth_env[mask], 5)
        steth_clean = np.maximum(steth_env - s_base, 0)
        s_norm = steth_clean / (np.max(steth_clean[mask]) + 1e-12)
    else:
        r_norm = rf_env
        s_norm = steth_env
        
    return t_rf, r_norm, s_norm


# ── Plot Config ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11.5,
    'axes.labelsize': 12.0, 'axes.labelweight': 'bold',
    'axes.titlesize': 13.0, 'axes.titleweight': 'bold',
    'legend.fontsize': 10, 'lines.linewidth': 2.2,
})

fig, axs = plt.subplots(4, 2, figsize=(16, 14), dpi=300, facecolor='white')

# Subjects definition
SUB1 = 'Sub_1_Prof_kan'
SUB2 = 'Sub_2_Rajveer'

# Plot sessions
sessions_sub1 = [1, 3, 6]
sessions_sub2 = [2, 4, 6]

t_std = np.linspace(22.0, 45.0, 2000)

# Process Subject 1 (Left Column)
all_s1 = []
all_r1 = []
valid_all_sub1 = [1, 2, 3, 6, 7, 8, 9]

print("Processing Subject 1 (Prof. Kan)...")
for idx, s_idx in enumerate(sessions_sub1):
    t_rf, r_norm, s_norm = process_session(SUB1, s_idx)
    ax = axs[idx, 0]
    if t_rf is not None:
        ax.plot(t_rf, s_norm, color='#2980B9', alpha=0.85, label='Stethoscope Envelope')
        ax.plot(t_rf, r_norm, color='#C0392B', alpha=0.85, label='RF Micro-Velocity')
        ax.set_title(f"Subject 1 (Prof. Kan) - Session {s_idx}", fontweight='bold')
    ax.set_xlim([22, 45])
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')
    if idx < 3:
        ax.set_xticklabels([])

# Gather all valid Subject 1 sessions for Grand Average
for s_idx in valid_all_sub1:
    t_rf, r_norm, s_norm = process_session(SUB1, s_idx)
    if t_rf is not None:
        all_s1.append(np.interp(t_std, t_rf, s_norm))
        all_r1.append(np.interp(t_std, t_rf, r_norm))

# Process Subject 2 (Right Column)
all_s2 = []
all_r2 = []
valid_all_sub2 = [1, 2, 3, 4, 5, 6]

print("Processing Subject 2 (Rajveer)...")
for idx, s_idx in enumerate(sessions_sub2):
    t_rf, r_norm, s_norm = process_session(SUB2, s_idx)
    ax = axs[idx, 1]
    if t_rf is not None:
        ax.plot(t_rf, s_norm, color='#2980B9', alpha=0.85, label='Stethoscope Envelope')
        ax.plot(t_rf, r_norm, color='#C0392B', alpha=0.85, label='RF Micro-Velocity')
        ax.set_title(f"Subject 2 (Rajveer) - Session {s_idx}", fontweight='bold')
    ax.set_xlim([22, 45])
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')
    if idx < 3:
        ax.set_xticklabels([])

# Gather all valid Subject 2 sessions for Grand Average
for s_idx in valid_all_sub2:
    t_rf, r_norm, s_norm = process_session(SUB2, s_idx)
    if t_rf is not None:
        all_s2.append(np.interp(t_std, t_rf, s_norm))
        all_r2.append(np.interp(t_std, t_rf, r_norm))

# ── Plot Grand Average for Subject 1 ──
ax_avg1 = axs[3, 0]
mean_s1 = np.mean(all_s1, axis=0)
std_s1  = np.std(all_s1, axis=0)
mean_r1 = np.mean(all_r1, axis=0)
std_r1  = np.std(all_r1, axis=0)

ax_avg1.plot(t_std, mean_s1, color='#2980B9', lw=2.6, label='Steth Mean')
ax_avg1.fill_between(t_std, np.maximum(mean_s1 - std_s1, 0), np.minimum(mean_s1 + std_s1, 1.1), color='#2980B9', alpha=0.15)
ax_avg1.plot(t_std, mean_r1, color='#C0392B', lw=2.6, label='RF Mean')
ax_avg1.fill_between(t_std, np.maximum(mean_r1 - std_r1, 0), np.minimum(mean_r1 + std_r1, 1.1), color='#C0392B', alpha=0.15)

ax_avg1.set_title("Subject 1 - Grand Average (7 Sessions)", fontweight='bold')
ax_avg1.set_xlim([22, 45])
ax_avg1.set_ylim([0, 1.15])
ax_avg1.set_xlabel("Time (s)", fontweight='bold')
ax_avg1.set_ylabel("Normalized Env.", fontweight='bold')
ax_avg1.grid(True, alpha=0.3)
ax_avg1.legend(loc='upper right')

# ── Plot Grand Average for Subject 2 ──
ax_avg2 = axs[3, 1]
mean_s2 = np.mean(all_s2, axis=0)
std_s2  = np.std(all_s2, axis=0)
mean_r2 = np.mean(all_r2, axis=0)
std_r2  = np.std(all_r2, axis=0)

ax_avg2.plot(t_std, mean_s2, color='#2980B9', lw=2.6, label='Steth Mean')
ax_avg2.fill_between(t_std, np.maximum(mean_s2 - std_s2, 0), np.minimum(mean_s2 + std_s2, 1.1), color='#2980B9', alpha=0.15)
ax_avg2.plot(t_std, mean_r2, color='#C0392B', lw=2.6, label='RF Mean')
ax_avg2.fill_between(t_std, np.maximum(mean_r2 - std_r2, 0), np.minimum(mean_r2 + std_r2, 1.1), color='#C0392B', alpha=0.15)

ax_avg2.set_title("Subject 2 - Grand Average (6 Sessions)", fontweight='bold')
ax_avg2.set_xlim([22, 45])
ax_avg2.set_ylim([0, 1.15])
ax_avg2.set_xlabel("Time (s)", fontweight='bold')
ax_avg2.grid(True, alpha=0.3)
ax_avg2.legend(loc='upper right')

# Set shared Y labels
for r in range(4):
    axs[r, 0].set_ylabel("Normalized Env.", fontweight='bold')

fig.suptitle("Clinical Cohort Validation: RMG vs Digital Acoustic Stethoscope", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"DONE -> {OUT}")
