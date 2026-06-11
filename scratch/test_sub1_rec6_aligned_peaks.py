import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF = 10_000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE = LAMBDA_MM / (4.0 * np.pi)

# Clinical window in RF timeline
K_ON = 27.530
K_OFF = 43.330

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

print("Loading data...")
with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c  = i_raw - xc, q_raw - yc
phi_raw = robust_phase(i_c, q_c)

# Clean notches
b, a = signal.iirnotch(100.71, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_raw)
b, a = signal.iirnotch(201.43, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)

# Filter RF Phase Velocity (10-50 Hz)
sos_vk = butter(4, [10, 50], btype='band', fs=FS_RF, output='sos')
phi_vel = np.append(np.diff(sosfiltfilt(sos_vk, phi_clean))*FS_RF, 0.0)*SCALE
# Enveloping
rf_tkeo = phi_vel[1:-1]**2 - phi_vel[:-2]*phi_vel[2:]
rf_tkeo = np.maximum(rf_tkeo, 0)
rf_tkeo = np.insert(rf_tkeo, 0, 0.0)
rf_tkeo = np.append(rf_tkeo, 0.0)
win_rf = int(0.08 * FS_RF)
rf_env = np.convolve(rf_tkeo, np.ones(win_rf)/win_rf, mode='same')

# Load stethoscope
from scipy.io import wavfile
fs_a, audio = wavfile.read(WAV_PATH)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

# Filter stethoscope in the clinical band (30-150 Hz)
sos_st = butter(4, [30, 150], btype='band', fs=fs_a, output='sos')
audio_filt = sosfiltfilt(sos_st, audio)
steth_tkeo = audio_filt[1:-1]**2 - audio_filt[:-2]*audio_filt[2:]
steth_tkeo = np.maximum(steth_tkeo, 0)
steth_tkeo = np.insert(steth_tkeo, 0, 0.0)
steth_tkeo = np.append(steth_tkeo, 0.0)
win_st = int(0.08 * fs_a)
steth_env = np.convolve(steth_tkeo, np.ones(win_st)/win_st, mode='same')

# Time vectors
t_rf = np.arange(len(rf_env))/FS_RF
t_st = np.arange(len(steth_env))/fs_a

# Lag is +4.767s (meaning Steth start is delayed by 4.767s relative to RF)
steth_lag = 4.767
t_st_aligned = t_st + steth_lag

# Interpolate aligned stethoscope onto the RF time grid
steth_env_resamp = np.interp(t_rf, t_st_aligned, steth_env)

# Focus on Korotkoff window in RF timeline
mask_koro = (t_rf >= K_ON) & (t_rf <= K_OFF)
rf_koro = rf_env[mask_koro]
steth_koro = steth_env_resamp[mask_koro]
t_koro = t_rf[mask_koro]

# Normalise
rf_koro_n = (rf_koro - np.min(rf_koro)) / (np.max(rf_koro) - np.min(rf_koro) + 1e-20)
steth_koro_n = (steth_koro - np.min(steth_koro)) / (np.max(steth_koro) - np.min(steth_koro) + 1e-20)

# Peak detection
peaks_rf, _ = signal.find_peaks(rf_koro_n, prominence=0.15, distance=int(0.6*FS_RF))
peaks_st, _ = signal.find_peaks(steth_koro_n, prominence=0.15, distance=int(0.6*FS_RF))

print(f"Detected peaks in RF: {len(peaks_rf)}")
print(f"Detected peaks in Stethoscope: {len(peaks_st)}")

# Match peaks
matched_diffs = []
for p_st in peaks_st:
    t_st_peak = t_koro[p_st]
    closest_idx = np.argmin(np.abs(t_koro[peaks_rf] - t_st_peak))
    t_rf_peak = t_koro[peaks_rf[closest_idx]]
    diff_ms = (t_rf_peak - t_st_peak) * 1000
    if np.abs(diff_ms) < 200: # only match if within 200ms
        matched_diffs.append(diff_ms)
        print(f"  Steth Peak: {t_st_peak:.3f} s | RF Peak: {t_rf_peak:.3f} s | Diff: {diff_ms:+.1f} ms")

if len(matched_diffs) > 0:
    print(f"\nMatched {len(matched_diffs)} peaks.")
    print(f"Mean delay (RF relative to Steth): {np.mean(matched_diffs):+.2f} ms")
    print(f"Std dev of delay: {np.std(matched_diffs):.2f} ms")
else:
    print("\nNo peaks matched within 200ms.")

# Let's save a plot showing the aligned envelopes
plt.figure(figsize=(12, 6))
plt.plot(t_koro, rf_koro_n, color='#C0392B', lw=1.5, label='RF Phase Velocity TKEO (Aligned)')
plt.plot(t_koro, steth_koro_n, color='#2980B9', lw=1.5, ls='--', label='Stethoscope TKEO (Aligned)')
plt.scatter(t_koro[peaks_rf], rf_koro_n[peaks_rf], color='#C0392B', s=40, zorder=5)
plt.scatter(t_koro[peaks_st], steth_koro_n[peaks_st], color='#2980B9', s=40, marker='x', zorder=5)
plt.title(f"Aligned Korotkoff TKEO Envelopes (Subject 1, Rec 6) | Lag = {steth_lag:.3f} s")
plt.xlabel("Time (s)")
plt.ylabel("Normalized TKEO Energy")
plt.legend()
plt.grid(True)
plt.savefig("scratch/sub1_rec6_aligned_envelopes.png", dpi=300)
print("Saved envelope comparison plot to scratch/sub1_rec6_aligned_envelopes.png")
