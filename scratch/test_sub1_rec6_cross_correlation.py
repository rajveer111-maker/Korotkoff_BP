import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF = 10_000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE = LAMBDA_MM / (4.0 * np.pi)

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

# Interpolate raw stethoscope (without lag) onto RF grid
steth_env_resamp = np.interp(t_rf, t_st, steth_env)

# Focus on a broad range around the Korotkoff window to find lag
# e.g., 20 to 48s in RF
mask_rf = (t_rf >= 20.0) & (t_rf <= 48.0)
rf_w = rf_env[mask_rf]
steth_w = steth_env_resamp[mask_rf]

# Normalise
rf_w_n = (rf_w - np.min(rf_w)) / (np.max(rf_w) - np.min(rf_w) + 1e-20)
steth_w_n = (steth_w - np.min(steth_w)) / (np.max(steth_w) - np.min(steth_w) + 1e-20)

# Cross-correlate with a maximum shift of 10 seconds (100,000 samples)
max_lag_samples = int(10 * FS_RF)
corr = signal.correlate(rf_w_n - np.mean(rf_w_n), steth_w_n - np.mean(steth_w_n), mode='full')
lags = signal.correlation_lags(len(rf_w_n), len(steth_w_n), mode='full')

# Find peak within max lag range
mask_lags = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
best_idx = np.argmax(corr[mask_lags])
best_lag = lags[mask_lags][best_idx]
best_time_lag = best_lag / FS_RF

print(f"Optimal lag (RF starts before Steth): {best_time_lag:+.3f} s ({best_time_lag*1000:+.1f} ms)")

# Print cross-correlation details
plt.figure(figsize=(10, 5))
plt.plot(lags[mask_lags]/FS_RF, corr[mask_lags], color='purple', lw=1.5)
plt.axvline(best_time_lag, color='red', ls='--', label=f'Best Lag: {best_time_lag:.3f} s')
plt.title("Cross-Correlation of RF and Stethoscope TKEO Envelopes")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation Coefficient")
plt.legend()
plt.grid(True)
plt.savefig("scratch/sub1_rec6_cross_correlation.png", dpi=300)
print("Saved correlation plot to scratch/sub1_rec6_cross_correlation.png")
