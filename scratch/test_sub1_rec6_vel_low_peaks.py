import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF = 10_000
K_ON = 27.530
K_OFF = 43.330
STETH_LAG = 4.7672 # Aligned lag

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

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, _ = fit_circle(i_raw, q_raw)
phi_raw = robust_phase(i_raw - xc, q_raw - yc)

b, a = signal.iirnotch(100.71, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_raw)

# Filter RF Phase Velocity in low frequency band (0.5 - 10 Hz)
sos_vk = butter(4, [0.5, 10.0], btype='band', fs=FS_RF, output='sos')
phi_vel_low = np.append(np.diff(sosfiltfilt(sos_vk, phi_clean))*FS_RF, 0.0)

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
steth_env = np.convolve(steth_tkeo, np.ones(int(0.08*fs_a))/int(0.08*fs_a), mode='same')

t_rf = np.arange(len(phi_vel_low))/FS_RF
t_st = np.arange(len(steth_env))/fs_a

# Align Stethoscope time
t_st_aligned = t_st + STETH_LAG
steth_env_resamp = np.interp(t_rf, t_st_aligned, steth_env)

# Normalize in window
mask_koro = (t_rf >= K_ON) & (t_rf <= K_OFF)
t_koro = t_rf[mask_koro]
rf_vel_koro = phi_vel_low[mask_koro]
# We want the peaks of positive velocity (maximum rate of rise)
rf_vel_koro_n = (rf_vel_koro - np.min(rf_vel_koro)) / (np.max(rf_vel_koro) - np.min(rf_vel_koro) + 1e-20)
steth_koro_n = (steth_env_resamp[mask_koro] - np.min(steth_env_resamp[mask_koro])) / (np.max(steth_env_resamp[mask_koro]) - np.min(steth_env_resamp[mask_koro]) + 1e-20)

# Peak detection
peaks_rf, _ = signal.find_peaks(rf_vel_koro_n, prominence=0.2, distance=int(0.5*FS_RF))
peaks_st, _ = signal.find_peaks(steth_koro_n, prominence=0.1, distance=int(0.5*FS_RF))

print(f"Detected peaks in RF Low-Freq Velocity: {len(peaks_rf)}")
print(f"Detected peaks in Stethoscope: {len(peaks_st)}")

print("\nPeak Matching Table:")
matched_diffs = []
for p_st in peaks_st:
    t_st_peak = t_koro[p_st]
    closest_idx = np.argmin(np.abs(t_koro[peaks_rf] - t_st_peak))
    t_rf_peak = t_koro[peaks_rf[closest_idx]]
    diff_ms = (t_rf_peak - t_st_peak) * 1000
    if np.abs(diff_ms) < 300: # Match within 300ms
        matched_diffs.append(diff_ms)
        print(f"  Steth Peak: {t_st_peak:.3f} s | RF Vel Peak: {t_rf_peak:.3f} s | Diff: {diff_ms:+.1f} ms")

if len(matched_diffs) > 0:
    print(f"\nMatched {len(matched_diffs)} peaks.")
    print(f"Mean delay (RF relative to Steth): {np.mean(matched_diffs):+.2f} ms")
    print(f"Std dev of delay: {np.std(matched_diffs):.2f} ms")
else:
    print("\nNo peaks matched within 300ms.")
