import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF = 10_000
DEC = 10
FS = FS_RF // DEC
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

print("Loading RF data ...")
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
b, a = signal.iirnotch(302.14, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)
b, a = signal.iirnotch(402.86, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)

# Bandpass filter 10-200 Hz
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
phi_filt = sosfiltfilt(sos_vk, phi_clean)
phi_vel = np.append(np.diff(phi_filt)*FS_RF, 0.0)*SCALE

# Let's compute TKEO with a short window (e.g. 50ms)
tkeo = phi_vel[1:-1]**2 - phi_vel[:-2]*phi_vel[2:]
tkeo = np.maximum(tkeo, 0)
tkeo = np.insert(tkeo, 0, 0.0)
tkeo = np.append(tkeo, 0.0)

# Short-term smooth (100ms)
win_len = int(0.1 * FS_RF)
tkeo_smooth = np.convolve(tkeo, np.ones(win_len)/win_len, mode='same')

# Load stethoscope
print("Loading Stethoscope audio ...")
from scipy.io import wavfile
fs_a, audio = wavfile.read(WAV_PATH)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

# Find peaks in stethoscope audio to identify heartbeat locations
audio_filt = butter(4, [50, 400], btype='band', fs=fs_a, output='sos')
audio_filt = sosfiltfilt(audio_filt, audio)
steth_env = np.abs(signal.hilbert(audio_filt))
steth_env_smooth = np.convolve(steth_env, np.ones(int(0.15*fs_a))/int(0.15*fs_a), mode='same')

# Let's find peaks in steth_env_smooth during the Korotkoff window
mask_koro_st = (np.arange(len(audio))/fs_a >= K_ON) & (np.arange(len(audio))/fs_a <= K_OFF)
t_a = np.arange(len(audio))/fs_a

peaks_idx, _ = signal.find_peaks(steth_env_smooth, prominence=0.01, distance=int(0.6*fs_a))
peaks_koro = peaks_idx[(t_a[peaks_idx] >= K_ON) & (t_a[peaks_idx] <= K_OFF)]

print(f"Detected {len(peaks_koro)} peaks in Stethoscope during Korotkoff window")

# Now let's calculate the gated PSD around these peaks in RF Phase Velocity!
# We'll take a 200 ms window around each peak for the "Systolic Active" PSD
# and a 200 ms window in the middle between peaks for the "Diastolic Quiet" PSD
half_w_rf = int(0.1 * FS_RF) # 100 ms half-width

active_segments = []
quiet_segments = []

for idx_st in peaks_koro:
    t_peak = t_a[idx_st]
    idx_rf = int(t_peak * FS_RF)
    if idx_rf - half_w_rf >= 0 and idx_rf + half_w_rf < len(phi_vel):
        active_segments.append(phi_vel[idx_rf - half_w_rf : idx_rf + half_w_rf])
        
        # Quiet segment: 500 ms after the peak (diastole)
        idx_rf_q = idx_rf + int(0.5 * FS_RF)
        if idx_rf_q - half_w_rf >= 0 and idx_rf_q + half_w_rf < len(phi_vel):
            quiet_segments.append(phi_vel[idx_rf_q - half_w_rf : idx_rf_q + half_w_rf])

if len(active_segments) > 0:
    active_sig = np.concatenate(active_segments)
    quiet_sig = np.concatenate(quiet_segments)
    
    f_gated, pxx_active = welch(active_sig, fs=FS_RF, nperseg=min(len(active_sig), int(FS_RF*0.2)))
    _, pxx_quiet = welch(quiet_sig, fs=FS_RF, nperseg=min(len(quiet_sig), int(FS_RF*0.2)))
    
    print("\n--- Gated PSD (Systole vs Diastole within Korotkoff window) ---")
    for freq in [20, 30, 40, 50, 60, 80, 120, 150]:
        idx = np.argmin(np.abs(f_gated - freq))
        diff_db = 10 * np.log10(pxx_active[idx] / pxx_quiet[idx])
        print(f"  {freq} Hz: Systole {10*np.log10(pxx_active[idx]):.1f} dB, Diastole {10*np.log10(pxx_quiet[idx]):.1f} dB, Contrast {diff_db:+.2f} dB")
        
    # Baseline comparison (outside Korotkoff window, e.g. 18-25 s)
    mask_base_rf = (np.arange(len(phi_vel))/FS_RF >= 18.0) & (np.arange(len(phi_vel))/FS_RF <= K_ON - 2.0)
    base_sig = phi_vel[mask_base_rf]
    _, pxx_base = welch(base_sig, fs=FS_RF, nperseg=min(len(base_sig), int(FS_RF*0.2)))
    
    print("\n--- Gated PSD (Systole vs Baseline Noise) ---")
    for freq in [20, 30, 40, 50, 60, 80, 120, 150]:
        idx_g = np.argmin(np.abs(f_gated - freq))
        idx_b = np.argmin(np.abs(f_gated - freq)) # same grid
        diff_db = 10 * np.log10(pxx_active[idx_g] / pxx_base[idx_b])
        print(f"  {freq} Hz: Systole {10*np.log10(pxx_active[idx_g]):.1f} dB, Baseline Noise {10*np.log10(pxx_base[idx_b]):.1f} dB, Contrast {diff_db:+.2f} dB")
else:
    print("No segments extracted!")
