import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch

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

print("Loading RF data ...")
with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c  = i_raw - xc, q_raw - yc
phi_raw = robust_phase(i_c, q_c)

# Let's try different bandpass and notch filters on phase and velocity
# and compute PSD contrast (Korotkoff vs baseline)
mask_koro = (np.arange(len(phi_raw))/FS_RF >= K_ON) & (np.arange(len(phi_raw))/FS_RF <= K_OFF)
mask_base = (np.arange(len(phi_raw))/FS_RF >= 45.0) & (np.arange(len(phi_raw))/FS_RF <= 50.0)

# 1. Raw phase PSD (no derivative)
f_raw, pxx_k_raw = welch(phi_raw[mask_koro], fs=FS_RF, nperseg=int(FS_RF*1.0))
_, pxx_b_raw = welch(phi_raw[mask_base], fs=FS_RF, nperseg=int(FS_RF*1.0))

print("\n--- Raw Phase PSD comparison ---")
for freq in [1, 2, 5, 10, 20, 50, 100]:
    idx = np.argmin(np.abs(f_raw - freq))
    diff_db = 10 * np.log10(pxx_k_raw[idx] / pxx_b_raw[idx])
    print(f"  {freq} Hz: Korotkoff {10*np.log10(pxx_k_raw[idx]):.1f} dB, Baseline {10*np.log10(pxx_b_raw[idx]):.1f} dB, Contrast {diff_db:+.2f} dB")

# 2. Phase Velocity PSD (filtered 10-200 Hz, differentiated)
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
phi_filt = sosfiltfilt(sos_vk, phi_raw)
phi_vel = np.append(np.diff(phi_filt)*FS_RF, 0.0)*SCALE

f_vel, pxx_k_vel = welch(phi_vel[mask_koro], fs=FS_RF, nperseg=int(FS_RF*1.0))
_, pxx_b_vel = welch(phi_vel[mask_base], fs=FS_RF, nperseg=int(FS_RF*1.0))

print("\n--- Phase Velocity (10-200 Hz BPF) PSD comparison ---")
for freq in [10, 20, 30, 40, 50, 80, 100, 150]:
    idx = np.argmin(np.abs(f_vel - freq))
    diff_db = 10 * np.log10(pxx_k_vel[idx] / pxx_b_vel[idx])
    print(f"  {freq} Hz: Korotkoff {10*np.log10(pxx_k_vel[idx]):.1f} dB, Baseline {10*np.log10(pxx_b_vel[idx]):.1f} dB, Contrast {diff_db:+.2f} dB")

# Let's inspect the raw spectrum to find the exact frequency of major noise peaks
f, pxx = welch(phi_raw[mask_base], fs=FS_RF, nperseg=int(FS_RF*2.0))
pxx_db = 10 * np.log10(pxx + 1e-20)
# Find peaks in the spectrum above 10 Hz
peaks, _ = signal.find_peaks(pxx_db, prominence=5, distance=50)
peaks = peaks[f[peaks] >= 10]
sorted_peaks = peaks[np.argsort(pxx_db[peaks])[::-1]]

print("\n--- Top Spectral Peaks in Baseline Noise (> 10 Hz) ---")
for idx in sorted_peaks[:10]:
    print(f"  Freq: {f[idx]:.2f} Hz, Power: {pxx_db[idx]:.1f} dB")

# Apply notches at the top peaks
phi_notched = phi_raw.copy()
for idx in sorted_peaks[:5]:
    freq_notch = f[idx]
    # Notch out the peak
    b, a = signal.iirnotch(freq_notch, 30, FS_RF)
    phi_notched = signal.filtfilt(b, a, phi_notched)

# Now check integrated band power contrast of the notched signal
print("\n--- Integrated Band Power Contrast (Notched) ---")
bands = [(0.5, 3), (3, 10), (10, 30), (30, 80), (80, 200)]
for lo, hi in bands:
    sos = butter(4, [lo, hi], btype='band', fs=FS_RF, output='sos')
    sig_band = sosfiltfilt(sos, phi_notched)*SCALE
    pow_k = np.var(sig_band[mask_koro])
    pow_b = np.var(sig_band[mask_base])
    ratio_db = 10 * np.log10(pow_k / pow_b)
    print(f"  {lo}-{hi} Hz: Koro Var={pow_k:.2e} um^2, Base Var={pow_b:.2e} um^2, Contrast={ratio_db:+.2f} dB")

