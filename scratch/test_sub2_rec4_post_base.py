import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')

FS_RF = 10_000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE = LAMBDA_MM / (4.0 * np.pi)

K_ON = 27.380
K_OFF = 42.000
T_MAX = 51.0

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
b, a = signal.iirnotch(64.0, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_raw)
b, a = signal.iirnotch(100.6, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)
b, a = signal.iirnotch(50.0, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)

# Filter RF Phase Velocity (10-200 Hz)
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
phi_vel = np.append(np.diff(sosfiltfilt(sos_vk, phi_clean))*FS_RF, 0.0)*SCALE

# Compare pre- vs post- Korotkoff baseline contrast for Sub 2 Rec 4
mask_koro = (np.arange(len(phi_vel))/FS_RF >= K_ON) & (np.arange(len(phi_vel))/FS_RF <= K_OFF)
mask_base_pre = (np.arange(len(phi_vel))/FS_RF >= 20.0) & (np.arange(len(phi_vel))/FS_RF <= K_ON - 2.0)
mask_base_post = (np.arange(len(phi_vel))/FS_RF >= K_OFF + 2.0) & (np.arange(len(phi_vel))/FS_RF <= T_MAX)

# Check variance in different bands
bands = [(0.5, 3), (3, 10), (10, 30), (30, 80), (80, 200)]
print("\n--- Integrated Band Power Contrast (Pre-Korotkoff Baseline) ---")
for lo, hi in bands:
    sos = butter(4, [lo, hi], btype='band', fs=FS_RF, output='sos')
    sig_band = sosfiltfilt(sos, phi_clean)*SCALE
    pow_k = np.var(sig_band[mask_koro])
    pow_b = np.var(sig_band[mask_base_pre])
    ratio_db = 10 * np.log10(pow_k / pow_b)
    print(f"  {lo}-{hi} Hz: Koro Var={pow_k:.2e} um^2, Base Var={pow_b:.2e} um^2, Contrast={ratio_db:+.2f} dB")

print("\n--- Integrated Band Power Contrast (Post-Korotkoff Quiet Baseline) ---")
for lo, hi in bands:
    sos = butter(4, [lo, hi], btype='band', fs=FS_RF, output='sos')
    sig_band = sosfiltfilt(sos, phi_clean)*SCALE
    pow_k = np.var(sig_band[mask_koro])
    pow_b = np.var(sig_band[mask_base_post])
    ratio_db = 10 * np.log10(pow_k / pow_b)
    print(f"  {lo}-{hi} Hz: Koro Var={pow_k:.2e} um^2, Base Var={pow_b:.2e} um^2, Contrast={ratio_db:+.2f} dB")
