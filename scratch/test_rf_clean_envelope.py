"""
Test different DSP techniques to extract a clean, non-flat Korotkoff energy envelope
from the RF channel (using both magnitude and phase velocity).
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, detrend, iirnotch, filtfilt, fftconvolve
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_1_Prof_kan'
rf_path = os.path.join(BASE, sub_dir, 'Rec_1.h5')

FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

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

# Load
with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
xc, yc = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc

# 1. Phase Velocity
phi = robust_phase(i_c, q_c)
phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
vel_hi = np.append(np.diff(bpf(phi_clean, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
vel_tkeo = tkeo(vel_hi)
vel_dec = decimate(smooth(vel_tkeo, 0.15, FS_RF), DEC, ftype='fir')
vel_env = smooth(np.maximum(vel_dec, 0), 1.5, FS)

# 2. Magnitude
mag_raw = np.sqrt(i_c**2 + q_c**2)
mag_clean = nf(nf(nf(mag_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
mag_hi = bpf(mag_clean, 30, 200, FS_RF)
mag_tkeo = tkeo(mag_hi)
mag_dec = decimate(smooth(mag_tkeo, 0.15, FS_RF), DEC, ftype='fir')
mag_env = smooth(np.maximum(mag_dec, 0), 1.5, FS)

t = np.arange(len(vel_env)) / FS
mask = (t >= 22.0) & (t <= 45.0)

# Let's inspect baseline subtraction for Phase Velocity
# We can find the baseline noise floor by taking the 10th percentile inside the zoom window
vel_base = np.percentile(vel_env[mask], 10)
vel_env_clean = np.maximum(vel_env - vel_base, 0)
vel_env_clean_norm = vel_env_clean / (np.max(vel_env_clean[mask]) + 1e-12)

# Baseline subtraction for Magnitude
mag_base = np.percentile(mag_env[mask], 10)
mag_env_clean = np.maximum(mag_env - mag_base, 0)
mag_env_clean_norm = mag_env_clean / (np.max(mag_env_clean[mask]) + 1e-12)

print("Baseline subtraction test:")
print(f"  Phase Env range in mask (raw): min={np.min(vel_env[mask]):.2f}, max={np.max(vel_env[mask]):.2f}, ratio={np.min(vel_env[mask])/np.max(vel_env[mask]):.3f}")
print(f"  Phase Env range in mask (cleaned): min={np.min(vel_env_clean_norm[mask]):.3f}, max={np.max(vel_env_clean_norm[mask]):.3f}")
print(f"  Mag Env range in mask (raw): min={np.min(mag_env[mask]):.2f}, max={np.max(mag_env[mask]):.2f}, ratio={np.min(mag_env[mask])/np.max(mag_env[mask]):.3f}")
print(f"  Mag Env range in mask (cleaned): min={np.min(mag_env_clean_norm[mask]):.3f}, max={np.max(mag_env_clean_norm[mask]):.3f}")
