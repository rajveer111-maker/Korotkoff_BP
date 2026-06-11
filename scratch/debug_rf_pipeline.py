"""
Debug script to inspect RF intermediate values for Session 1 of Subject 1
to see where the signal becomes flat or if there is an issue with the pipeline.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, detrend, iirnotch, filtfilt, fftconvolve

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

# Load and process
with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
xc, yc = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc

phi = robust_phase(i_c, q_c)
print(f"Phase range: min={np.min(phi):.3f}, max={np.max(phi):.3f}, mean={np.mean(phi):.3f}")

phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
print(f"Clean phase range: min={np.min(phi_clean):.3f}, max={np.max(phi_clean):.3f}")

vel_hi = np.append(np.diff(bpf(phi_clean, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
print(f"Velocity range: min={np.min(vel_hi):.3f}, max={np.max(vel_hi):.3f}, std={np.std(vel_hi):.3f}")

tkeo_vel = tkeo(vel_hi)
print(f"TKEO range: min={np.min(tkeo_vel):.3f}, max={np.max(tkeo_vel):.3f}, mean={np.mean(tkeo_vel):.3f}")

smoothed_tkeo = smooth(tkeo_vel, 0.15, FS_RF)
print(f"Smoothed TKEO range: min={np.min(smoothed_tkeo):.3f}, max={np.max(smoothed_tkeo):.3f}")

vel_dec = decimate(smoothed_tkeo, DEC, ftype='fir')
print(f"Decimated velocity range: min={np.min(vel_dec):.3f}, max={np.max(vel_dec):.3f}")

rf_env = smooth(np.maximum(vel_dec, 0), 1.5, FS)
print(f"Final RF Env range: min={np.min(rf_env):.3f}, max={np.max(rf_env):.3f}")

t_rf = np.arange(len(rf_env)) / FS
mask = (t_rf >= 22.0) & (t_rf <= 45.0)
print(f"Final RF Env in mask range: min={np.min(rf_env[mask]):.3f}, max={np.max(rf_env[mask]):.3f}")
