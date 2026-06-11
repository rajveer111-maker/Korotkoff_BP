import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
FS_RF = 10_000
DEC = 10
FS = 1000
SCALE = ((299_792_458.0 / 0.9e9) * 1000) / (4.0 * np.pi)

def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')
def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)
def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)
def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same'))

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
phi_raw = robust_phase(i_raw - xc, q_raw - yc)

# Extract Phase Vel (20-90Hz)
phi_vel_rf = np.append(np.diff(bpf(phi_raw, 20, 90, FS_RF))*FS_RF, 0.0)*SCALE

# Notch the 64 Hz noise!
b, a = signal.iirnotch(64.0, 5, FS_RF)
phi_vel_rf_notched = signal.filtfilt(b, a, phi_vel_rf)

phi_tkeo_env_rf = smooth_energy(calc_tkeo(phi_vel_rf_notched), 1.5, FS_RF)
phi_tkeo = decimate(phi_tkeo_env_rf, DEC, ftype='fir')

t = np.arange(len(phi_tkeo))/FS
mask_koro = (t > 27.38) & (t < 42.0)
mask_base = (t > 20.0) & (t < 25.38)

print(f"WITH 64 Hz NOTCH:")
print(f"Max in Baseline : {np.max(phi_tkeo[mask_base]):.5f}")
print(f"Mean in Baseline: {np.mean(phi_tkeo[mask_base]):.5f}")
print(f"Max in Korotkoff: {np.max(phi_tkeo[mask_koro]):.5f}")
print(f"Mean in Korotkoff:{np.mean(phi_tkeo[mask_koro]):.5f}")
