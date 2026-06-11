import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, welch

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
FS_RF = 10_000
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

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
phi_raw = robust_phase(i_raw - xc, q_raw - yc)

phi_vel_rf = np.append(np.diff(bpf(phi_raw, 10, 200, FS_RF))*FS_RF, 0.0)*SCALE
t = np.arange(len(phi_vel_rf))/FS_RF
mask_koro = (t > 27.38) & (t < 42.0)
mask_base = (t > 20.0) & (t < 25.38)

f_k, pxx_k = welch(phi_vel_rf[mask_koro], fs=FS_RF, nperseg=1024)
f_b, pxx_b = welch(phi_vel_rf[mask_base], fs=FS_RF, nperseg=1024)

diff_pxx = 10*np.log10(pxx_k + 1e-20) - 10*np.log10(pxx_b + 1e-20)
mask = (f_k >= 10) & (f_k <= 200)

for f_val, snr_val in zip(f_k[mask], diff_pxx[mask]):
    if snr_val > 1:
        print(f"Freq: {f_val:.2f} Hz -> SNR: {snr_val:.2f} dB")
