import h5py, os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, spectrogram
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
OUT = os.path.join(BASE, 'debug_sub2.png')

FS_RF = 10_000
DEC = 10
FS = FS_RF // DEC
FC = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000) / (4.0 * np.pi)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])

def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
phi_raw = robust_phase(i_raw - xc, q_raw - yc)

v_raw_p = np.append(np.diff(phi_raw) * FS_RF, 0.0) * SCALE
v_dec_p = decimate(v_raw_p, DEC, ftype='fir')
sos = butter(4, [20, 200], btype='band', fs=FS, output='sos')
phi_vel = sosfiltfilt(sos, v_dec_p)

# notch
b, a = signal.iirnotch(50, 30, FS)
phi_vel = signal.filtfilt(b, a, phi_vel)
b, a = signal.iirnotch(100, 30, FS)
phi_vel = signal.filtfilt(b, a, phi_vel)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)
def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same'))

phi_tkeo = smooth_energy(calc_tkeo(phi_vel), 1.5, FS)

t = np.arange(len(phi_vel))/FS
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(t, phi_vel)
axs[0].set_title("Raw 20-200Hz Velocity")
axs[0].set_ylim(-200, 200)

axs[1].plot(t, phi_tkeo)
axs[1].set_title("TKEO")
axs[1].set_ylim(0, np.max(phi_tkeo[(t>27) & (t<42)]) * 1.5)
axs[1].axvspan(27.38, 42.0, color='r', alpha=0.2)

f_p, t_p, Sxx_p = spectrogram(phi_vel, fs=FS, nperseg=int(FS*0.5), noverlap=int(FS*0.45), scaling='spectrum')
Sxx_p_log = 10 * np.log10(Sxx_p + 1e-20)
axs[2].pcolormesh(t_p, f_p, Sxx_p_log, shading='gouraud')
axs[2].set_ylim(20, 200)

plt.tight_layout()
plt.savefig(OUT)
