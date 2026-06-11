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

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)
def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
phi_raw = robust_phase(i_raw - xc, q_raw - yc)

v_raw = np.append(np.diff(phi_raw) * FS_RF, 0) * SCALE
v_dec = decimate(v_raw, DEC, ftype='fir')
sos_vk = butter(4, [20, 200], btype='band', fs=FS, output='sos')
vk_dec = sosfiltfilt(sos_vk, v_dec)

# apply notch 50, 100, 150
b, a = signal.iirnotch(50, 30, FS)
vk_dec = signal.filtfilt(b, a, vk_dec)
b, a = signal.iirnotch(100.59, 10, FS)
vk_dec = signal.filtfilt(b, a, vk_dec)

tkeo_env = smooth_energy(calc_tkeo(vk_dec), 1.5, FS)

t = np.arange(len(tkeo_env))/FS
mask_koro = (t > 27.38) & (t < 42.0)
mask_base = (t > 20.0) & (t < 25.38)

b_min = np.percentile(tkeo_env[mask_base], 5)
e_shifted = np.maximum(tkeo_env - b_min, 0)
tkeo_n = e_shifted / (np.max(e_shifted[mask_koro]) + 1e-10)

peak_p = np.max(tkeo_n[mask_koro])
noise_p = np.mean(tkeo_n[mask_base])
snr_p = 10 * np.log10(peak_p / (noise_p + 1e-10))
print(f"SNR matching adaptive_spectrogram_validation.py exactly: {snr_p:.2f} dB")
