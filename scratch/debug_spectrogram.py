"""
Test script to generate a Short-Time Fourier Transform (STFT) spectrogram
for RF Phase Velocity and check if it runs fast and looks clean.
"""
import h5py, os, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram, detrend

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; FC = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000.0) / (4.0 * np.pi)

s = dict(sub_dir='Sub_1_Prof_kan', rec=6, k_on=27.53, k_off=43.33)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

# Load RF
rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
with h5py.File(rp, 'r') as f: raw = f['data'][:]
ic, qc = -raw[0,:], raw[1,:]
xc, yc = fit_circle(ic, qc); ic -= xc; qc -= yc

phi = robust_phase(ic, qc)
vel_hi = np.append(np.diff(bpf(phi, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE

# Calculate STFT spectrogram
f, t_spec, Sxx = spectrogram(vel_hi, fs=FS_RF, nperseg=512, noverlap=460)

# Crop to 30 - 200 Hz
f_mask = (f >= 30) & (f <= 200)
f_crop = f[f_mask]
Sxx_crop = Sxx[f_mask, :]

plt.figure(figsize=(10, 4))
plt.pcolormesh(t_spec, f_crop, 10 * np.log10(Sxx_crop + 1e-12), shading='gouraud', cmap='inferno')
plt.colorbar(label='Power (dB)')
plt.xlim(22, 48)
plt.title("STFT Spectrogram of RF Phase Velocity (30-200 Hz)")
plt.savefig('d:\\Bioview\\My_RF_work_v1\\scratch\\debug_spectrogram.png')
print("Spectrogram saved successfully!")
