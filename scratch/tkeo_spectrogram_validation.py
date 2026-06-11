import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, detrend, spectrogram
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'tkeo_spectrogram_validation.png')
FS_RF = 10000
DEC = 10
FS = FS_RF // DEC
FC = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def robust_phase(i_c, q_c):
    iq = i_c + 1j * q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3 * iqr, 0.01)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return detrend(phase, type='linear')

def tkeo(x):
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return y

# Load Subject 1 Rec 6
rf_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
with h5py.File(rf_path, 'r') as f:
    rf_data = f['data'][:]

i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
phi = robust_phase(i_c, q_c)

# High-Frequency Micro-Velocity (10-200 Hz)
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE
vk_dec = decimate(vk, DEC, ftype='fir')
t = np.arange(len(vk_dec)) / FS

from scipy.signal import hilbert

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

# 1. Algorithmic Envelope Thresholding
# Use a 1.5 second smoothing window on the 10-200 Hz micro-velocity
vk_env = env_smooth(vk_dec, 1.5, FS)

# Search window during deflation (20s to 45s)
search_mask = (t > 20.0) & (t < 48.0)
env_max = np.max(vk_env[search_mask])

# Threshold is 20% of the maximum Korotkoff envelope energy
THRESHOLD = 0.20 * env_max

active_indices = np.where((vk_env > THRESHOLD) & search_mask)[0]

if len(active_indices) > 0:
    auto_k_on = t[active_indices[0]]
    auto_k_off = t[active_indices[-1]]
else:
    auto_k_on, auto_k_off = 0, 0

# 2. Plotting
plt.rcParams.update({'font.family': 'sans-serif'})
fig, axs = plt.subplots(2, 1, figsize=(14, 10), dpi=300, facecolor='white', gridspec_kw={'height_ratios': [1.5, 1]})

# Top Panel: Spectrogram
ax0 = axs[0]
f_spec, t_spec, Sxx = spectrogram(vk_dec, fs=FS, nperseg=int(FS*0.5), noverlap=int(FS*0.45), scaling='spectrum')
mask_f = (f_spec >= 10) & (f_spec <= 200)
Sxx = Sxx[mask_f, :]
f_spec = f_spec[mask_f]

# Log scale the spectrogram for better contrast
Sxx_log = 10 * np.log10(Sxx + 1e-10)

pcm = ax0.pcolormesh(t_spec, f_spec, Sxx_log, shading='gouraud', cmap='magma', vmin=np.percentile(Sxx_log, 40), vmax=np.percentile(Sxx_log, 99))
ax0.axvline(auto_k_on, color='#2ECC71', lw=3, ls='--', label=f'Auto $K_{{ON}}$ (SBP): {auto_k_on:.1f}s')
ax0.axvline(auto_k_off, color='#E74C3C', lw=3, ls='--', label=f'Auto $K_{{OFF}}$ (DBP): {auto_k_off:.1f}s')

ax0.set_title("Visual Confirmation: Spectrogram of RF Korotkoff Transients", fontweight='bold', fontsize=16)
ax0.set_ylabel("Frequency (Hz)", fontweight='bold')
ax0.set_xlim([15, 48])
ax0.set_ylim([10, 150])
ax0.legend(loc='upper right', framealpha=0.9)
fig.colorbar(pcm, ax=ax0, label='Power (dB)')

# Bottom Panel: Envelope and Threshold
ax1 = axs[1]
ax1.plot(t, vk_env, color='#2980B9', lw=2.5, label='10-200 Hz Korotkoff Envelope')
ax1.axhline(THRESHOLD, color='#F39C12', lw=2, ls='-', label='Threshold (20% of Max Envelope)')

ax1.axvspan(auto_k_on, auto_k_off, color='#27AE60', alpha=0.15, label='Automatically Detected Korotkoff Window')

ax1.set_title("Algorithmic Confirmation: Peak-Percentage Thresholding", fontweight='bold', fontsize=14)
ax1.set_xlabel("Time (Sec.)", fontweight='bold')
ax1.set_ylabel("Amplitude", fontweight='bold')
ax1.set_xlim([15, 48])

# Zoom y-axis to see threshold clearly
ymax = np.max(vk_env[search_mask]) * 1.2
ax1.set_ylim([0, ymax])

ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved to {OUT}")
