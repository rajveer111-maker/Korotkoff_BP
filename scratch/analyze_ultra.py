import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, spectrogram
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"
os.makedirs(out_dir, exist_ok=True)

# Define circle fitting to get clean phase
def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    # Simple unwrapping
    return np.insert(np.cumsum(dp), 0, 0.0)

# Load one body and one table file for direct comparison
body_file = os.path.join(ultra_dir, 'ultra_rfbody1.h5')
table_file = os.path.join(ultra_dir, 'ultra_rftable1.h5')

# Sampling rate check (Assume 10 kHz based on 500k samples / 50s)
FS = 10000

print(f"Comparing:")
print(f"  Body file: {os.path.basename(body_file)}")
print(f"  Table file: {os.path.basename(table_file)}")

def process_file(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    # I/Q raw
    i_raw, q_raw = data[0, :], data[1, :]
    
    # Fit circle to center I/Q
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Extract magnitude and phase
    mag = np.sqrt(i_c**2 + q_c**2)
    phase = robust_phase(i_c, q_c)
    
    # Detrend phase to remove overall drift
    phase_detrend = phase - np.polyval(np.polyfit(np.arange(len(phase)), phase, 1), np.arange(len(phase)))
    
    return i_raw, q_raw, i_c, q_c, mag, phase_detrend

# Extract for both
b_i_raw, b_q_raw, b_ic, b_qc, b_mag, b_phase = process_file(body_file)
t_i_raw, t_q_raw, t_ic, t_qc, t_mag, t_phase = process_file(table_file)

# Plot Raw IQ constellations
fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
axes[0].scatter(b_i_raw[::10], b_q_raw[::10], s=1, alpha=0.5, color='red')
axes[0].set_title("Body Raw IQ Constellation")
axes[0].set_xlabel("I")
axes[0].set_ylabel("Q")
axes[0].grid(True)
axes[0].axis('equal')

axes[1].scatter(t_i_raw[::10], t_q_raw[::10], s=1, alpha=0.5, color='blue')
axes[1].set_title("Table Raw IQ Constellation")
axes[1].set_xlabel("I")
axes[1].set_ylabel("Q")
axes[1].grid(True)
axes[1].axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_iq_comparison.png"), dpi=200)
plt.close()

# Plot PSD comparison of Phase and Magnitude
fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')

# Phase PSD
f_b_p, psd_b_p = welch(b_phase, fs=FS, nperseg=4096)
f_t_p, psd_t_p = welch(t_phase, fs=FS, nperseg=4096)

axes[0].semilogy(f_b_p, psd_b_p, color='red', label='Body')
axes[0].semilogy(f_t_p, psd_t_p, color='blue', label='Table')
axes[0].set_title("Phase PSD Comparison")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Power Spectral Density")
axes[0].set_xlim([0.1, 500])  # Look up to 500 Hz
axes[0].grid(True, which="both", ls="-")
axes[0].legend()

# Magnitude PSD
f_b_m, psd_b_m = welch(b_mag, fs=FS, nperseg=4096)
f_t_m, psd_t_m = welch(t_mag, fs=FS, nperseg=4096)

axes[1].semilogy(f_b_m, psd_b_m, color='red', label='Body')
axes[1].semilogy(f_t_m, psd_t_m, color='blue', label='Table')
axes[1].set_title("Magnitude PSD Comparison")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power Spectral Density")
axes[1].set_xlim([0.1, 500])
axes[1].grid(True, which="both", ls="-")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_psd_comparison.png"), dpi=200)
plt.close()

print("Saved ultra_iq_comparison.png and ultra_psd_comparison.png")

# Look for peaks in PSD
print("\nPeaks in Body Phase PSD (Freq > 1Hz):")
# Find indices of frequencies in 1 - 200 Hz
mask = (f_b_p >= 1.0) & (f_b_p <= 200.0)
f_range = f_b_p[mask]
psd_range = psd_b_p[mask]
top_indices = np.argsort(psd_range)[::-1][:5]
for idx in top_indices:
    print(f"  Freq: {f_range[idx]:.2f} Hz -> PSD: {psd_range[idx]:.2e}")

print("\nPeaks in Table Phase PSD (Freq > 1Hz):")
mask_t = (f_t_p >= 1.0) & (f_t_p <= 200.0)
f_range_t = f_t_p[mask_t]
psd_range_t = psd_t_p[mask_t]
top_indices_t = np.argsort(psd_range_t)[::-1][:5]
for idx in top_indices_t:
    print(f"  Freq: {f_range_t[idx]:.2f} Hz -> PSD: {psd_range_t[idx]:.2e}")
