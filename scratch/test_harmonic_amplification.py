import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000
F0 = 100.714  # Fundamental carrier frequency
SCALE_FACTOR = 333333.3 / (4 * np.pi)

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def demodulate_harmonic(filepath, harmonic_order, t_start=3.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC at n * F0
    fc = -harmonic_order * F0
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * fc * t_full)
    
    # Lowpass filter to isolate the harmonic carrier
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    # Phase extraction and unwrapping
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to stable window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Linear detrending (1st-order poly) to keep the respiration wave visible
    p = np.polyfit(t_crop, phase_crop, 1)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Physical displacement: divide phase by harmonic_order to normalize
    disp_um = (phase_detrended / harmonic_order) * SCALE_FACTOR
    
    return t_crop, disp_um

# Get demodulated signals
t_1, disp_1 = demodulate_harmonic(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 1)
t_3, disp_3 = demodulate_harmonic(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 3)

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_1, disp_1, color='teal', linewidth=1.5)
axes[0].set_title("Body 2 - Demodulation at Fundamental Carrier (1f0 = 100.71 Hz)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_3, disp_3, color='indigo', linewidth=1.5)
axes[1].set_title("Body 2 - Demodulation at 3rd Harmonic (3f0 = 302.14 Hz)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Harmonic Phase Amplification Comparison (Body 2)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_harmonic_amplification_comparison.png"), dpi=200)
plt.close()

# Calculate statistics of noise/fluctuations
std_1 = np.std(disp_1)
std_3 = np.std(disp_3)
print(f"Fundamental 1f0 displacement RMS: {std_1:.3f} um")
print(f"3rd Harmonic 3f0 displacement RMS: {std_3:.3f} um")
print(f"Ratio of RMS (3f0 / 1f0): {std_3 / std_1:.3f}")
