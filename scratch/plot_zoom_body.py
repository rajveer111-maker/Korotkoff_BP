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
FC = -100.714  # Carrier frequency
SCALE_FACTOR = 333333.3 / (4 * np.pi)

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def extract_zoom_phase(filepath, t_start=3.0, t_end=8.5, plot_start=4.5, plot_end=7.5):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC on full signal
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # Lowpass filter at 8 Hz (retains heartbeat and removes high-frequency jitter)
    iq_baseband = lowpass_filter_sos(iq_shifted, 8.0, FS, order=2)
    
    # Phase extraction
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to wider window for detrending
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Detrend using a 3rd-order polynomial
    p = np.polyfit(t_crop, phase_crop, 3)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    disp_um = phase_detrended * SCALE_FACTOR
    
    # Crop to plotting window to avoid polynomial edge distortion
    idx_plot = (t_crop >= plot_start) & (t_crop <= plot_end)
    return t_crop[idx_plot], disp_um[idx_plot]

# Get signals
t_b, disp_b = extract_zoom_phase(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, disp_t = extract_zoom_phase(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_b, disp_b, color='teal', linewidth=2.0)
axes[0].set_title("Body 2 (Radial Artery) - Physiological Displacement Zoom (4.5s - 7.5s)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

# Add markers for potential heartbeat cycles
# Let's see: peaks look like they are spaced by ~0.7 to 0.9s
axes[1].plot(t_t, disp_t, color='crimson', linewidth=2.0)
axes[1].set_title("Table 2 (Static Table Control) - Static Baseline Noise Zoom (4.5s - 7.5s)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("High-Fidelity Demodulation Zoom (900 MHz RF, 4.5s - 7.5s)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_zoom_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_zoom_comparison.png")
