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
FC = -100.714
SCALE_FACTOR = 333333.3 / (4 * np.pi)

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def get_true_heartbeat(filepath, t_start=3.0, t_end=8.0, plot_start=3.5, plot_end=7.5):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # 1. DDC
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # 2. Lowpass filter at 2.5 Hz (very small time constant, no ringing after 0.25s)
    iq_baseband = lowpass_filter_sos(iq_shifted, 2.5, FS, order=2)
    
    # 3. Phase extraction and unwrapping
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # 4. Crop to stable window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # 5. Detrend using 4th-order polynomial (removes respiration wave)
    p = np.polyfit(t_crop, phase_crop, 4)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Convert to physical displacement in micrometers
    disp_um = phase_detrended * SCALE_FACTOR
    
    # Crop to plotting window to avoid polynomial edge transients
    idx_plot = (t_crop >= plot_start) & (t_crop <= plot_end)
    return t_crop[idx_plot], disp_um[idx_plot]

# Get signals
t_b, disp_b = get_true_heartbeat(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, disp_t = get_true_heartbeat(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_b, disp_b, color='teal', linewidth=2.0)
axes[0].set_title("Body 2 (Radial Artery) - True Low-Frequency Displacement (No Highpass Ringing)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t, disp_t, color='crimson', linewidth=2.0)
axes[1].set_title("Table 2 (Static Table Control) - True Low-Frequency Displacement (No Highpass Ringing)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Physiological Heartbeat Extraction (2.5 Hz Lowpass + Poly Detrend, 3.5s - 7.5s)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_true_heartbeat_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_true_heartbeat_comparison.png")
