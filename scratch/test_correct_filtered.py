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
SCALE_FACTOR = 333333.3 / (4 * np.pi)  # 900 MHz scaling

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def bandpass_filter_sos(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def extract_correct_phase(filepath, t_start=3.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC on full signal
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # Lowpass filter full signal (cutoff 15 Hz to retain heartbeat)
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    # Phase extraction and unwrapping on full signal
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to stable window [t_start, t_end]
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Detrend cropped phase
    p = np.polyfit(t_crop, phase_crop, 2)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Convert to displacement
    disp_um = phase_detrended * SCALE_FACTOR
    
    # Bandpass filter cropped signal (0.8 - 2.5 Hz)
    disp_filt = bandpass_filter_sos(disp_um, 0.8, 2.5, FS, order=2)
    
    return t_crop, disp_filt

# Get displacement
t_b, disp_b = extract_correct_phase(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, disp_t = extract_correct_phase(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot (zoom in to 4.0s - 7.0s to avoid the bandpass filter startup at 3.0s!)
idx_zoom_b = (t_b >= 4.0) & (t_b <= 7.0)
idx_zoom_t = (t_t >= 4.0) & (t_t <= 7.0)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_b[idx_zoom_b], disp_b[idx_zoom_b], color='teal', linewidth=2.0)
axes[0].set_title("Body 2 (Radial Artery) - Bandpass Filtered Heartbeat (0.8-2.5 Hz)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t[idx_zoom_t], disp_t[idx_zoom_t], color='crimson', linewidth=2.0)
axes[1].set_title("Table 2 (Static Table Control) - Bandpass Filtered Noise (0.8-2.5 Hz)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Correct Bandpass Filtered Comparison (900 MHz RF, 4.0s - 7.0s)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_correct_filtered_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_correct_filtered_comparison.png")
