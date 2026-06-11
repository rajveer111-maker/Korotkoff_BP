import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000
FC = -100.714  # Carrier frequency

def lowpass_filter_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def bandpass_filter_sos(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def process_file_v2(filepath, t_start=1.0, t_end=9.0, plot_start=4.0, plot_end=7.0):
    with h5py.File(filepath, 'r') as f:
        start_idx = int(t_start * FS)
        end_idx = int(t_end * FS)
        data = f['data'][:, start_idx:end_idx]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS + t_start
    
    # 1. Digital Downconversion
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t)
    
    # 2. Lowpass filter (cutoff 5 Hz)
    iq_baseband = lowpass_filter_sos(iq_shifted, 5.0, FS)
    
    # 3. Extract phase and unwrap
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # 4. Detrend phase using 2nd-order polynomial
    p = np.polyfit(t, phase, 2)
    phase_detrended = phase - np.polyval(p, t)
    
    # 5. Convert to displacement in micrometers (lambda = 12.5 cm -> d = phase * 10000 um)
    disp_um = phase_detrended * 10000
    
    # 6. Bandpass filter in heartbeat range (0.8 to 2.5 Hz)
    disp_filt = bandpass_filter_sos(disp_um, 0.8, 2.5, FS)
    
    # Crop to plotting region to remove transients
    idx_plot = (t >= plot_start) & (t <= plot_end)
    return t[idx_plot], disp_filt[idx_plot]

# Process Body 2 and Table 2
t_b, disp_b = process_file_v2(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, disp_t = process_file_v2(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot time-domain comparison (individual y-scales)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

# Body 2
axes[0].plot(t_b, disp_b, color='teal', linewidth=2.0)
axes[0].set_title("Body 2 (Radial Artery) - Demodulated Carrier Phase Displacement (4.0s - 7.0s)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

# Table 2
axes[1].plot(t_t, disp_t, color='crimson', linewidth=2.0)
axes[1].set_title("Table 2 (Static Table Baseline) - Demodulated Carrier Phase Displacement (4.0s - 7.0s)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Micro-Displacement Extraction from -100.71 Hz Ultrasound Carrier", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_demodulated_heartbeat_comparison_v2.png"), dpi=200)
plt.close()
print("Saved ultra_demodulated_heartbeat_comparison_v2.png")
