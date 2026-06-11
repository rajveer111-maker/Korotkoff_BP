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

def bandpass_filter_sos(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def get_signal_with_cutoff(filepath, lowcut, highcut):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # 1. DDC
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # 2. Lowpass filter
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    # 3. Phase extraction and unwrapping
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # 4. Bandpass filter
    phase_filt = bandpass_filter_sos(phase, lowcut, highcut, FS, order=2)
    disp_filt_um = phase_filt * SCALE_FACTOR
    
    # Crop to 4.0s - 7.0s
    idx_crop = (t_full >= 4.0) & (t_full <= 7.0)
    return t_full[idx_crop], disp_filt_um[idx_crop]

# Process Body 2 with two different cutoffs
t_1, disp_1 = get_signal_with_cutoff(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 0.8, 2.5)
t_2, disp_2 = get_signal_with_cutoff(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 1.3, 3.0)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')
axes[0].plot(t_1, disp_1, color='teal', linewidth=2.0)
axes[0].set_title("Body 2 - Bandpass 0.8 - 2.5 Hz", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_2, disp_2, color='navy', linewidth=2.0)
axes[1].set_title("Body 2 - Bandpass 1.3 - 3.0 Hz", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Filter Resonance Test: Frequency Shift Analysis", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_filter_resonance_test.png"), dpi=200)
plt.close()
print("Saved ultra_filter_resonance_test.png")
