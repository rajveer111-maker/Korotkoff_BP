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

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def get_raw_phase(filepath, t_start=3.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC on full signal
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # Lowpass filter full signal to isolate carrier (15 Hz)
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    # Phase extraction and unwrapping on full signal
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to stable window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    return t_full[idx_crop], phase[idx_crop]

t_b, phase_b = get_raw_phase(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, phase_t = get_raw_phase(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')
axes[0].plot(t_b, phase_b, color='teal', linewidth=1.5)
axes[0].set_title("Body 2 Raw Unwrapped Phase (3.0s - 8.0s)", fontsize=12, weight='bold')
axes[0].set_ylabel("Phase (rad)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t, phase_t, color='crimson', linewidth=1.5)
axes[1].set_title("Table 2 Raw Unwrapped Phase (3.0s - 8.0s)", fontsize=12, weight='bold')
axes[1].set_ylabel("Phase (rad)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Raw Unwrapped Phase Comparison (No Detrending)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_raw_phase_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_raw_phase_comparison.png")
