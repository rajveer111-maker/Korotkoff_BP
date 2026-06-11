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

def get_magnitude(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC on full signal
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # Lowpass filter to isolate the carrier
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    magnitude = np.abs(iq_baseband)
    return t_full, magnitude

t_b, mag_b = get_magnitude(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, mag_t = get_magnitude(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')
axes[0].plot(t_b, mag_b, color='teal', linewidth=1.5)
axes[0].set_title("Body 2 Carrier Magnitude Envelope", fontsize=12, weight='bold')
axes[0].set_ylabel("Magnitude (V)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t, mag_t, color='crimson', linewidth=1.5)
axes[1].set_title("Table 2 Carrier Magnitude Envelope", fontsize=12, weight='bold')
axes[1].set_ylabel("Magnitude (V)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Carrier Magnitude Comparison (100.71 Hz Carrier)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_carrier_magnitude_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_carrier_magnitude_comparison.png")
