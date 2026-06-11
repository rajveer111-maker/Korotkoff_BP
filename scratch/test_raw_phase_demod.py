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

def extract_raw_phase(filepath, t_start=3.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        start_idx = int(t_start * FS)
        end_idx = int(t_end * FS)
        data = f['data'][:, start_idx:end_idx]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS + t_start
    
    # 1. Digital Downconversion
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t)
    
    # 2. Lowpass filter to isolate the carrier (cutoff 10 Hz)
    iq_baseband = lowpass_filter_sos(iq_shifted, 10.0, FS, order=2)
    
    # 3. Phase extraction and unwrapping
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # 4. Detrend using a 3rd-order polynomial to remove slow drift
    p = np.polyfit(t, phase, 3)
    phase_detrended = phase - np.polyval(p, t)
    
    # Convert to displacement
    disp_um = phase_detrended * SCALE_FACTOR
    
    return t, disp_um

# Get raw phase displacement
t_b, disp_b = extract_raw_phase(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
t_t, disp_t = extract_raw_phase(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_b, disp_b, color='teal', linewidth=1.5)
axes[0].set_title("Body 2 (Radial Artery) - Raw Detrended Phase Displacement (No Bandpass)", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t, disp_t, color='crimson', linewidth=1.5)
axes[1].set_title("Table 2 (Static Table Control) - Raw Detrended Phase Displacement (No Bandpass)", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Raw Detrended Phase Displacement (900 MHz RF, 3.0s - 8.0s)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_raw_phase_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_raw_phase_comparison.png")
