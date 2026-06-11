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
    
    # DDC at n * F0 (negative sign)
    fc = -harmonic_order * F0
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * fc * t_full)
    
    # Lowpass filter to isolate the harmonic carrier
    iq_baseband = lowpass_filter_sos(iq_shifted, 2.5, FS, order=2)
    
    # Phase extraction and unwrapping
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to stable window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Detrend using 4th-order polynomial
    p = np.polyfit(t_crop, phase_crop, 4)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Physical displacement: divide phase by harmonic_order to normalize
    disp_um = (phase_detrended / harmonic_order) * SCALE_FACTOR
    
    return t_crop, disp_um

# Get demodulated signals
t_b, disp_b = demodulate_harmonic(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 4)
t_t, disp_t = demodulate_harmonic(os.path.join(ultra_dir, 'ultra_rftable2.h5'), 4)

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')

axes[0].plot(t_b, disp_b, color='darkviolet', linewidth=2.0)
axes[0].set_title("Body 2 - 4th Harmonic (4f0 = 402.86 Hz) Demodulated Displacement", fontsize=12, weight='bold')
axes[0].set_ylabel("Displacement (µm)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_t, disp_t, color='crimson', linewidth=2.0)
axes[1].set_title("Table 2 (Static Control) - 4th Harmonic (4f0 = 402.86 Hz) Demodulated Displacement", fontsize=12, weight='bold')
axes[1].set_ylabel("Displacement (µm)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.suptitle("4th Harmonic Displacement Comparison: Body vs Table Control", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_4f0_body_vs_table.png"), dpi=200)
plt.close()

# Calculate statistics
std_b = np.std(disp_b)
std_t = np.std(disp_t)
print(f"Body 2 4f0 displacement RMS: {std_b:.3f} um")
print(f"Table 2 4f0 displacement RMS: {std_t:.3f} um")
print(f"Ratio of RMS (Table / Body): {std_t / std_b:.3f}")
