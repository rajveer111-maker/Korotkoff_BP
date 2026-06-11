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
F0 = 100.714
SCALE_FACTOR = 333333.3 / (4 * np.pi)

all_files = [
    ('Table 1', 'ultra_rftable1.h5'),
    ('Body 1', 'ultra_rfbody01.h5'),
    ('Table 2', 'ultra_rftable2.h5'),
    ('Body 2', 'ultra_rfbody1.h5'),
    ('Table 3', 'ultra_rftable3.h5'),
    ('Body 3', 'ultra_rfbody2.h5'),
    ('Table 4', 'ultra_rftable4.h5'),
    ('Body 4', 'ultra_rfbody3.h5'),
]

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def detect_active_window(filepath, thresh_frac=0.15):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC at F0 to check magnitude
    iq_shifted = iq_raw * np.exp(1j * 2 * np.pi * F0 * t_full) # try positive first
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    mag = np.abs(iq_baseband)
    
    # Also check negative F0
    iq_shifted_neg = iq_raw * np.exp(-1j * 2 * np.pi * F0 * t_full)
    iq_baseband_neg = lowpass_filter_sos(iq_shifted_neg, 15.0, FS, order=2)
    mag_neg = np.abs(iq_baseband_neg)
    
    if np.max(mag_neg) > np.max(mag):
        mag = mag_neg
        fc_sign = -1.0
    else:
        fc_sign = 1.0
        
    max_mag = np.max(mag)
    active_idx = np.where(mag > thresh_frac * max_mag)[0]
    
    if len(active_idx) == 0:
        # Fallback to defaults
        return 3.0, 8.0, fc_sign
        
    t_start = t_full[active_idx[0]]
    t_end = t_full[active_idx[-1]]
    
    # Add a safety margin (crop 0.5s from start and end to avoid transients)
    t_start_safe = t_start + 0.5
    t_end_safe = t_end - 0.5
    
    # Ensure window is at least 3 seconds long
    if t_end_safe - t_start_safe < 3.0:
        t_start_safe = t_start
        t_end_safe = t_end
        
    return t_start_safe, t_end_safe, fc_sign

def process_file(filepath, name, t_start, t_end, fc_sign, harmonic_order=4):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC
    fc = fc_sign * harmonic_order * F0
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * fc * t_full)
    
    # Lowpass filter at 2.5 Hz
    iq_baseband = lowpass_filter_sos(iq_shifted, 2.5, FS, order=2)
    
    # Phase extraction
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Crop to active window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Detrend using 4th-order polynomial
    p = np.polyfit(t_crop, phase_crop, 4)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Scale to displacement
    disp_um = (phase_detrended / harmonic_order) * SCALE_FACTOR
    
    # Return time axis relative to start of window
    t_rel = t_crop - t_start
    return t_rel, disp_um

# Process and plot all 8 files
fig, axes = plt.subplots(4, 2, figsize=(15, 18), facecolor='white')
axes = axes.flatten()

print("Processing files...")
for idx, (name, filename) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    t_start, t_end, fc_sign = detect_active_window(filepath)
    print(f"{name}: Active window detected as {t_start:.2f}s to {t_end:.2f}s (sign={fc_sign})")
    
    t_rel, disp = process_file(filepath, name, t_start, t_end, fc_sign, harmonic_order=4)
    
    # We want to plot the middle 4 seconds of the detected window to avoid boundary effects
    mid_t = (t_rel[-1] - t_rel[0]) / 2.0
    plot_mask = (t_rel >= mid_t - 2.0) & (t_rel <= mid_t + 2.0)
    
    t_plot = t_rel[plot_mask] - (mid_t - 2.0)
    disp_plot = disp[plot_mask]
    
    color = 'teal' if 'Body' in name else 'crimson'
    axes[idx].plot(t_plot, disp_plot, color=color, linewidth=2.0)
    axes[idx].set_title(f"{name} (4th Harmonic, {filename})", fontsize=12, weight='bold')
    axes[idx].set_ylabel("Displacement (µm)")
    axes[idx].set_xlabel("Relative Time (s)")
    axes[idx].grid(True, alpha=0.3)
    
    # Calculate statistics
    std_val = np.std(disp_plot)
    p2p_val = np.max(disp_plot) - np.min(disp_plot)
    axes[idx].text(0.05, 0.9, f"RMS: {std_val:.2f} µm\nP2P: {p2p_val:.2f} µm", 
                   transform=axes[idx].transAxes, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.suptitle("Master Acousto-RF Demodulated Displacements (4th Harmonic, 2.5 Hz Lowpass, 4.0s Window)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(out_dir, "ultra_master_demod_4f0.png"), dpi=200)
plt.close()
print("Saved ultra_master_demod_4f0.png")
