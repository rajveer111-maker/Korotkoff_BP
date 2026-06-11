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

def get_raw_phase_psd(filepath, t_start=3.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    # DDC on full signal
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    
    # We do NOT apply any lowpass filter to the I/Q signal before phase extraction
    # so we don't introduce any filter bias!
    # Just take the angle of the shifted I/Q signal.
    # Note: since the carrier is shifted to 0 Hz, the angle represents the phase directly.
    phase = np.unwrap(np.angle(iq_shifted))
    
    # Crop to stable window
    idx_crop = (t_full >= t_start) & (t_full <= t_end)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Detrend using a 3rd-order polynomial to remove slow drift
    p = np.polyfit(t_crop, phase_crop, 3)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    # Compute PSD using Welch's method with high resolution
    # Nfft = 4096 gives a frequency resolution of 10000 / 4096 = 2.44 Hz.
    # For better resolution in the 0.1-20 Hz range, we can downsample the phase signal first!
    # Downsample by 100x to 100 Hz sampling rate.
    downsample_factor = 100
    phase_downsampled = phase_detrended[::downsample_factor]
    fs_down = FS / downsample_factor  # 100 Hz
    
    # Now Nfft = 512 gives a frequency resolution of 100 / 512 = 0.19 Hz!
    f_psd, p_psd = welch(phase_downsampled, fs=fs_down, nperseg=len(phase_downsampled), noverlap=0, nfft=1024)
    
    return f_psd, p_psd

# Compute PSDs
f_b, p_b = get_raw_phase_psd(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
f_t, p_t = get_raw_phase_psd(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot
plt.figure(figsize=(10, 6), facecolor='white')
plt.semilogy(f_b, p_b, color='teal', label='Body 2 (Radial Artery)', linewidth=2.0)
plt.semilogy(f_t, p_t, color='crimson', label='Table 2 (Static Table Control)', linewidth=2.0)

plt.xlim(0.1, 15.0)
plt.xlabel("Frequency (Hz)", fontsize=12, weight='bold')
plt.ylabel("Phase Power Spectral Density (rad²/Hz)", fontsize=12, weight='bold')
plt.title("Raw Phase Power Spectrum Comparison (0.1 Hz to 15 Hz)", fontsize=14, weight='bold')
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_raw_phase_spectrum.png"), dpi=200)
plt.close()
print("Saved ultra_raw_phase_spectrum.png")
