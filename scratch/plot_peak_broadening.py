import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000
F0 = 100.714

def get_spectrum(filepath, start_t, end_t):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, int(start_t*FS):int(end_t*FS)]
    iq_raw = data[0, :] + 1j * data[1, :]
    N = len(iq_raw)
    fft_vals = np.fft.fft(iq_raw)
    fft_freqs = np.fft.fftfreq(N, 1/FS)
    # Shift
    fft_vals = np.fft.fftshift(fft_vals)
    fft_freqs = np.fft.fftshift(fft_freqs)
    # Normalize power to peak
    power = np.abs(fft_vals)**2
    power_db = 10 * np.log10(power / np.max(power) + 1e-12)
    return fft_freqs, power_db

# Compare Table 2 vs Body 2 around -100.71 Hz
f_t2, p_t2 = get_spectrum(os.path.join(ultra_dir, 'ultra_rftable2.h5'), 2.0, 9.0)
f_b2, p_b2 = get_spectrum(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 2.0, 9.0)

# Compare Table 4 vs Body 4 around -100.71 Hz
f_t4, p_t4 = get_spectrum(os.path.join(ultra_dir, 'ultra_rftable4.h5'), 2.0, 9.0)
f_b4, p_b4 = get_spectrum(os.path.join(ultra_dir, 'ultra_rfbody3.h5'), 2.0, 9.0)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')

# Plot 1: Table 2 vs Body 2
axes[0].plot(f_t2, p_t2, color='crimson', label='Table 2 (Static Table)', linewidth=1.5)
axes[0].plot(f_b2, p_b2, color='teal', label='Body 2 (Physiological Target)', linewidth=1.5)
axes[0].axvline(-F0, color='black', linestyle='--', alpha=0.5, label='Carrier Center (-100.71 Hz)')
axes[0].set_xlim([-103.0, -98.0])
axes[0].set_ylim([-50, 2])
axes[0].set_title("Carrier Line Shape Comparison: Table 2 vs Body 2", fontsize=12, weight='bold')
axes[0].set_ylabel("Normalized Power (dB)")
axes[0].legend(fontsize=10, loc='lower left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Table 4 vs Body 4
axes[1].plot(f_t4, p_t4, color='crimson', label='Table 4 (Static Table)', linewidth=1.5)
axes[1].plot(f_b4, p_b4, color='teal', label='Body 4 (Physiological Target)', linewidth=1.5)
axes[1].axvline(-F0, color='black', linestyle='--', alpha=0.5, label='Carrier Center (-100.71 Hz)')
axes[1].set_xlim([-103.0, -98.0])
axes[1].set_ylim([-50, 2])
axes[1].set_title("Carrier Line Shape Comparison: Table 4 vs Body 4", fontsize=12, weight='bold')
axes[1].set_ylabel("Normalized Power (dB)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].legend(fontsize=10, loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.suptitle("Spectral Line Broadening: Ultrasound Carrier Modulated by Body Motion", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_carrier_spectral_broadening.png"), dpi=200)
plt.close()
print("Saved ultra_carrier_spectral_broadening.png")
