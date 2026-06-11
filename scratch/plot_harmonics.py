import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000
F0 = 100.714  # Fundamental frequency

all_plots = [
    ('Table 1', 'ultra_rftable1.h5', 5.0, 15.0),
    ('Body 1', 'ultra_rfbody01.h5', 28.0, 38.0),
    ('Body 2', 'ultra_rfbody1.h5', 2.0, 9.0),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='white')

for idx, (name, filename, start_t, end_t) in enumerate(all_plots):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, int(start_t*FS):int(end_t*FS)]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    
    # Compute FFT
    N = len(iq_raw)
    fft_vals = np.fft.fft(iq_raw)
    fft_freqs = np.fft.fftfreq(N, 1/FS)
    
    # Shift
    fft_vals = np.fft.fftshift(fft_vals)
    fft_freqs = np.fft.fftshift(fft_freqs)
    
    power_db = 10 * np.log10(np.abs(fft_vals)**2 + 1e-12)
    
    # Plot PSD
    axes[idx].plot(fft_freqs, power_db, color='navy', linewidth=1.0)
    
    # Mark harmonics
    harmonics = [-4*F0, -3*F0, -2*F0, -F0, 0, F0, 2*F0, 3*F0, 4*F0]
    for h in harmonics:
        axes[idx].axvline(h, color='red', linestyle='--', alpha=0.6, linewidth=0.8)
        # Label the harmonic
        h_order = int(round(h / F0))
        if h_order != 0:
            axes[idx].text(h, ax_ylim := axes[idx].get_ylim()[1] - 5, f"{h_order}f₀", 
                           color='red', fontsize=8, ha='center', va='top', weight='bold')

    axes[idx].set_xlim([-600, 100])
    axes[idx].set_title(f"{name} Spectrum - Active Phase showing Ultrasound Harmonics (f₀ = {F0:.2f} Hz)", fontsize=12, weight='bold')
    axes[idx].set_ylabel("Power (dB)")
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel("Frequency (Hz)")
plt.suptitle("RF Spectrum Analysis: Non-linear Ultrasound-RF Phase Modulation", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_rf_spectrum_harmonics.png"), dpi=200)
plt.close()
print("Saved ultra_rf_spectrum_harmonics.png")
