import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"
FS = 10000

all_files = [
    ('Table 1', 'ultra_rftable1.h5', 5, 25),
    ('Body 1', 'ultra_rfbody01.h5', 28, 45),
    ('Table 2', 'ultra_rftable2.h5', 2, 9),
    ('Body 2', 'ultra_rfbody1.h5', 2, 9),
    ('Table 3', 'ultra_rftable3.h5', 2, 9),
    ('Body 3', 'ultra_rfbody2.h5', 2, 9),
    ('Table 4', 'ultra_rftable4.h5', 2, 9),
    ('Body 4', 'ultra_rfbody3.h5', 2, 9),
]

fig, axes = plt.subplots(4, 2, figsize=(15, 18), facecolor='white')
axes = axes.flatten()

for idx, (name, filename, t_start, t_end) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    iq_raw = data[0, :] + 1j * data[1, :]
    
    active_sig = iq_raw[int(t_start * FS) : int(t_end * FS)]
    
    # Compute FFT
    N = len(active_sig)
    fft_vals = np.fft.fftshift(np.fft.fft(active_sig))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/FS))
    
    power_db = 10 * np.log10(np.abs(fft_vals)**2 + 1e-10)
    
    # Filter frequencies between -600 and +600
    mask = (fft_freqs >= -600) & (fft_freqs <= 600)
    
    axes[idx].plot(fft_freqs[mask], power_db[mask], color='navy', linewidth=1.0)
    axes[idx].set_title(f"{name} Spectrum ({filename})", fontsize=11, weight='bold')
    axes[idx].set_xlabel("Frequency (Hz)")
    axes[idx].set_ylabel("Power (dB)")
    axes[idx].grid(True, alpha=0.3)
    
    # Mark harmonics
    for h in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        target_f = h * 100.714
        axes[idx].axvline(target_f, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

plt.suptitle("Raw RF Signal Power Spectra (-600 Hz to +600 Hz)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(out_dir, "ultra_full_spectra_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_full_spectra_comparison.png")
