import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000

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

fig, axes = plt.subplots(4, 2, figsize=(15, 16), facecolor='white')
axes = axes.flatten()

for idx, (name, filename) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        # Load first 30 seconds (or full file if shorter)
        data = f['data'][:, :30 * FS]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    
    # Compute spectrogram of raw complex IQ signal
    f_spec, t_spec, Sxx = spectrogram(iq_raw, fs=FS, nperseg=1024, noverlap=512, return_onesided=False)
    
    # Shift frequency axis to center around 0 Hz
    f_spec = np.fft.fftshift(f_spec)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    # Plot spectrogram (focus on -1000 to +1000 Hz)
    im = axes[idx].pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
    axes[idx].set_ylim([-1000, 1000])
    axes[idx].set_title(f"{name} - Raw I/Q Spectrogram", fontsize=11, weight='bold')
    axes[idx].set_ylabel("Frequency (Hz)")
    axes[idx].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[idx], label='Power (dB)')

plt.suptitle("Raw I/Q Spectrograms (-1000 to +1000 Hz)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_raw_spectrograms.png"), dpi=150)
plt.close()
print("Saved ultra_raw_spectrograms.png")
