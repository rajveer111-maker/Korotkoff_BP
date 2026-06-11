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

# Let's inspect raw IQ signals
# We'll plot 0.2 seconds (2000 samples)
start_sample = 20 * FS
end_sample = int(20.2 * FS)
t = np.arange(start_sample, end_sample) / FS

files_to_inspect = {
    'Body 1': 'ultra_rfbody1.h5',
    'Table 1': 'ultra_rftable1.h5',
    'Table 2': 'ultra_rftable2.h5',
}

fig, axes = plt.subplots(len(files_to_inspect), 2, figsize=(14, 10), facecolor='white')

for i, (name, filename) in enumerate(files_to_inspect.items()):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, start_sample:end_sample]
        # Let's also read a larger segment to compute a spectrogram
        spec_data = f['data'][:, start_sample:start_sample + 50000]
        
    i_raw, q_raw = data[0, :], data[1, :]
    
    # Plot raw IQ time series
    axes[i, 0].plot(t, i_raw, color='black', label='I', alpha=0.8)
    axes[i, 0].plot(t, q_raw, color='gray', label='Q', alpha=0.8)
    axes[i, 0].set_title(f"{name} ({filename}) - Raw I/Q Time Series (0.2s)", fontsize=11, weight='bold')
    axes[i, 0].set_xlabel("Time (s)")
    axes[i, 0].set_ylabel("Amplitude (V)")
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend()
    
    # Compute spectrogram of complex signal I + j*Q
    iq_complex = spec_data[0, :] + 1j * spec_data[1, :]
    f_spec, t_spec, Sxx = spectrogram(iq_complex, fs=FS, nperseg=256, noverlap=128, return_onesided=False)
    
    # Shift frequency to center it around 0
    f_spec = np.fft.fftshift(f_spec)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Plot spectrogram
    im = axes[i, 1].pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    axes[i, 1].set_title(f"{name} ({filename}) - Spectrogram of I+jQ", fontsize=11, weight='bold')
    axes[i, 1].set_xlabel("Time (s)")
    axes[i, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[i, 1], label='Power (dB)')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_raw_iq_time_spectrogram.png"), dpi=200)
plt.close()
print("Saved ultra_raw_iq_time_spectrogram.png")
