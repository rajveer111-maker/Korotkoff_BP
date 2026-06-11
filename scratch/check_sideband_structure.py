import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"
FS = 10000

def get_high_res_spectrum(filepath, t_start=4.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        start_idx = int(t_start * FS)
        end_idx = int(t_end * FS)
        data = f['data'][:, start_idx:end_idx]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    N = len(iq_raw)
    
    # Apply a Blackman-Harris window to suppress sidelobes and see sidebands clearly
    win = windows.blackmanharris(N)
    iq_win = iq_raw * win
    
    # Zero-pad to get high resolution in frequency domain
    n_fft = 2**18  # ~262k points, df = 10000 / 262144 = 0.038 Hz
    fft_vals = np.fft.fft(iq_win, n=n_fft)
    fft_freqs = np.fft.fftfreq(n_fft, 1/FS)
    
    fft_vals = np.fft.fftshift(fft_vals)
    fft_freqs = np.fft.fftshift(fft_freqs)
    
    power = np.abs(fft_vals)**2
    # Find peak around -100.7 Hz
    idx_search = (fft_freqs > -102.0) & (fft_freqs < -99.5)
    peak_idx = np.where(idx_search)[0][np.argmax(power[idx_search])]
    peak_freq = fft_freqs[peak_idx]
    peak_val = power[peak_idx]
    
    psd_db = 10 * np.log10(power / peak_val + 1e-12)
    return fft_freqs, psd_db, peak_freq

f_b, psd_b, peak_b = get_high_res_spectrum(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
f_t, psd_t, peak_t = get_high_res_spectrum(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot zoom around the peak
plt.figure(figsize=(12, 6), facecolor='white')
plt.plot(f_b - peak_b, psd_b, color='teal', linewidth=1.5, label=f'Body 2 (Peak: {peak_b:.3f} Hz)')
plt.plot(f_t - peak_t, psd_t, color='crimson', linewidth=1.5, alpha=0.8, label=f'Table 2 (Peak: {peak_t:.3f} Hz)')

plt.xlim([-5.0, 5.0])
plt.ylim([-60, 0])
plt.xlabel("Offset from Carrier (Hz)", fontsize=12)
plt.ylabel("Normalized Power (dB)", fontsize=12)
plt.title("High-Resolution Sideband Analysis of the Ultrasound Carrier", fontsize=14, weight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_high_res_sidebands.png"), dpi=200)
plt.close()
print("Saved ultra_high_res_sidebands.png")
