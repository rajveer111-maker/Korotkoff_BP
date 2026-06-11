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
FC = -100.714

def lowpass_filter_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def process_phase_psd(filepath, t_start=4.0, t_end=8.0):
    with h5py.File(filepath, 'r') as f:
        start_idx = int(t_start * FS)
        end_idx = int(t_end * FS)
        data = f['data'][:, start_idx:end_idx]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS + t_start
    
    # 1. Digital Downconversion
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t)
    
    # 2. Lowpass filter (cutoff 5 Hz)
    iq_baseband = lowpass_filter_sos(iq_shifted, 5.0, FS)
    
    # 3. Extract phase and unwrap
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # 4. Detrend phase using 2nd-order polynomial
    p = np.polyfit(t, phase, 2)
    phase_detrended = phase - np.polyval(p, t)
    
    # 5. Convert to displacement in micrometers
    disp_um = phase_detrended * 10000
    
    # 6. Compute PSD
    f_welch, p_welch = welch(disp_um, fs=FS, nperseg=int(2.0*FS)) # 2-second windows for 0.5 Hz resolution
    
    return f_welch, p_welch

# Process
f_b, p_b = process_phase_psd(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
f_t, p_t = process_phase_psd(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

# Plot
plt.figure(figsize=(10, 6), facecolor='white')
plt.semilogy(f_b, p_b, color='teal', linewidth=2.0, label='Body 2 (Radial Artery)')
plt.semilogy(f_t, p_t, color='crimson', linewidth=2.0, label='Table 2 (Static Table Baseline)')
plt.xlim([0.1, 5.0])
plt.ylim([1e-3, 1e8])
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("PSD (µm²/Hz)", fontsize=12)
plt.title("PSD of Demodulated Carrier Phase: Body 2 vs Table 2 (4.0s - 8.0s)", fontsize=14, weight='bold')
plt.legend(fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_phase_psd_stable_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_phase_psd_stable_comparison.png")
