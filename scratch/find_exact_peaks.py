import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
filename = "ultra_rfbody1.h5"
filepath = os.path.join(ultra_dir, filename)

FS = 10000

with h5py.File(filepath, 'r') as f:
    data = f['data'][:]

iq_raw = data[0, :] + 1j * data[1, :]

# Active: 0 to 10s
iq_active = iq_raw[:10 * FS]
# Quiet: 10 to 40s
iq_quiet = iq_raw[10 * FS:40 * FS]

def print_top_peaks(signal, fs, segment_name):
    # Compute FFT
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    
    # Compute power
    power = np.abs(fft_vals)**2
    
    # Sort peaks
    # Filter out DC (freq = 0)
    non_dc_idx = np.abs(fft_freqs) > 0.5
    freqs_ndc = fft_freqs[non_dc_idx]
    power_ndc = power[non_dc_idx]
    
    sorted_idx = np.argsort(power_ndc)[::-1]
    
    print(f"\n--- Top Peaks in {segment_name} ---")
    print(f"{'Rank':<5}{'Frequency (Hz)':<20}{'Power (dB)':<15}")
    for rank in range(10):
        idx = sorted_idx[rank]
        print(f"{rank+1:<5}{freqs_ndc[idx]:<20.2f}{10*np.log10(power_ndc[idx]):<15.2f}")

print_top_peaks(iq_active, FS, "Active (0-10s)")
print_top_peaks(iq_quiet, FS, "Quiet (10-40s)")
