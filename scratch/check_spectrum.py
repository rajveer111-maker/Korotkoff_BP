import h5py
import numpy as np
import os
from scipy import signal

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 

def analyze():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    magnitude = np.sqrt(data[0,:]**2 + data[1,:]**2)
    
    # Calculate PSD
    freqs, psd = signal.welch(magnitude - np.mean(magnitude), fs, nperseg=fs*2)
    
    # Check for 50Hz peak
    idx_50 = np.argmin(np.abs(freqs - 50))
    psd_50 = psd[idx_50]
    psd_max_koro = np.max(psd[(freqs >= 10) & (freqs <= 45)])
    
    print(f"PSD at 50Hz: {psd_50:.6e}")
    print(f"Max PSD in 10-45Hz: {psd_max_koro:.6e}")
    
    if psd_50 > psd_max_koro * 10:
        print("WARNING: Strong 50Hz interference detected.")
    
    # Find dominant frequency in koro band
    koro_indices = (freqs >= 10) & (freqs <= 50)
    dom_freq = freqs[koro_indices][np.argmax(psd[koro_indices])]
    print(f"Dominant frequency in Korotkoff band: {dom_freq:.2f} Hz")

if __name__ == '__main__':
    analyze()
