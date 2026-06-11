import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 

def check_carrier():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    complex_sig = data[0, :] + 1j * data[1, :]
    
    # Calculate PSD
    freqs, psd = signal.welch(complex_sig, fs, nperseg=fs, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    peak_freq = freqs[np.argmax(psd)]
    print(f"Peak frequency: {peak_freq} Hz")
    
    if abs(peak_freq) > 100:
        print(f"WARNING: Carrier found at {peak_freq} Hz. Data is NOT fully demodulated.")
    else:
        print("Data appears to be at Baseband (demodulated).")

if __name__ == '__main__':
    check_carrier()
