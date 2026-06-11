import h5py
import numpy as np
import os
from scipy import signal

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 

def analyze():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_signal = data[0, :]
    q_signal = data[1, :]
    
    print(f"Data contains NaNs: {np.isnan(data).any()}")
    print(f"Data contains Infs: {np.isinf(data).any()}")
    
    magnitude = np.sqrt(i_signal**2 + q_signal**2)
    print(f"Magnitude NaNs: {np.isnan(magnitude).any()}")
    
    # Check filter stability
    b, a = signal.butter(4, [10, 50], btype='bandpass', fs=fs)
    print(f"Filter coefficients b: {b}")
    print(f"Filter coefficients a: {a}")
    
    # Try filtering in chunks or check the result
    try:
        koro_mag = signal.filtfilt(b, a, magnitude)
        print(f"Filtered signal max: {np.nanmax(np.abs(koro_mag))}")
        print(f"Filtered signal min: {np.nanmin(np.abs(koro_mag))}")
        print(f"Filtered signal NaNs: {np.isnan(koro_mag).any()}")
    except Exception as e:
        print(f"Filtering error: {e}")

if __name__ == '__main__':
    analyze()
