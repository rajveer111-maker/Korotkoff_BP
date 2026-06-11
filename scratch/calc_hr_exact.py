import h5py
import numpy as np
from scipy import signal

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 

def calculate_hr():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    phase = np.unwrap(np.angle(i_centered + 1j * q_centered))
    
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase)
    
    freqs, psd = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60
    
    print(f"Peak Frequency: {peak_freq:.4f} Hz")
    print(f"Calculated Heart Rate: {hr_bpm:.2f} BPM")

if __name__ == '__main__':
    calculate_hr()
