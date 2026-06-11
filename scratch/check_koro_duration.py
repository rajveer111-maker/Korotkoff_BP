import h5py
import numpy as np
from scipy import signal
import os

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 

def find_duration():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    phase = np.unwrap(np.angle(i_centered + 1j * q_centered))
    velocity = np.diff(phase) * fs
    velocity = np.append(velocity, velocity[-1])
    
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, velocity)
    
    envelope = np.abs(signal.hilbert(koro_filtered))
    threshold = np.mean(envelope) + 2.5 * np.std(envelope)
    
    active_indices = np.where(envelope > threshold)[0]
    if len(active_indices) > 0:
        start_time = active_indices[0] / fs
        end_time = active_indices[-1] / fs
        duration = end_time - start_time
        print(f"Start: {start_time:.2f} s")
        print(f"End: {end_time:.2f} s")
        print(f"Total Duration: {duration:.2f} s")
    else:
        print("No Korotkoff sounds detected above threshold.")

if __name__ == '__main__':
    find_duration()
