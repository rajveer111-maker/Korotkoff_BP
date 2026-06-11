import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # Calculate Magnitude and Phase
    magnitude = np.sqrt(i_sig**2 + q_sig**2)
    phase = np.unwrap(np.angle(i_sig + 1j * q_sig))
    
    # Filters
    # 1. Korotkoff (10-50 Hz)
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, magnitude)
    
    # 2. Heart Rate (0.7 - 2.5 Hz) -> 42-150 BPM
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase) # Phase often shows pulse better
    
    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # 1. Raw I and Q
    axes[0].plot(time, i_sig, label='I', alpha=0.7, color='blue')
    axes[0].plot(time, q_sig, label='Q', alpha=0.7, color='orange')
    axes[0].set_title('Raw I and Q Signals')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Magnitude
    axes[1].plot(time, magnitude, color='green')
    axes[1].set_title('Magnitude (sqrt(I^2 + Q^2))')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Phase
    axes[2].plot(time, phase, color='red')
    axes[2].set_title('Unwrapped Phase (Displacement)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Korotkoff Band (10-50 Hz)
    axes[3].plot(time, koro_sig, color='purple')
    axes[3].set_title('Korotkoff Band (10-50 Hz) - Mechanical Snaps')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Heart Rate Band (0.7-2.5 Hz)
    axes[4].plot(time, hr_sig, color='brown')
    axes[4].set_title('Heart Rate Band (0.7-2.5 Hz) - Arterial Pulse')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
