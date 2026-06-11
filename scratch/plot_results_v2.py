import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v2.png'

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
    
    # 2. Heart Rate (0.7 - 2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase)
    
    # Plotting - 6 subplots
    fig = plt.figure(figsize=(14, 18))
    
    # Grid layout: 
    # Row 0: I and Q (parallel)
    # Row 1: Mag and Phase
    # Row 2: Koro
    # Row 3: HR
    
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(time, i_sig, color='blue')
    ax1.set_title('In-phase (I) Channel')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 2, 2, sharey=ax1)
    ax2.plot(time, q_sig, color='orange')
    ax2.set_title('Quadrature (Q) Channel')
    ax2.set_ylabel('Amplitude (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(time, magnitude, color='green')
    ax3.set_title('Magnitude ($\sqrt{I^2 + Q^2}$)')
    ax3.set_ylabel('Amplitude (a.u.)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(time, phase, color='red')
    ax4.set_title('Unwrapped Phase')
    ax4.set_ylabel('Phase (radians)')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(4, 1, 3)
    ax5.plot(time, koro_sig, color='purple')
    ax5.set_title('Korotkoff Signal (10-50 Hz Bandpass)')
    ax5.set_ylabel('Filtered Amp (a.u.)')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(4, 1, 4)
    ax6.plot(time, hr_sig, color='brown')
    ax6.set_title('Heart Rate (0.7-2.5 Hz Bandpass)')
    ax6.set_ylabel('Filtered Amp (a.u.)')
    ax6.set_xlabel('Time (seconds)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Bioview RF Analysis: rec_koro_may11.h5\nSampling Rate: {fs} Hz', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
