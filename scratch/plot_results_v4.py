import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v4.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # 1. Magnitude and Phase
    complex_sig = i_sig + 1j * q_sig
    magnitude = np.abs(complex_sig)
    phase = np.unwrap(np.angle(complex_sig))
    
    # 2. Velocity (Derivative of Phase) - better for fast events like Koro
    velocity = np.diff(phase) * fs
    velocity = np.append(velocity, velocity[-1]) # Pad to match length
    
    # 3. Filters
    # Korotkoff (10-50 Hz)
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, velocity) # Use velocity for Koro
    
    # HR (0.7 - 2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase)
    
    # Plotting
    fig = plt.figure(figsize=(15, 20))
    
    # Top Row: I and Q side-by-side
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(time, i_sig, color='blue', linewidth=0.5)
    ax1.set_title('In-Phase (I) Channel')
    ax1.set_ylabel('Amp (a.u.)')
    ax1.set_ylim(np.percentile(i_sig, 1), np.percentile(i_sig, 99) * 1.1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(5, 2, 2)
    ax2.plot(time, q_sig, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q) Channel')
    ax2.set_ylabel('Amp (a.u.)')
    ax2.set_ylim(np.percentile(q_sig, 1), np.percentile(q_sig, 99) * 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Second Row: Magnitude and Phase (Detrended)
    ax3 = plt.subplot(5, 2, 3)
    mag_detrend = magnitude - np.mean(magnitude)
    ax3.plot(time, mag_detrend, color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations')
    ax3.set_ylabel('Amp (a.u.)')
    ax3.set_ylim(np.percentile(mag_detrend, 1), np.percentile(mag_detrend, 99))
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(5, 2, 4)
    phase_detrend = phase - np.mean(phase)
    ax4.plot(time, phase_detrend, color='red', linewidth=0.5)
    ax4.set_title('Phase (Displacement)')
    ax4.set_ylabel('Rad')
    ax4.set_ylim(np.percentile(phase_detrend, 1), np.percentile(phase_detrend, 99))
    ax4.grid(True, alpha=0.3)
    
    # Third Row: Velocity and HR
    ax5 = plt.subplot(5, 2, 5)
    ax5.plot(time, velocity, color='black', linewidth=0.5, alpha=0.5)
    ax5.set_title('Phase Velocity (dφ/dt)')
    ax5.set_ylabel('Rad/s')
    ax5.set_ylim(np.percentile(velocity, 2), np.percentile(velocity, 98))
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(5, 2, 6)
    ax6.plot(time, hr_sig, color='brown', linewidth=0.8)
    ax6.set_title('Heart Rate (0.7-2.5 Hz)')
    ax6.set_ylabel('Rad')
    ax6.set_ylim(np.percentile(hr_sig, 1), np.percentile(hr_sig, 99))
    ax6.grid(True, alpha=0.3)
    
    # Fourth Row: Korotkoff Band
    ax7 = plt.subplot(5, 1, 4)
    ax7.plot(time, koro_sig, color='purple', linewidth=0.8)
    ax7.set_title('Korotkoff Signal (10-50 Hz Bandpass on Velocity)')
    ax7.set_ylabel('Filtered Amp')
    ax7.set_ylim(np.percentile(koro_sig, 0.5), np.percentile(koro_sig, 99.5))
    ax7.grid(True, alpha=0.3)
    
    # Fifth Row: Spectrogram of Korotkoff Band
    ax8 = plt.subplot(5, 1, 5)
    f, t, Sxx = signal.spectrogram(velocity, fs, nperseg=int(fs/2), noverlap=int(fs/4))
    ax8.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
    ax8.set_ylim(0, 100)
    ax8.set_title('Spectrogram (0-100 Hz)')
    ax8.set_ylabel('Freq (Hz)')
    ax8.set_xlabel('Time (s)')
    
    plt.suptitle(f'Advanced Korotkoff Analysis: rec_koro_may11.h5', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Advanced plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
