import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v3.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # Calculate Magnitude and Phase
    magnitude = np.sqrt(i_sig**2 + q_sig**2)
    phase = np.unwrap(np.angle(i_sig + 1j * q_sig))
    
    # Detrend phase and magnitude for better visualization of variations
    magnitude_var = magnitude - np.mean(magnitude)
    phase_var = phase - np.mean(phase)
    
    # Filters
    # 1. Korotkoff (10-50 Hz)
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, magnitude)
    
    # 2. Heart Rate (0.7 - 2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase)
    
    fig = plt.figure(figsize=(14, 18))
    
    def set_tight_ylim(ax, data, margin=0.1):
        if len(data) == 0: return
        d_min, d_max = np.min(data), np.max(data)
        d_range = d_max - d_min
        if d_range == 0: d_range = 1.0
        ax.set_ylim(d_min - margin*d_range, d_max + margin*d_range)

    # Subplot 1: I Channel
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(time, i_sig, color='blue', linewidth=0.5)
    ax1.set_title('In-phase (I) Channel')
    ax1.set_ylabel('Amplitude (a.u.)')
    set_tight_ylim(ax1, i_sig)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Q Channel
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(time, q_sig, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q) Channel')
    ax2.set_ylabel('Amplitude (a.u.)')
    set_tight_ylim(ax2, q_sig)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Magnitude (AC Coupled)
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(time, magnitude_var, color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations (DC Removed)')
    ax3.set_ylabel('Amplitude (a.u.)')
    set_tight_ylim(ax3, magnitude_var)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Phase (AC Coupled)
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(time, phase_var, color='red', linewidth=0.5)
    ax4.set_title('Phase Variations (DC Removed)')
    ax4.set_ylabel('Phase (radians)')
    set_tight_ylim(ax4, phase_var)
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Korotkoff Band
    ax5 = plt.subplot(4, 1, 3)
    ax5.plot(time, koro_sig, color='purple', linewidth=0.7)
    ax5.set_title('Korotkoff Signal (10-50 Hz)')
    ax5.set_ylabel('Filtered Amp (a.u.)')
    # Focus on the pulses, avoid excessive empty space
    set_tight_ylim(ax5, koro_sig)
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Heart Rate Band
    ax6 = plt.subplot(4, 1, 4)
    ax6.plot(time, hr_sig, color='brown', linewidth=0.7)
    ax6.set_title('Heart Rate Pulse (0.7-2.5 Hz)')
    ax6.set_ylabel('Filtered Amp (a.u.)')
    ax6.set_xlabel('Time (seconds)')
    set_tight_ylim(ax6, hr_sig)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Bioview RF Analysis (Zoomed): rec_koro_may11.h5', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Zoomed plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
