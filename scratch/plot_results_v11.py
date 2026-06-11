import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v11.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    
    magnitude = np.abs(i_centered + 1j * q_centered)
    phase_var = signal.detrend(np.unwrap(np.angle(i_centered + 1j * q_centered)))
    
    # 1. Time-Domain Heart Rate Calculation
    # Bandpass filter
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    
    # Smooth for reliable peak detection
    hr_smooth = signal.savgol_filter(hr_sig, 1001, 3)
    
    # Find Peaks (Beats)
    min_dist = int(fs * 0.6) # ~100 BPM max
    peaks, _ = signal.find_peaks(hr_smooth, distance=min_dist, prominence=np.std(hr_smooth)*0.3)
    
    # Calculate Inter-Beat Intervals (IBI)
    peak_times = time[peaks]
    ibi = np.diff(peak_times) # Time between beats in seconds
    inst_hr = 60 / ibi      # Instantaneous HR in BPM
    avg_hr = np.mean(inst_hr)
    
    # Plotting
    fig = plt.figure(figsize=(15, 12))
    
    # Top Plot: Time Domain Beats
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time, hr_smooth, color='brown', linewidth=1.5, label='Filtered Pulse')
    ax1.plot(peak_times, hr_smooth[peaks], "ro", markersize=8, label='Detected Beats')
    
    # Add vertical lines and labels for intervals
    for i in range(len(ibi)):
        mid_point = (peak_times[i] + peak_times[i+1]) / 2
        ax1.annotate(f'{ibi[i]:.2f}s\n({inst_hr[i]:.1f} BPM)', 
                     xy=(mid_point, np.max(hr_smooth)*1.1), 
                     ha='center', fontsize=9, color='blue', fontweight='bold')
        ax1.axvline(peak_times[i], color='red', linestyle='--', alpha=0.3)
    
    ax1.set_title(f'Time-Domain Heart Rate Verification\nAverage Calculated HR: {avg_hr:.2f} BPM', fontsize=14)
    ax1.set_ylabel('Phase Variation (rad)')
    ax1.set_xlabel('Time (s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(np.min(hr_smooth)*1.2, np.max(hr_smooth)*1.8) # Room for labels
    
    # Bottom Plot: Distribution of HRs
    ax2 = plt.subplot(2, 1, 2)
    ax2.step(peak_times[:-1], inst_hr, where='post', color='blue', linewidth=2)
    ax2.axhline(avg_hr, color='red', linestyle='--', label=f'Average = {avg_hr:.1f} BPM')
    ax2.set_title('Instantaneous Heart Rate (Beat-to-Beat)')
    ax2.set_ylabel('BPM')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(40, 60)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Time-domain HR verification plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
