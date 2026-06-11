import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v10.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    duration = time[-1]
    
    magnitude = np.abs(i_centered + 1j * q_centered)
    phase_var = signal.detrend(np.unwrap(np.angle(i_centered + 1j * q_centered)))
    
    # IMPROVED HR EXTRACTION
    # 1. Bandpass filter
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    
    # 2. Moving Average smoothing to help with peak detection
    window_size = int(fs * 0.1) # 100ms window
    hr_smooth = np.convolve(hr_sig, np.ones(window_size)/window_size, mode='same')
    
    # 3. Peak Detection with much lower threshold and adaptive logic
    # Min distance for 48 BPM (1.25s interval) is approx fs * 0.8
    # We use fs * 0.5 to allow up to 120 BPM
    peaks, _ = signal.find_peaks(hr_smooth, distance=int(fs*0.5), prominence=np.std(hr_smooth)*0.2)
    
    # 4. Segmented Plotting (10 seconds per row) to see individual pulses clearly
    seg_len = 10 # seconds
    num_segs = int(np.ceil(duration / seg_len))
    
    fig, axes = plt.subplots(num_segs, 1, figsize=(15, 4 * num_segs), sharey=True)
    
    for i in range(num_segs):
        t_start = i * seg_len
        t_end = min((i + 1) * seg_len, duration)
        mask = (time >= t_start) & (time <= t_end)
        
        ax = axes[i]
        ax.plot(time[mask], hr_smooth[mask], color='brown', linewidth=1.5, label='Filtered Heartbeat')
        
        # Plot detected peaks in this segment
        seg_peaks = peaks[(time[peaks] >= t_start) & (time[peaks] <= t_end)]
        ax.plot(time[seg_peaks], hr_smooth[seg_peaks], "ro", markersize=6, label=f'Beat')
        
        ax.set_title(f'Segment {i+1}: {t_start}s to {t_end}s | Beats in segment: {len(seg_peaks)}')
        ax.set_ylabel('Amp (rad)')
        ax.grid(True, alpha=0.3)
        if i == num_segs - 1:
            ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')

    plt.suptitle(f'Detailed Heartbeat Visualization: rec_koro_may11.h5\nTotal Pulses Detected: {len(peaks)} | Expected (~48 BPM): ~33-35', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Segmented HR plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
