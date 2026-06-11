import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v9.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    duration = time[-1]
    
    magnitude = np.abs(i_centered + 1j * q_centered)
    phase_var = signal.detrend(np.unwrap(np.angle(i_centered + 1j * q_centered)))
    
    # 1. Heart Rate Processing
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    
    # FFT for HR - The "Frequency Method" (48 BPM)
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    peak_hr_idx = np.argmax(psd_hr)
    peak_freq = freqs_hr[peak_hr_idx]
    hr_bpm_fft = peak_freq * 60
    
    # Time-domain Peak Detection - The "Counting Method"
    # Adjusted threshold for more sensitivity
    min_dist = int(fs * 0.4) 
    peaks, _ = signal.find_peaks(hr_sig, distance=min_dist, height=np.std(hr_sig)*0.3)
    total_beats = len(peaks)
    hr_bpm_count = (total_beats / duration) * 60 if duration > 0 else 0
    
    # 2. Korotkoff Processing
    velocity = np.diff(phase_var) * fs
    velocity = np.append(velocity, velocity[-1])
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, velocity)
    
    # Plotting
    fig = plt.figure(figsize=(16, 26))
    
    # Row 1: I and Q
    ax1 = plt.subplot(6, 2, 1)
    ax1.plot(time, i_centered, color='blue', linewidth=0.5)
    ax1.set_title('In-Phase (I)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_xlabel('Time (s)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(6, 2, 2)
    ax2.plot(time, q_centered, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q)')
    ax2.set_ylabel('Amplitude (a.u.)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Magnitude and Phase
    ax3 = plt.subplot(6, 2, 3)
    ax3.plot(time, magnitude - np.mean(magnitude), color='green', linewidth=0.5)
    ax3.set_title('Magnitude (AC)')
    ax3.set_ylabel('Amplitude (a.u.)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(6, 2, 4)
    ax4.plot(time, phase_var, color='red', linewidth=0.5)
    ax4.set_title('Phase (Detrended)')
    ax4.set_ylabel('Angle (radians)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylim(np.percentile(phase_var, 1), np.percentile(phase_var, 99))
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Heart Rate Confirmation
    ax5 = plt.subplot(6, 2, 5)
    ax5.plot(time, hr_sig, color='brown', linewidth=1.0)
    ax5.plot(time[peaks], hr_sig[peaks], "ro", markersize=4, label='Detected Pulse')
    ax5.set_title(f'HR Time-Domain: {total_beats} Pulses Counted')
    ax5.set_ylabel('Phase Variation (rad)')
    ax5.set_xlabel('Time (s)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(6, 2, 6)
    ax6.semilogy(freqs_hr, psd_hr, color='brown')
    ax6.axvline(peak_freq, color='red', linestyle='--')
    ax6.set_xlim(0.5, 3.0)
    ax6.set_title(f'HR FFT Spectrum: {hr_bpm_fft:.1f} BPM Peak')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power Density (dB/Hz)')
    ax6.grid(True, alpha=0.3)
    
    # Row 4: Korotkoff Velocity
    ax7 = plt.subplot(6, 1, 4)
    ax7.plot(time, koro_filtered, color='purple', linewidth=0.8)
    ax7.set_title('Korotkoff Signal (10-50 Hz Velocity)')
    ax7.set_ylabel('Velocity (rad/s)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylim(np.percentile(koro_filtered, 0.5), np.percentile(koro_filtered, 99.5))
    ax7.grid(True, alpha=0.3)
    
    # Row 5: Spectrogram
    ax8 = plt.subplot(6, 1, 5)
    f, t, Sxx = signal.spectrogram(koro_filtered, fs, nperseg=int(fs/4))
    im = ax8.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    ax8.set_ylim(0, 60)
    ax8.set_title('Korotkoff Spectrogram')
    ax8.set_ylabel('Frequency (Hz)')
    ax8.set_xlabel('Time (s)')
    fig.colorbar(im, ax=ax8, label='Power (dB)')
    
    # Row 6: Full PSD
    ax9 = plt.subplot(6, 1, 6)
    freqs_full, psd_full = signal.welch(phase_var, fs, nperseg=int(fs*2))
    ax9.semilogy(freqs_full, psd_full, color='black')
    ax9.set_xlim(0, 100)
    ax9.set_title('Global Power Spectrum (0-100 Hz)')
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('PSD (dB/Hz)')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fully Labeled Analysis: rec_koro_may11.h5\nFFT-Rate: {hr_bpm_fft:.1f} BPM | Total Count: {total_beats} Pulses', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Labeled plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
