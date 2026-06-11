import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

# TARGET FILE
file_name = 'rec_koro11_1.h5'
data_dir = r'd:\Bioview\My_RF_work_v1\data_new'
file_path = os.path.join(data_dir, file_name)
fs = 10000 
output_img = os.path.join(data_dir, 'final_analysis_rec1.png')

def run_final_analysis():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    duration = time[-1]
    
    magnitude = np.abs(i_centered + 1j * q_centered)
    # Using DC Centered vector for phase to avoid large offsets
    phase_var = signal.detrend(np.unwrap(np.angle(i_centered + 1j * q_centered)))
    
    # 1. Heart Rate Verification (Time & Frequency)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    hr_smooth = signal.savgol_filter(hr_sig, 501, 3) # Smoothing for peak detection
    
    # FFT Peak
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    peak_freq = freqs_hr[np.argmax(psd_hr)]
    hr_bpm_fft = peak_freq * 60
    
    # IBI (Inter-Beat Interval)
    peaks, _ = signal.find_peaks(hr_smooth, distance=int(fs*0.5), prominence=np.std(hr_smooth)*0.2)
    ibi = np.diff(time[peaks])
    hr_bpm_ibi = 60 / np.mean(ibi) if len(ibi) > 0 else 0
    
    # 2. Korotkoff Signal (Velocity)
    velocity = np.diff(phase_var) * fs
    velocity = np.append(velocity, velocity[-1])
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, velocity)
    
    # Energy Envelope for duration
    analytic_signal = signal.hilbert(koro_filtered)
    koro_envelope = np.abs(analytic_signal)
    threshold = np.mean(koro_envelope) + 2.5 * np.std(koro_envelope)
    active_mask = koro_envelope > threshold
    koro_duration = (time[np.where(active_mask)[0][-1]] - time[np.where(active_mask)[0][0]]) if np.any(active_mask) else 0

    # PLOTTING
    fig = plt.figure(figsize=(16, 28))
    
    # Panel 1: I & Q (Centered)
    ax1 = plt.subplot(7, 2, 1)
    ax1.plot(time, i_centered, color='blue', linewidth=0.5)
    ax1.set_title('In-Phase (I) - DC Removed')
    ax1.set_ylabel('Amp (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(7, 2, 2)
    ax2.plot(time, q_centered, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q) - DC Removed')
    ax2.set_ylabel('Amp (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 2: Magnitude & Phase
    ax3 = plt.subplot(7, 2, 3)
    ax3.plot(time, magnitude - np.mean(magnitude), color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations')
    ax3.set_ylabel('ΔAmp (a.u.)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(7, 2, 4)
    ax4.plot(time, phase_var, color='red', linewidth=0.5)
    ax4.set_title('Phase Variations (Radians)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_ylim(np.percentile(phase_var, 1), np.percentile(phase_var, 99))
    ax4.grid(True, alpha=0.3)
    
    # Panel 3: Heart Rate Verification (Time)
    ax5 = plt.subplot(7, 2, 5)
    ax5.plot(time, hr_smooth, color='brown', linewidth=1.2)
    ax5.plot(time[peaks], hr_smooth[peaks], "ro", markersize=5)
    ax5.set_title(f'HR Verification: {len(peaks)} Beats Detected')
    ax5.set_ylabel('Amp (rad)')
    ax5.grid(True, alpha=0.3)
    
    # Panel 4: Heart Rate Verification (FFT)
    ax6 = plt.subplot(7, 2, 6)
    ax6.semilogy(freqs_hr, psd_hr, color='brown')
    ax6.axvline(peak_freq, color='k', linestyle='--')
    ax6.set_xlim(0.5, 3.0)
    ax6.set_title(f'HR Spectrum: Peak at {peak_freq:.2f}Hz ({hr_bpm_fft:.1f} BPM)')
    ax6.set_xlabel('Freq (Hz)')
    ax6.grid(True, alpha=0.3)
    
    # Panel 5: Korotkoff Velocity
    ax7 = plt.subplot(7, 1, 4)
    ax7.plot(time, koro_filtered, color='purple', linewidth=0.8, alpha=0.6)
    ax7.plot(time, koro_envelope, color='orange', linewidth=1.5, label='Envelope')
    ax7.set_title(f'Korotkoff Signal (10-50 Hz) | Detected Duration: {koro_duration:.2f} s')
    ax7.set_ylabel('Velocity (rad/s)')
    ax7.set_ylim(np.percentile(koro_filtered, 0.5), np.percentile(koro_filtered, 99.5))
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    # Panel 6: Spectrogram
    ax8 = plt.subplot(7, 1, 5)
    f, t, Sxx = signal.spectrogram(koro_filtered, fs, nperseg=int(fs/4))
    ax8.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    ax8.set_ylim(0, 60)
    ax8.set_title('Korotkoff Spectrogram')
    ax8.set_ylabel('Freq (Hz)')
    
    # Panel 7: Segmented HR View (First 20s)
    ax9 = plt.subplot(7, 1, 6)
    mask20 = time < 20
    ax9.plot(time[mask20], hr_smooth[mask20], color='brown', linewidth=1.5)
    ax9.plot(time[peaks[time[peaks] < 20]], hr_smooth[peaks[time[peaks] < 20]], "ro")
    ax9.set_title('Zoomed View: First 20 Seconds (HR Pulse Check)')
    ax9.set_xlabel('Time (s)')
    ax9.grid(True, alpha=0.3)
    
    # Panel 8: Final Results Summary
    ax10 = plt.subplot(7, 1, 7)
    ax10.axis('off')
    results_text = (f"FINAL ANALYSIS FOR {file_name}\n"
                    f"----------------------------------------\n"
                    f"Estimated Heart Rate (FFT): {hr_bpm_fft:.2f} BPM\n"
                    f"Estimated Heart Rate (IBI): {hr_bpm_ibi:.2f} BPM\n"
                    f"Total Duration: {duration:.2f} s\n"
                    f"Korotkoff Active Duration: {koro_duration:.2f} s\n"
                    f"Total Pulse Peaks Detected: {len(peaks)}")
    ax10.text(0.1, 0.5, results_text, fontsize=14, family='monospace')
    
    plt.suptitle(f'Comprehensive Report: {file_name}', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Final analysis plot saved to: {output_img}")

if __name__ == '__main__':
    run_final_analysis()
