import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v6.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # Complex Conversion (NO AGGRESSIVE CLEANING)
    complex_sig = i_sig + 1j * q_sig
    magnitude = np.abs(complex_sig)
    phase = np.unwrap(np.angle(complex_sig))
    
    # 1. Heart Rate Analysis (0.7 - 2.5 Hz)
    # Filter phase for HR
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase)
    
    # FFT for HR confirmation
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10)) # High resolution
    peak_hr_idx = np.argmax(psd_hr)
    hr_bpm = freqs_hr[peak_hr_idx] * 60
    
    # 2. Korotkoff Analysis (10-50 Hz)
    # Use magnitude derivative for Koro to preserve sharp events
    koro_raw = np.diff(magnitude) * fs
    koro_raw = np.append(koro_raw, koro_raw[-1])
    
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, koro_raw)
    
    # Energy Envelope for duration
    analytic_signal = signal.hilbert(koro_filtered)
    koro_envelope = np.abs(analytic_signal)
    
    # FFT for Koro confirmation
    freqs_koro, psd_koro = signal.welch(koro_filtered, fs, nperseg=int(fs*2))
    
    # 3. Detect Koro Duration
    # Thresholding envelope to find start/end
    threshold = np.mean(koro_envelope) + 2 * np.std(koro_envelope)
    active_mask = koro_envelope > threshold
    if np.any(active_mask):
        indices = np.where(active_mask)[0]
        start_time = time[indices[0]]
        end_time = time[indices[-1]]
        koro_duration = end_time - start_time
    else:
        start_time, end_time, koro_duration = 0, 0, 0

    # Plotting
    fig = plt.figure(figsize=(16, 24))
    
    # Panel 1: Magnitude and Phase Variations
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(time, magnitude - np.mean(magnitude), color='green', linewidth=0.5)
    ax1.set_title('Magnitude Variations (Raw)')
    ax1.set_ylabel('Amp (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(5, 2, 2)
    ax2.plot(time, phase - np.mean(phase), color='red', linewidth=0.5)
    ax2.set_title('Phase Variations (Raw)')
    ax2.set_ylabel('Rad')
    ax2.grid(True, alpha=0.3)
    
    # Panel 2: Heart Rate Time & Frequency
    ax3 = plt.subplot(5, 2, 3)
    ax3.plot(time, hr_sig, color='brown', linewidth=0.8)
    ax3.set_title(f'Heart Rate Band (0.7-2.5 Hz)\nEstimated HR: {hr_bpm:.1f} BPM')
    ax3.set_ylabel('Amp (rad)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(5, 2, 4)
    ax4.semilogy(freqs_hr, psd_hr, color='brown')
    ax4.axvline(freqs_hr[peak_hr_idx], color='k', linestyle='--', alpha=0.5)
    ax4.set_xlim(0, 5)
    ax4.set_title('HR Power Spectrum (FFT)')
    ax4.set_xlabel('Freq (Hz)')
    ax4.set_ylabel('PSD')
    ax4.grid(True, alpha=0.3)
    
    # Panel 3: Korotkoff Time & Frequency
    ax5 = plt.subplot(5, 2, 5)
    ax5.plot(time, koro_filtered, color='purple', linewidth=0.8, alpha=0.6)
    ax5.plot(time, koro_envelope, color='orange', linewidth=1.5, label='Envelope')
    ax5.set_title(f'Korotkoff Signal (10-50 Hz)\nDetected Duration: {koro_duration:.2f} s')
    ax5.set_ylabel('Amp')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(5, 2, 6)
    ax6.semilogy(freqs_koro, psd_koro, color='purple')
    ax6.set_xlim(0, 100)
    ax6.set_title('Korotkoff Power Spectrum (FFT)')
    ax6.set_xlabel('Freq (Hz)')
    ax6.grid(True, alpha=0.3)
    
    # Panel 4: Spectrogram
    ax7 = plt.subplot(5, 1, 4)
    f, t, Sxx = signal.spectrogram(koro_filtered, fs, nperseg=int(fs/4), noverlap=int(fs/8))
    ax7.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
    ax7.set_ylim(0, 60)
    ax7.set_title('Korotkoff Spectrogram (Activity Tracking)')
    ax7.set_ylabel('Freq (Hz)')
    
    # Panel 5: Detailed Zoom of Koro Pulses (Center region)
    ax8 = plt.subplot(5, 1, 5)
    zoom_start = max(0, start_time - 1)
    zoom_end = min(time[-1], end_time + 1)
    mask = (time >= zoom_start) & (time <= zoom_end)
    ax8.plot(time[mask], koro_filtered[mask], color='purple')
    ax8.set_title('Zoomed View of Detected Korotkoff Active Region')
    ax8.set_xlabel('Time (s)')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'Frequency Domain Analysis: rec_koro_may11.h5\nHR: {hr_bpm:.1f} BPM | Koro Duration: {koro_duration:.2f} s', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Frequency analysis plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
