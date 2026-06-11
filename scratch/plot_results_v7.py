import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v7.png'

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # 1. Complex Demodulation and Centering
    # Subtracting mean (DC) from I and Q to center the complex vector.
    # This is crucial for correct phase unwrapping of small vibrations.
    i_centered = i_sig - np.mean(i_sig)
    q_centered = q_sig - np.mean(q_sig)
    complex_centered = i_centered + 1j * q_centered
    
    magnitude = np.abs(complex_centered)
    # Use angle from the centered complex vector
    phase_rad = np.unwrap(np.angle(complex_centered))
    
    # Detrend phase to remove any residual linear drift
    phase_var = signal.detrend(phase_rad)
    
    # 2. Heart Rate Analysis (0.7 - 2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    
    # FFT for HR confirmation
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    peak_hr_idx = np.argmax(psd_hr)
    hr_bpm = freqs_hr[peak_hr_idx] * 60
    
    # 3. Korotkoff Analysis (10-50 Hz)
    # Using derivative of centered phase (Velocity) for high-freq events
    velocity = np.diff(phase_var) * fs
    velocity = np.append(velocity, velocity[-1])
    
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, velocity)
    
    # Energy Envelope
    analytic_signal = signal.hilbert(koro_filtered)
    koro_envelope = np.abs(analytic_signal)
    
    # Detect Koro Duration
    threshold = np.mean(koro_envelope) + 2.5 * np.std(koro_envelope)
    active_mask = koro_envelope > threshold
    if np.any(active_mask):
        indices = np.where(active_mask)[0]
        start_time, end_time = time[indices[0]], time[indices[-1]]
        koro_duration = end_time - start_time
    else:
        start_time, end_time, koro_duration = 0, 0, 0

    # Plotting
    fig = plt.figure(figsize=(16, 24))
    
    # Panel 1: Centered I and Q
    ax1 = plt.subplot(6, 2, 1)
    ax1.plot(time, i_centered, color='blue', linewidth=0.5)
    ax1.set_title('In-Phase (I) - DC Removed')
    ax1.set_ylabel('Amp (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(6, 2, 2)
    ax2.plot(time, q_centered, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q) - DC Removed')
    ax2.set_ylabel('Amp (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 2: Magnitude and Phase Variations (SMALL SCALE)
    ax3 = plt.subplot(6, 2, 3)
    ax3.plot(time, magnitude - np.mean(magnitude), color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations')
    ax3.set_ylabel('ΔAmp (a.u.)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(6, 2, 4)
    ax4.plot(time, phase_var, color='red', linewidth=0.5)
    ax4.set_title('Phase Variations (Detrended)')
    ax4.set_ylabel('Phase (radians)')
    # Tight limits to show the actual range (likely +/- few radians)
    ax4.set_ylim(np.percentile(phase_var, 1), np.percentile(phase_var, 99))
    ax4.grid(True, alpha=0.3)
    
    # Panel 3: Heart Rate Time & Frequency
    ax5 = plt.subplot(6, 2, 5)
    ax5.plot(time, hr_sig, color='brown', linewidth=0.8)
    ax5.set_title(f'Heart Rate Band (0.7-2.5 Hz)\nHR: {hr_bpm:.1f} BPM')
    ax5.set_ylabel('Phase (rad)')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(6, 2, 6)
    ax6.semilogy(freqs_hr, psd_hr, color='brown')
    ax6.set_xlim(0, 5)
    ax6.set_title('HR Power Spectrum (FFT)')
    ax6.set_xlabel('Freq (Hz)')
    ax6.grid(True, alpha=0.3)
    
    # Panel 4: Korotkoff Velocity
    ax7 = plt.subplot(6, 1, 4)
    ax7.plot(time, koro_filtered, color='purple', linewidth=0.8, alpha=0.6)
    ax7.plot(time, koro_envelope, color='orange', linewidth=1.2, label='Envelope')
    ax7.set_title(f'Korotkoff Velocity (10-50 Hz) | Duration: {koro_duration:.2f} s')
    ax7.set_ylabel('Velocity (rad/s)')
    ax7.set_ylim(np.percentile(koro_filtered, 0.5), np.percentile(koro_filtered, 99.5))
    ax7.grid(True, alpha=0.3)
    
    # Panel 5: Spectrogram
    ax8 = plt.subplot(6, 1, 5)
    f, t, Sxx = signal.spectrogram(koro_filtered, fs, nperseg=int(fs/4), noverlap=int(fs/8))
    ax8.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    ax8.set_ylim(0, 60)
    ax8.set_title('Korotkoff Spectrogram')
    ax8.set_ylabel('Freq (Hz)')
    
    # Panel 6: FFT Confirmation (Full range)
    ax9 = plt.subplot(6, 1, 6)
    freqs_full, psd_full = signal.welch(phase_var, fs, nperseg=int(fs*2))
    ax9.semilogy(freqs_full, psd_full, color='black', alpha=0.7)
    ax9.set_xlim(0, 100)
    ax9.set_title('Full Signal Power Spectrum (0-100 Hz)')
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('PSD')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced Corrected Analysis: rec_koro_may11.h5\nDC Centered | Detrended Phase', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Corrected analysis plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
