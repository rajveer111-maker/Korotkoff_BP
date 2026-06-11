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
output_img = os.path.join(data_dir, 'ultimate_analysis_report.png')

# Physical Constants
carrier_freq = 0.9e9
c = 3e8
wavelength_mm = (c / carrier_freq) * 1000 # ~333.33 mm

def run_ultimate_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_raw = data[0,:]
    q_raw = data[1,:]
    time = np.arange(len(i_raw)) / fs
    
    # 1. Complex Centering
    i_centered = i_raw - np.mean(i_raw)
    q_centered = q_raw - np.mean(q_raw)
    complex_sig = i_centered + 1j * q_centered
    
    # 2. Basic Conversions (Physical Units)
    magnitude_au = np.abs(complex_sig)
    magnitude_mm = (magnitude_au * wavelength_mm) / (4 * np.pi) # Relative magnitude in mm
    phase_rad = np.unwrap(np.angle(complex_sig))
    displacement_mm = (phase_rad * wavelength_mm) / (4 * np.pi)
    displacement_mm = signal.detrend(displacement_mm)
    
    # 3. Velocity (mm/s)
    velocity_mms = np.diff(displacement_mm) * fs
    velocity_mms = np.append(velocity_mms, velocity_mms[-1])
    
    # 4. Filters (Time Domain)
    # HR (0.7-2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, displacement_mm)
    
    # Koro (10-50 Hz)
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, velocity_mms)
    
    # 5. Frequency Domain (PSD)
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    freqs_koro, psd_koro = signal.welch(koro_sig, fs, nperseg=int(fs*2))
    
    # 6. TFD (Spectrogram)
    f_spec, t_spec, Sxx = signal.spectrogram(velocity_mms, fs, nperseg=int(fs/4), noverlap=int(fs/8))

    # PLOTTING
    fig = plt.figure(figsize=(18, 30))
    
    # --- ANALYSIS 1: TIME DOMAIN (RAW & BASE) ---
    # Row 1: Raw I & Q
    ax1 = plt.subplot(7, 2, 1)
    ax1.plot(time, i_raw, color='blue', linewidth=0.5)
    ax1.set_title('Raw In-Phase (I) Signal')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(7, 2, 2)
    ax2.plot(time, q_raw, color='orange', linewidth=0.5)
    ax2.set_title('Raw Quadrature (Q) Signal')
    ax2.set_ylabel('Amplitude (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Magnitude & Phase (Displacement)
    ax3 = plt.subplot(7, 2, 3)
    ax3.plot(time, magnitude_mm - np.mean(magnitude_mm), color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations (Physical)')
    ax3.set_ylabel('ΔAmplitude (mm)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(7, 2, 4)
    ax4.plot(time, displacement_mm, color='red', linewidth=0.5)
    ax4.set_title('Phase Displacement (Physical)')
    ax4.set_ylabel('Displacement (mm)')
    ax4.set_ylim(np.percentile(displacement_mm, 1), np.percentile(displacement_mm, 99))
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Korotkoff & Heart Rate (Time Domain Waveforms)
    ax5 = plt.subplot(7, 1, 3)
    ax5.plot(time, koro_sig, color='purple', linewidth=0.8)
    ax5.set_title('Korotkoff Signal (Time Domain: 10-50 Hz)')
    ax5.set_ylabel('Velocity (mm/s)')
    ax5.set_ylim(np.percentile(koro_sig, 0.5), np.percentile(koro_sig, 99.5))
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(7, 1, 4)
    ax6.plot(time, hr_sig, color='brown', linewidth=1.2)
    ax6.set_title('Heart Rate Pulse (Time Domain: 0.7-2.5 Hz)')
    ax6.set_ylabel('Displacement (mm)')
    ax6.grid(True, alpha=0.3)
    
    # --- ANALYSIS 2: FREQUENCY DOMAIN (FFT) ---
    ax7 = plt.subplot(7, 2, 9)
    ax7.semilogy(freqs_hr, psd_hr, color='brown')
    ax7.set_xlim(0, 5)
    ax7.set_title('Heart Rate Power Spectrum (FFT)')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('PSD (mm^2/Hz)')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(7, 2, 10)
    ax8.semilogy(freqs_koro, psd_koro, color='purple')
    ax8.set_xlim(0, 100)
    ax8.set_title('Korotkoff Power Spectrum (FFT)')
    ax8.set_xlabel('Frequency (Hz)')
    ax8.set_ylabel('PSD (mm^2/s^2/Hz)')
    ax8.grid(True, alpha=0.3)
    
    # --- ANALYSIS 3: TIME-FREQUENCY DOMAIN (TFD) ---
    ax9 = plt.subplot(7, 1, 6)
    im = ax9.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    ax9.set_ylim(0, 60)
    ax9.set_title('Time-Frequency Analysis (Spectrogram)')
    ax9.set_ylabel('Frequency (Hz)')
    ax9.set_xlabel('Time (s)')
    fig.colorbar(im, ax=ax9, label='Power (dB)')
    
    # Summary Info
    ax10 = plt.subplot(7, 1, 7)
    ax10.axis('off')
    hr_bpm = freqs_hr[np.argmax(psd_hr)] * 60
    summary = (f"ULTIMATE RF ANALYSIS REPORT: {file_name}\n"
               f"--------------------------------------------------\n"
               f"Carrier Frequency: {carrier_freq/1e9:.1f} GHz | Sample Rate: {fs} Hz\n"
               f"Heart Rate Confirmation: {hr_bpm:.2f} BPM (Dominant FFT Peak)\n"
               f"Analysis Types: Time-Domain (Wave), Frequency-Domain (FFT), Time-Frequency (TFD)\n"
               f"Units: mm (Displacement), mm/s (Velocity), Hz (Frequency)")
    ax10.text(0.05, 0.5, summary, fontsize=15, family='monospace')
    
    plt.suptitle(f'Comprehensive RF Physiological Analysis\nFile: {file_name}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Ultimate analysis plot saved to: {output_img}")

if __name__ == '__main__':
    run_ultimate_analysis()
