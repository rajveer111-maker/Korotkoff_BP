import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\analysis_plot_v5.png'

# Physical Constants
carrier_freq = 0.9e9
c = 3e8
wavelength_mm = (c / carrier_freq) * 1000 # ~333.33 mm

def plot_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_sig = data[0, :]
    q_sig = data[1, :]
    time = np.arange(len(i_sig)) / fs
    
    # 1. Complex Demodulation and Basic Processing
    complex_sig = i_sig + 1j * q_sig
    
    # Artifact Removal: Median filter to remove impulsive noise
    i_clean = signal.medfilt(i_sig, kernel_size=5)
    q_clean = signal.medfilt(q_sig, kernel_size=5)
    complex_clean = i_clean + 1j * q_clean
    
    magnitude = np.abs(complex_clean)
    phase_rad = np.unwrap(np.angle(complex_clean))
    
    # 2. Conversion to Physical Units (mm)
    # Displacement = (Phase * Lambda) / (4 * PI)
    displacement_mm = (phase_rad * wavelength_mm) / (4 * np.pi)
    
    # 3. High-Pass Filter to remove slow drifts (VLF)
    # 0.5 Hz high-pass for both magnitude and phase
    sos_vlf = signal.butter(4, 0.5, btype='high', fs=fs, output='sos')
    mag_ac = signal.sosfiltfilt(sos_vlf, magnitude)
    disp_ac = signal.sosfiltfilt(sos_vlf, displacement_mm)
    
    # 4. Velocity Calculation (mm/s)
    velocity_mms = np.diff(disp_ac) * fs
    velocity_mms = np.append(velocity_mms, velocity_mms[-1])
    
    # Smoothing Velocity for better Koro detection
    velocity_smooth = signal.savgol_filter(velocity_mms, 51, 3)
    
    # 5. Extract Bands
    # Korotkoff (10-50 Hz) on Velocity
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, velocity_smooth)
    
    # Heart Rate (0.7 - 2.5 Hz) 
    # Use disp_ac for HR as it is cleaner after HPF
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, disp_ac)
    
    # Plotting
    fig = plt.figure(figsize=(16, 22))
    
    # Subplot 1: I and Q side-by-side
    ax1 = plt.subplot(6, 2, 1)
    ax1.plot(time, i_clean, color='blue', linewidth=0.5)
    ax1.set_title('In-Phase (I) - Cleaned')
    ax1.set_ylabel('Amp (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(6, 2, 2)
    ax2.plot(time, q_clean, color='orange', linewidth=0.5)
    ax2.set_title('Quadrature (Q) - Cleaned')
    ax2.set_ylabel('Amp (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 2: Magnitude AC
    ax3 = plt.subplot(6, 2, 3)
    ax3.plot(time, mag_ac, color='green', linewidth=0.5)
    ax3.set_title('Magnitude Variations (DC Removed)')
    ax3.set_ylabel('Amp (a.u.)')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 3: Displacement mm
    ax4 = plt.subplot(6, 2, 4)
    ax4.plot(time, disp_ac, color='red', linewidth=0.5)
    ax4.set_title('Displacement (AC Coupled)')
    ax4.set_ylabel('Displacement (mm)')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 4: Velocity mm/s
    ax5 = plt.subplot(6, 1, 3)
    ax5.plot(time, velocity_smooth, color='black', linewidth=0.5)
    ax5.set_title('Mechanical Velocity (Savitzky-Golay Smoothed)')
    ax5.set_ylabel('Velocity (mm/s)')
    ax5.set_ylim(np.percentile(velocity_smooth, 2), np.percentile(velocity_smooth, 98))
    ax5.grid(True, alpha=0.3)
    
    # Subplot 5: Korotkoff Band
    ax6 = plt.subplot(6, 1, 4)
    ax6.plot(time, koro_sig, color='purple', linewidth=0.8)
    ax6.set_title('Korotkoff Signal (10-50 Hz Bandpass)')
    ax6.set_ylabel('Amp (mm/s)')
    ax6.grid(True, alpha=0.3)
    
    # Subplot 6: HR Band
    ax7 = plt.subplot(6, 1, 5)
    ax7.plot(time, hr_sig, color='brown', linewidth=1.0)
    ax7.set_title('Heart Rate (0.7-2.5 Hz Bandpass)')
    ax7.set_ylabel('Amp (mm)')
    ax7.grid(True, alpha=0.3)
    
    # Subplot 7: Spectrogram
    ax8 = plt.subplot(6, 1, 6)
    f, t, Sxx = signal.spectrogram(velocity_smooth, fs, nperseg=int(fs/2), noverlap=int(fs/4))
    ax8.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
    ax8.set_ylim(0, 60)
    ax8.set_title('Spectrogram (0-60 Hz)')
    ax8.set_ylabel('Freq (Hz)')
    ax8.set_xlabel('Time (s)')
    
    plt.suptitle(f'Advanced Physical Analysis: rec_koro_may11.h5\nArtifact Removed | Units: mm, mm/s', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Physical plot saved to: {output_img}")

if __name__ == '__main__':
    plot_analysis()
