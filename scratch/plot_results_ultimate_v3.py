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
output_img = os.path.join(data_dir, 'ultimate_analysis_v3.png')

# Physical Constants (Verified Units)
carrier_freq = 0.9e9 # 0.9 GHz
c = 299792458 # Exact speed of light (m/s)
wavelength_mm = (c / carrier_freq) * 1000 # ~333.10 mm

def run_ultimate_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    
    # 1. Raw Math
    magnitude_au = np.abs(i_centered + 1j * q_centered)
    phase_rad = np.unwrap(np.angle(i_centered + 1j * q_centered))
    
    # 2. Total Displacement (Cuff + Pulse)
    # The 4*pi accounts for the round-trip (2x distance)
    displacement_mm = (phase_rad * wavelength_mm) / (4 * np.pi)
    
    # 3. Separate Cuff Motion (Slow) from Physiological Pulses (Fast)
    # Cuff movement is usually < 0.2 Hz
    sos_cuff = signal.butter(4, 0.2, btype='lowpass', fs=fs, output='sos')
    cuff_motion_mm = signal.sosfiltfilt(sos_cuff, displacement_mm)
    
    # Heart Rate band (0.7 - 2.5 Hz)
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_pulse_mm = signal.sosfiltfilt(sos_hr, displacement_mm)
    
    # Korotkoff band (10 - 50 Hz) Velocity
    velocity_mms = np.diff(displacement_mm) * fs
    velocity_mms = np.append(velocity_mms, velocity_mms[-1])
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered_mms = signal.sosfiltfilt(sos_koro, velocity_mms)

    # PLOTTING
    fig = plt.figure(figsize=(18, 32))
    
    # Row 1: Raw I & Q
    plt.subplot(8, 2, 1); plt.plot(time, i_centered, color='blue'); plt.title('Raw I'); plt.ylabel('a.u.')
    plt.subplot(8, 2, 2); plt.plot(time, q_centered, color='orange'); plt.title('Raw Q'); plt.ylabel('a.u.')
    
    # Row 2: Total Raw Displacement (The "Cuff" View)
    plt.subplot(8, 1, 2)
    plt.plot(time, displacement_mm - displacement_mm[0], color='black', label='Total Displacement')
    plt.plot(time, cuff_motion_mm - cuff_motion_mm[0], color='red', linewidth=2, label='Slow Cuff Motion')
    plt.title('Total Physical Displacement (Cuff Deflation View)')
    plt.ylabel('Movement (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Row 3: Heart Rate Pulses (The "Physiology" View)
    plt.subplot(8, 1, 3)
    plt.plot(time, hr_pulse_mm, color='brown')
    plt.title('Heart Rate Pulse (Separated from Cuff Motion)')
    plt.ylabel('Pulse Amp (mm)')
    plt.grid(True, alpha=0.3)
    
    # Row 4: Korotkoff Velocity
    plt.subplot(8, 1, 4)
    plt.plot(time, koro_filtered_mms, color='purple')
    plt.title('Korotkoff Signal (High-Freq Arterial Snaps)')
    plt.ylabel('Velocity (mm/s)')
    plt.grid(True, alpha=0.3)
    
    # Row 5: Spectrogram
    plt.subplot(8, 1, 5)
    f, t, Sxx = signal.spectrogram(koro_filtered_mms, fs, nperseg=int(fs/4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    plt.ylim(0, 60); plt.title('Spectrogram (TFD Analysis)'); plt.ylabel('Hz')
    
    # Summary
    plt.subplot(8, 1, 6)
    plt.axis('off')
    hr_bpm = signal.welch(hr_pulse_mm, fs, nperseg=int(fs*10))[1].argmax() * (fs/int(fs*10)) * 60
    summary = (f"ULTIMATE RF ANALYSIS V3 (CUFF VS PULSE)\n"
               f"--------------------------------------------------\n"
               f"Total Cuff Displacement: {np.max(cuff_motion_mm) - np.min(cuff_motion_mm):.2f} mm\n"
               f"Individual Heartbeat Amplitude: {np.max(hr_pulse_mm) - np.min(hr_pulse_mm):.4f} mm\n"
               f"Confirming HR: {hr_bpm:.2f} BPM")
    plt.text(0.05, 0.5, summary, fontsize=15, family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Cuff vs Pulse analysis saved to: {output_img}")

if __name__ == '__main__':
    run_ultimate_analysis()
