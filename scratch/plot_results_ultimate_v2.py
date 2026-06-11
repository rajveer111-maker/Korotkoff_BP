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
output_img = os.path.join(data_dir, 'ultimate_analysis_v2.png')

# Physical Constants
carrier_freq = 0.9e9
c = 3e8
wavelength_mm = (c / carrier_freq) * 1000 

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
    
    # 2. RAW MATH (a.u. and radians)
    magnitude_au = np.abs(complex_sig)
    phase_rad = np.unwrap(np.angle(complex_sig))
    phase_rad_detrend = signal.detrend(phase_rad)
    
    # 3. PHYSICAL CONVERSION (mm)
    magnitude_mm = (magnitude_au * wavelength_mm) / (4 * np.pi)
    displacement_mm = (phase_rad_detrend * wavelength_mm) / (4 * np.pi)
    
    # 4. Velocity (mm/s)
    velocity_mms = np.diff(displacement_mm) * fs
    velocity_mms = np.append(velocity_mms, velocity_mms[-1])
    
    # 5. Filters
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, displacement_mm)
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, velocity_mms)
    
    # 6. Frequency Analysis
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    freqs_koro, psd_koro = signal.welch(koro_sig, fs, nperseg=int(fs*2))
    f_spec, t_spec, Sxx = signal.spectrogram(velocity_mms, fs, nperseg=int(fs/4))

    # PLOTTING
    fig = plt.figure(figsize=(18, 35))
    
    # --- ROW 1: RAW SENSOR DATA ---
    plt.subplot(8, 2, 1)
    plt.plot(time, i_raw, color='blue', linewidth=0.5)
    plt.title('Raw I Channel (a.u.)'); plt.ylabel('Amp'); plt.grid(True, alpha=0.3)
    
    plt.subplot(8, 2, 2)
    plt.plot(time, q_raw, color='orange', linewidth=0.5)
    plt.title('Raw Q Channel (a.u.)'); plt.ylabel('Amp'); plt.grid(True, alpha=0.3)
    
    # --- ROW 2: RAW MATHEMATICAL EXTRACTION (REQUESTED) ---
    plt.subplot(8, 2, 3)
    plt.plot(time, magnitude_au - np.mean(magnitude_au), color='green', linewidth=0.5)
    plt.title('Magnitude (a.u.)'); plt.ylabel('Amp (a.u.)'); plt.grid(True, alpha=0.3)
    
    plt.subplot(8, 2, 4)
    plt.plot(time, phase_rad_detrend, color='red', linewidth=0.5)
    plt.title('Phase (radians)'); plt.ylabel('Phase (rad)'); plt.grid(True, alpha=0.3)
    
    # --- ROW 3: PHYSICAL DISPLACEMENT ---
    plt.subplot(8, 2, 5)
    plt.plot(time, magnitude_mm - np.mean(magnitude_mm), color='darkgreen', linewidth=0.5)
    plt.title('Magnitude (Physical mm)'); plt.ylabel('ΔAmp (mm)'); plt.grid(True, alpha=0.3)
    
    plt.subplot(8, 2, 6)
    plt.plot(time, displacement_mm, color='darkred', linewidth=0.5)
    plt.title('Displacement (Physical mm)'); plt.ylabel('Disp (mm)'); plt.grid(True, alpha=0.3)
    
    # --- ROW 4: VELOCITY AND PULSE WAVEFORMS ---
    plt.subplot(8, 1, 4)
    plt.plot(time, koro_sig, color='purple', linewidth=0.8)
    plt.title('Korotkoff Signal (10-50 Hz Band)'); plt.ylabel('Velocity (mm/s)'); plt.grid(True, alpha=0.3)
    
    plt.subplot(8, 1, 5)
    plt.plot(time, hr_sig, color='brown', linewidth=1.2)
    plt.title('Heart Rate Pulse (0.7-2.5 Hz Band)'); plt.ylabel('Displacement (mm)'); plt.grid(True, alpha=0.3)
    
    # --- ROW 5: FREQUENCY ANALYSIS ---
    plt.subplot(8, 2, 11)
    plt.semilogy(freqs_hr, psd_hr, color='brown')
    plt.xlim(0, 5); plt.title('HR Spectrum'); plt.xlabel('Hz'); plt.grid(True, alpha=0.3)
    
    plt.subplot(8, 2, 12)
    plt.semilogy(freqs_koro, psd_koro, color='purple')
    plt.xlim(0, 100); plt.title('Koro Spectrum'); plt.xlabel('Hz'); plt.grid(True, alpha=0.3)
    
    # --- ROW 6: TFD ANALYSIS ---
    plt.subplot(8, 1, 7)
    im = plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    plt.ylim(0, 60); plt.title('Spectrogram (TFD Analysis)'); plt.ylabel('Hz'); plt.colorbar(im, label='dB')
    
    # --- ROW 7: SUMMARY ---
    plt.subplot(8, 1, 8)
    plt.axis('off')
    hr_bpm = freqs_hr[np.argmax(psd_hr)] * 60
    summary = (f"ULTIMATE RF ANALYSIS REPORT V2: {file_name}\n"
               f"--------------------------------------------------\n"
               f"Raw Magnitude Unit: a.u. | Raw Phase Unit: radians\n"
               f"Physical Conversion: Displacement (mm) | Velocity (mm/s)\n"
               f"Calculated Heart Rate: {hr_bpm:.2f} BPM")
    plt.text(0.05, 0.5, summary, fontsize=15, family='monospace')
    
    plt.suptitle(f'Comprehensive Analysis with Raw & Physical Units\nFile: {file_name}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Ultimate analysis v2 plot saved to: {output_img}")

if __name__ == '__main__':
    run_ultimate_analysis()
