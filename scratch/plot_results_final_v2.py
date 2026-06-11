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
output_img = os.path.join(data_dir, 'final_analysis_rec1_v2.png')

def run_final_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_centered = data[0,:] - np.mean(data[0,:])
    q_centered = data[1,:] - np.mean(data[1,:])
    time = np.arange(len(i_centered)) / fs
    duration = time[-1]
    
    phase_var = signal.detrend(np.unwrap(np.angle(i_centered + 1j * q_centered)))
    
    # 1. Korotkoff Processing
    velocity = np.diff(phase_var) * fs
    velocity = np.append(velocity, velocity[-1])
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_filtered = signal.sosfiltfilt(sos_koro, velocity)
    
    # 2. IMPROVED DURATION DETECTION (Energy Window)
    # Calculate energy in 0.5s windows
    win_size = int(fs * 0.5)
    energy = np.convolve(koro_filtered**2, np.ones(win_size)/win_size, mode='same')
    
    # Find the main "burst" area (where energy > mean + std)
    # We look for a continuous region
    energy_threshold = np.mean(energy) + 1.5 * np.std(energy)
    active_indices = np.where(energy > energy_threshold)[0]
    
    if len(active_indices) > 100: # Ensure it's not just a single spike
        start_time = time[active_indices[0]]
        end_time = time[active_indices[-1]]
        koro_duration = end_time - start_time
    else:
        start_time, end_time, koro_duration = 0, 0, 0

    # 3. Heart Rate
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_sig = signal.sosfiltfilt(sos_hr, phase_var)
    hr_smooth = signal.savgol_filter(hr_sig, 501, 3)
    
    # FFT Peak
    freqs_hr, psd_hr = signal.welch(hr_sig, fs, nperseg=int(fs*10))
    peak_freq = freqs_hr[np.argmax(psd_hr)]
    hr_bpm_fft = peak_freq * 60
    
    # PLOTTING
    fig = plt.figure(figsize=(16, 28))
    
    # Standard panels (simplified for the request)
    ax1 = plt.subplot(7, 2, 1); ax1.plot(time, i_centered); ax1.set_title('I Channel')
    ax2 = plt.subplot(7, 2, 2); ax2.plot(time, q_centered); ax2.set_title('Q Channel')
    
    # Panel 3: Korotkoff Velocity with ENERGY WINDOW
    ax3 = plt.subplot(7, 1, 2)
    ax3.plot(time, koro_filtered, color='purple', alpha=0.4)
    ax3.plot(time, energy * 5, color='orange', linewidth=2, label='Energy Envelope (scaled)')
    ax3.axvline(start_time, color='red', linestyle='--', label='Start')
    ax3.axvline(end_time, color='red', linestyle='--', label='End')
    ax3.set_title(f'Korotkoff Energy Window Analysis | Detected Duration: {koro_duration:.2f} s')
    ax3.set_ylabel('Velocity / Energy')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Heart Rate Confirmation
    ax4 = plt.subplot(7, 2, 5)
    ax4.plot(time, hr_smooth, color='brown')
    ax4.set_title(f'HR Signal ({hr_bpm_fft:.1f} BPM)')
    
    ax5 = plt.subplot(7, 2, 6)
    ax5.semilogy(freqs_hr, psd_hr, color='brown')
    ax5.set_xlim(0, 5)
    ax5.set_title('HR Spectrum')
    
    # Panel 5: Spectrogram (Activity Proof)
    ax6 = plt.subplot(7, 1, 4)
    f, t, Sxx = signal.spectrogram(koro_filtered, fs, nperseg=int(fs/4))
    ax6.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    ax6.set_ylim(0, 60)
    ax6.axvline(start_time, color='white', linestyle='--')
    ax6.axvline(end_time, color='white', linestyle='--')
    ax6.set_title('Spectrogram with Activity Window')
    
    # Panel 6: Summary Text
    ax7 = plt.subplot(7, 1, 5)
    ax7.axis('off')
    results_text = (f"DURATION ANALYSIS FOR {file_name}\n"
                    f"----------------------------------------\n"
                    f"Expected Duration: ~10.0 s\n"
                    f"Calculated Duration: {koro_duration:.2f} s\n"
                    f"Window: {start_time:.2f}s to {end_time:.2f}s\n"
                    f"Heart Rate: {hr_bpm_fft:.1f} BPM")
    ax7.text(0.1, 0.5, results_text, fontsize=14, family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Improved duration plot saved to: {output_img}")

if __name__ == '__main__':
    run_final_analysis()
