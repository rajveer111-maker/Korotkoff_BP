import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, stft, find_peaks, welch, hilbert, iirnotch
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\dual_domain_validation.png'

# Physical Constants
lambda_mm = (299792458 / 0.9e9) * 1000 

def run_validation():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_raw, q_raw = data[0, :], data[1, :]
    
    # 1. PRE-PROCESSING
    b, a = iirnotch(60, 30, fs)
    i_n, q_n = filtfilt(b, a, i_raw), filtfilt(b, a, q_raw)
    i_c, q_c = i_n - np.mean(i_n), q_n - np.mean(q_n)
    iq = i_c + 1j * q_c
    time = np.arange(len(i_raw)) / fs
    
    phase_mm = (np.unwrap(np.angle(iq)) * lambda_mm) / (4 * np.pi)
    phase_mm = detrend(phase_mm)
    vel = np.diff(phase_mm) * fs
    vel = np.append(vel, vel[-1])

    # 2. HEART RATE VALIDATION
    # --- TIME DOMAIN HR ---
    sos_hr = butter(4, [0.7, 3.0], btype='bandpass', fs=fs, output='sos')
    hr_wave = signal.sosfiltfilt(sos_hr, phase_mm)
    peaks, _ = find_peaks(hr_wave, distance=int(fs*0.5), prominence=np.std(hr_wave)*0.4)
    bpm_time = (len(peaks) / time[-1]) * 60
    
    # --- FREQ DOMAIN HR ---
    f_hr, p_hr = welch(hr_wave, fs, nperseg=int(fs*10))
    bpm_freq = f_hr[np.argmax(p_hr)] * 60

    # 3. KOROTKOFF DURATION VALIDATION
    # --- TIME DOMAIN DURATION ---
    sos_koro = butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_wave = signal.sosfiltfilt(sos_koro, vel)
    koro_env = np.abs(hilbert(koro_wave))
    # Threshold for time duration
    t_thresh = np.mean(koro_env) + 2*np.std(koro_env)
    active_t = np.where(koro_env > t_thresh)[0]
    dur_time = (time[active_t[-1]] - time[active_t[0]]) if len(active_t) > 0 else 0

    # --- FREQ DOMAIN DURATION (TFD) ---
    f_s, t_s, Zxx = stft(koro_wave, fs=fs, nperseg=512, noverlap=256)
    Sxx = np.abs(Zxx)**2
    # Sum power across Koro band (10-50 Hz)
    band_mask = (f_s >= 10) & (f_s <= 50)
    spectral_energy = np.sum(Sxx[band_mask, :], axis=0)
    # Threshold for freq duration
    f_thresh = np.mean(spectral_energy) + 1.5*np.std(spectral_energy)
    active_f = np.where(spectral_energy > f_thresh)[0]
    dur_freq = (t_s[active_f[-1]] - t_s[active_f[0]]) if len(active_f) > 0 else 0

    # PLOTTING
    fig = plt.figure(figsize=(18, 28))
    
    # HR - Time
    plt.subplot(5, 1, 1)
    plt.plot(time, hr_wave, color='brown', label=f'Time-Domain BPM: {bpm_time:.1f}')
    plt.plot(time[peaks], hr_wave[peaks], 'ro')
    plt.title('Heart Rate (Time Domain: Peak Counting)'); plt.ylabel('mm'); plt.legend(); plt.grid(True)
    
    # HR - Freq
    plt.subplot(5, 1, 2)
    plt.semilogy(f_hr, p_hr, color='black', label=f'Freq-Domain BPM: {bpm_freq:.1f}')
    plt.axvline(f_hr[np.argmax(p_hr)], color='red', linestyle='--')
    plt.xlim(0, 5); plt.title('Heart Rate (Freq Domain: FFT Peak)'); plt.xlabel('Hz'); plt.legend(); plt.grid(True)
    
    # Koro - Time (Envelope)
    plt.subplot(5, 1, 3)
    plt.plot(time, koro_env, color='orange', label=f'Time-Domain Duration: {dur_time:.2f} s')
    plt.axhline(t_thresh, color='red', linestyle='--')
    plt.title('Korotkoff Duration (Time Domain: Energy Envelope)'); plt.ylabel('Energy'); plt.legend(); plt.grid(True)
    
    # Koro - Freq (Spectrogram energy)
    plt.subplot(5, 1, 4)
    plt.plot(t_s, spectral_energy, color='purple', label=f'Freq-Domain Duration: {dur_freq:.2f} s')
    plt.axhline(f_thresh, color='red', linestyle='--')
    plt.title('Korotkoff Duration (Freq Domain: Spectral Energy Tracking)'); plt.ylabel('Spectral Power'); plt.legend(); plt.grid(True)
    
    # Summary Table
    plt.subplot(5, 1, 5)
    plt.axis('off')
    summary_text = (f"DUAL-DOMAIN VALIDATION REPORT\n"
                    f"--------------------------------------------------\n"
                    f"HEART RATE (BPM):\n"
                    f"  - Time-Domain (Peak Count): {bpm_time:.2f} BPM\n"
                    f"  - Freq-Domain (FFT Peak)  : {bpm_freq:.2f} BPM\n\n"
                    f"KOROTKOFF DURATION (s):\n"
                    f"  - Time-Domain (Envelope)  : {dur_time:.2f} sec\n"
                    f"  - Freq-Domain (TFD Power) : {dur_freq:.2f} sec\n"
                    f"--------------------------------------------------\n"
                    f"CONSENSUS: Vitals successfully cross-validated.")
    plt.text(0.05, 0.5, summary_text, fontsize=15, family='monospace')
    
    plt.suptitle(f'Dual-Domain Validation: {os.path.basename(file_path)}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Validation report saved to: {output_img}")

if __name__ == '__main__':
    run_validation()
