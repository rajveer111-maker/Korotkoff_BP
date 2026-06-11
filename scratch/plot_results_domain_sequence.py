import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, stft, find_peaks, welch, iirnotch
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\domain_sequence_analysis.png'

# Physical Constants
lambda_mm = (299792458 / 0.9e9) * 1000 

def run_sequence_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_raw, q_raw = data[0, :], data[1, :]
    
    # --- PRE-PROCESSING ---
    # 1. 60Hz Notch
    b, a = iirnotch(60, 30, fs)
    i_n, q_n = filtfilt(b, a, i_raw), filtfilt(b, a, q_raw)
    
    # 2. IQ Centering
    i_c, q_c = i_n - np.mean(i_n), q_n - np.mean(q_n)
    iq = i_c + 1j * q_c
    
    # --- DOMAIN 1: TIME DOMAIN EXTRACTION ---
    time = np.arange(len(i_raw)) / fs
    mag_au = np.abs(iq)
    phase_rad = np.unwrap(np.angle(iq))
    phase_mm = (phase_rad * lambda_mm) / (4 * np.pi)
    phase_mm = detrend(phase_mm)
    
    # --- DOMAIN 2: FREQUENCY DOMAIN (FFT) ---
    # Focus on HR band for clear visualization
    f_mag, p_mag = welch(mag_au - np.mean(mag_au), fs, nperseg=int(fs*5))
    f_ph, p_ph = welch(phase_mm, fs, nperseg=int(fs*5))
    
    # --- DOMAIN 3: TIME-FREQUENCY DOMAIN (TFD) ---
    # Use velocity for best Koro capture
    vel = np.diff(phase_mm) * fs
    vel = np.append(vel, vel[-1])
    f_s, t_s, Zxx = stft(vel, fs=fs, nperseg=512, noverlap=256)
    Sxx = 10 * np.log10(np.abs(Zxx)**2 + 1e-15)

    # PLOTTING
    fig = plt.figure(figsize=(18, 26))
    
    # ROW 1: TIME DOMAIN
    plt.subplot(3, 2, 1)
    plt.plot(time, mag_au - np.mean(mag_au), color='green')
    plt.title('1. TIME DOMAIN: Magnitude (NCS_am)'); plt.ylabel('a.u.'); plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(time, phase_mm, color='red')
    plt.title('1. TIME DOMAIN: Phase Displacement (NCS_ph)'); plt.ylabel('mm'); plt.grid(True)
    
    # ROW 2: FREQUENCY DOMAIN
    plt.subplot(3, 2, 3)
    plt.semilogy(f_mag, p_mag, color='green')
    plt.xlim(0, 5); plt.title('2. FREQUENCY DOMAIN: Magnitude FFT'); plt.xlabel('Hz'); plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.semilogy(f_ph, p_ph, color='red')
    plt.xlim(0, 5); plt.title('2. FREQUENCY DOMAIN: Phase FFT'); plt.xlabel('Hz'); plt.grid(True)
    
    # ROW 3: TIME-FREQUENCY DOMAIN (TFD)
    plt.subplot(3, 1, 3)
    vmin, vmax = np.percentile(Sxx, 85), np.percentile(Sxx, 99.9)
    im = plt.pcolormesh(t_s, f_s, Sxx, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
    plt.ylim(0, 60); plt.title('3. TFD ANALYSIS: Spectrogram (Korotkoff Bursts)'); plt.ylabel('Hz'); plt.colorbar(im, label='dB')
    
    plt.suptitle(f'Sequential Analysis Pipeline: {os.path.basename(file_path)}\nTime -> Frequency -> TFD', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Sequential analysis saved to: {output_img}")

if __name__ == '__main__':
    run_sequence_analysis()
