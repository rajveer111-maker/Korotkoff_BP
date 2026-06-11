import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, stft, find_peaks, decimate, hilbert, iirnotch
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs_orig = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\robust_analysis_v4.png'

# Physical Constants (0.9 GHz)
c = 299792458
lambda_mm = (c / 0.9e9) * 1000 

def run_robust_v4():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    # 1. RAW IQ PROCESSING
    i_raw, q_raw = data[0, :], data[1, :]
    
    # APPLY 60 Hz NOTCH FILTER (Powerline Removal)
    b_notch, a_notch = iirnotch(60, 30, fs_orig)
    i_notched = filtfilt(b_notch, a_notch, i_raw)
    q_notched = filtfilt(b_notch, a_notch, q_raw)
    
    # IQ Centering
    i_c, q_c = i_notched - np.mean(i_notched), q_notched - np.mean(q_notched)
    iq = i_c + 1j * q_c
    
    time = np.arange(len(i_raw)) / fs_orig
    magnitude_mm = (np.abs(iq) * lambda_mm) / (4 * np.pi)
    phase_rad = np.unwrap(np.angle(iq))
    disp_mm = (phase_rad * lambda_mm) / (4 * np.pi)
    disp_mm = detrend(disp_mm)
    
    # Velocity
    vel_mms = np.diff(disp_mm) * fs_orig
    vel_mms = np.append(vel_mms, vel_mms[-1])
    
    # 2. DOWNSAMPLING
    dec = 100
    fs_hr = fs_orig / dec
    disp_ds = decimate(disp_mm, dec, ftype='fir')
    t_ds = np.arange(len(disp_ds)) / fs_hr
    
    # 3. FILTERS
    sos_hr = butter(4, [0.7, 3.0], btype='bandpass', fs=fs_hr, output='sos')
    hr_pulse = signal.sosfiltfilt(sos_hr, disp_ds)
    
    sos_koro = butter(4, [10, 50], btype='bandpass', fs=fs_orig, output='sos')
    koro_vel = signal.sosfiltfilt(sos_koro, vel_mms)
    koro_env = np.abs(hilbert(koro_vel))
    
    # 4. TFD CAPTURE (Optimized)
    f_s, t_s, Zxx = stft(koro_vel, fs=fs_orig, nperseg=512, noverlap=256)
    Sxx = 10 * np.log10(np.abs(Zxx)**2 + 1e-15)
    
    # 5. PEAK DETECTION
    peaks, _ = find_peaks(hr_pulse, distance=int(fs_hr*0.5), prominence=np.std(hr_pulse)*0.4)
    bpm = (len(peaks) / t_ds[-1]) * 60 if t_ds[-1] > 0 else 0

    # PLOTTING
    fig = plt.figure(figsize=(18, 30))
    
    # Row 1: Raw Sensor
    plt.subplot(7, 2, 1); plt.plot(time, i_raw, color='blue', alpha=0.3); plt.plot(time, i_notched, color='blue'); plt.title('Raw I (Notched 60Hz)'); plt.grid(True)
    plt.subplot(7, 2, 2); plt.plot(time, q_raw, color='orange', alpha=0.3); plt.plot(time, q_notched, color='orange'); plt.title('Raw Q (Notched 60Hz)'); plt.grid(True)
    
    # Row 2: Physical Metrics (mm)
    plt.subplot(7, 2, 3); plt.plot(time, magnitude_mm - np.mean(magnitude_mm), color='green'); plt.title('NCS_am (mm)'); plt.ylabel('mm')
    plt.subplot(7, 2, 4); plt.plot(time, disp_mm, color='red'); plt.title('NCS_ph Displacement (mm)'); plt.ylabel('mm')
    
    # Row 3: Heart Rate Waveform
    plt.subplot(7, 1, 3)
    plt.plot(t_ds, hr_pulse, color='brown', label=f'Detected BPM: {bpm:.1f}')
    plt.plot(t_ds[peaks], hr_pulse[peaks], 'ro')
    plt.title('Cardiac Pulse (Notched)'); plt.ylabel('mm'); plt.legend(); plt.grid(True)
    
    # Row 4: Korotkoff Velocity
    plt.subplot(7, 1, 4)
    plt.plot(time, koro_vel, color='purple', label='Velocity')
    plt.title('Korotkoff Transients (10-50 Hz)'); plt.ylabel('mm/s'); plt.grid(True)
    
    # Row 5: Spectrogram
    plt.subplot(7, 1, 5)
    vmin = np.percentile(Sxx, 85)
    vmax = np.percentile(Sxx, 99.9)
    im = plt.pcolormesh(t_s, f_s, Sxx, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
    plt.ylim(0, 80); plt.title('TFD Analysis (60Hz Notched)'); plt.ylabel('Hz'); plt.colorbar(im, label='dB')
    
    # Row 6: Global Spectrum (Show the Notch)
    plt.subplot(7, 1, 6)
    f_full, p_full = signal.welch(disp_mm, fs_orig, nperseg=int(fs_orig*2))
    plt.semilogy(f_full, p_full, color='black')
    plt.axvline(60, color='red', linestyle='--', label='60 Hz Notch')
    plt.xlim(0, 100); plt.title('Global Spectrum (Proof of 60Hz Notch)'); plt.xlabel('Hz'); plt.legend(); plt.grid(True)
    
    plt.suptitle(f'Robust Analysis v4 (60Hz Notch Filtered): {os.path.basename(file_path)}', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Robust v4 with 60Hz notch saved to: {output_img}")

if __name__ == '__main__':
    run_robust_v4()
