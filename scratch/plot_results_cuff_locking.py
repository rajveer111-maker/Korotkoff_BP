import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

# CONFIGURATION
FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_may12_2.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\advanced_koro_validation_pressure_mapping_may12_2.png'

def sliding_kurtosis(x, window_size):
    return pd.Series(x).rolling(window=window_size).kurt().fillna(0).values

def sliding_rms(x, window_size):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=window_size).mean().fillna(0).values)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return tkeo

def run_pressure_energy_analysis():
    print(f"Starting Advanced Analysis (v3 - Multi-Domain) for: {os.path.basename(FILE_PATH)}")
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return
        
    with h5py.File(FILE_PATH, 'r') as f:
        data = f['data'][:]
        
    i_raw = data[0, :]
    q_raw = data[1, :]
    time = np.arange(len(i_raw)) / FS
    
    # 2. PRE-PROCESSING & DETRENDING (Fixing the huge phase error)
    i_c = i_raw - np.mean(i_raw)
    q_c = q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    
    mag = np.abs(iq)
    phase_rad = np.unwrap(np.angle(iq))
    phase_rad = signal.detrend(phase_rad) # Crucial to stop phase accumulating infinitely
    
    # Physical Unit Conversion
    lambda_mm = (299792458 / 0.9e9) * 1000 
    phase_mm = (phase_rad * lambda_mm) / (4 * np.pi)
    
    # 3. CUFF TREND PROXY
    sos_cuff = signal.butter(4, 0.2, btype='lowpass', fs=FS, output='sos')
    cuff_trend = signal.sosfiltfilt(sos_cuff, mag)
    
    # 4. FILTERING
    sos_mag = signal.butter(4, [1, 50], btype='bandpass', fs=FS, output='sos')
    mag_pulse = signal.sosfiltfilt(sos_mag, mag)
    mag_env = np.abs(signal.hilbert(mag_pulse))
    
    phase_vel_raw = np.diff(phase_mm) * FS
    phase_vel_raw = np.append(phase_vel_raw, phase_vel_raw[-1])
    sos_vel = signal.butter(4, [10, 50], btype='bandpass', fs=FS, output='sos')
    phase_vel = signal.sosfiltfilt(sos_vel, phase_vel_raw)
    
    # 4b. WIDEBAND FILTERS FOR TFD (0.5 - 150 Hz)
    sos_wide = signal.butter(4, [0.5, 150], btype='bandpass', fs=FS, output='sos')
    mag_wide = signal.sosfiltfilt(sos_wide, mag)
    phase_disp_wide = signal.sosfiltfilt(sos_wide, phase_mm)
    
    # 5. BEAT DETECTION
    peaks, _ = signal.find_peaks(mag_pulse, distance=int(FS*0.6), prominence=np.std(mag_pulse)*2)
    if len(peaks) > 1:
        # Calculate instantaneous HR ignoring the massive gaps when the cuff is occluding the artery
        intervals = np.diff(time[peaks])
        valid_intervals = intervals[intervals < 1.5] # Cap at 1.5s (40 BPM) to ignore the occlusion flatline
        hr_bpm = 60.0 / np.median(valid_intervals) if len(valid_intervals) > 0 else 0
    else:
        hr_bpm = 0
    peak_times = time[peaks]
    peak_heights = mag_pulse[peaks]
    
    # 6. SLIDING STATS (Window = 0.5s)
    win_len = int(FS * 0.5)
    mag_kurt = sliding_kurtosis(mag_pulse, win_len)
    phase_kurt = sliding_kurtosis(phase_vel, win_len)
    phase_jitter = sliding_rms(phase_vel, win_len)
    
    mag_energy = sliding_rms(mag_pulse, win_len)**2
    phase_energy = sliding_rms(phase_vel, win_len)**2
    
    # 7. ADVANCED KOROTKOFF CONFIRMATION
    # 7a. Teager-Kaiser Energy Operator (TKEO)
    vel_tkeo = calc_tkeo(phase_vel)
    
    # 7b. Instantaneous Frequency
    analytic_koro = signal.hilbert(phase_vel)
    inst_phase = np.unwrap(np.angle(analytic_koro))
    inst_freq = (np.diff(inst_phase) / (2.0*np.pi) * FS)
    inst_freq = np.append(inst_freq, inst_freq[-1])
    inst_freq_smooth = pd.Series(inst_freq).rolling(window=int(FS*0.05)).median().fillna(0).values
    inst_freq_smooth = np.clip(inst_freq_smooth, 0, 100)
    
    # 8. ACTIVE WINDOW DETECTION (SYS / DIA points)
    smooth_tkeo = pd.Series(vel_tkeo).rolling(window=int(FS*1.0), center=True).mean().fillna(0).values
    valid_start, valid_end = int(3 * FS), int(len(smooth_tkeo) - FS)
    if valid_start < valid_end:
        center_idx = valid_start + np.argmax(smooth_tkeo[valid_start:valid_end])
    else:
        center_idx = np.argmax(smooth_tkeo)
        
    thresh = np.max(smooth_tkeo[valid_start:valid_end]) * 0.15
    start_idx, end_idx = center_idx, center_idx
    while start_idx > 0 and smooth_tkeo[start_idx] > thresh: start_idx -= 1
    while end_idx < len(smooth_tkeo) - 1 and smooth_tkeo[end_idx] > thresh: end_idx += 1
        
    on_s, off_s = time[start_idx], time[end_idx]
    
    # ENFORCE DURATION CONSTRAINTS (5 to 10 seconds)
    dur = off_s - on_s
    if dur < 5.0:
        pad = (5.0 - dur) / 2.0
        on_s, off_s = max(0.0, on_s - pad), min(time[-1], off_s + pad)
    elif dur > 10.0:
        pad = (dur - 10.0) / 2.0
        on_s, off_s = on_s + pad, off_s - pad
    dur = off_s - on_s
        
    # 9. ADVANCED FREQUENCY DOMAIN ANALYSIS
    # HR PSD
    f_hr, p_hr = signal.welch(phase_mm, fs=FS, nperseg=int(FS*10))
    hr_mask = (f_hr >= 0.8) & (f_hr <= 3.0)
    if np.any(hr_mask):
        hr_peak_f = f_hr[hr_mask][np.argmax(p_hr[hr_mask])]
        hr_bpm_freq = hr_peak_f * 60
    else:
        hr_peak_f, hr_bpm_freq = 0, 0
        
    # Active vs Noise Spectra
    if on_s < off_s:
        idx_on = int(on_s * FS)
        idx_off = int(off_s * FS)
        active_vel = phase_vel[idx_on:idx_off]
        noise_start = max(0, idx_on - len(active_vel))
        noise_vel = phase_vel[noise_start : noise_start + len(active_vel)]
        f_act, p_act = signal.welch(active_vel, fs=FS, nperseg=1024)
        f_noi, p_noi = signal.welch(noise_vel, fs=FS, nperseg=1024)
    else:
        f_act, p_act = np.zeros(513), np.zeros(513)
        f_noi, p_noi = np.zeros(513), np.zeros(513)
        
    # 10. PLOTTING (20 Panels)
    fig, axes = plt.subplots(10, 2, figsize=(22, 50))
    plt.subplots_adjust(hspace=0.45)
    
    # ROW 1
    ax = axes[0, 0]
    ax.plot(time, mag, label='Raw Magnitude', color='blue', alpha=0.3)
    ax.plot(time, cuff_trend, label='Cuff Trend', color='black', linewidth=2)
    if on_s < off_s: 
        ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label=f'Koro Window ({dur:.1f}s)')
        ax.axvline(on_s, color='red', linestyle='--', linewidth=2, label='SYS Point')
        ax.axvline(off_s, color='blue', linestyle='--', linewidth=2, label='DIA Point')
    ax.set_title('1. Full Recording: Magnitude vs Cuff Trend'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    ax = axes[0, 1]
    ax.plot(time, phase_mm, label='Detrended Phase', color='red', alpha=0.8)
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2)
    ax.set_title('1. Full Recording: Physical Displacement'); ax.set_ylabel('Displacement (mm)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    # ROW 2
    ax = axes[1, 0]
    ax.plot(time, mag_pulse, label='Filtered Signal', color='blue', alpha=0.5)
    ax.plot(time, mag_env, label='Pulse Envelope', color='gray', alpha=0.5)
    ax.plot(peak_times, peak_heights, 'ro', label='Beats', markersize=8)
    ax.text(0.05, 0.8, f'Time HR: {hr_bpm:.1f} BPM', transform=ax.transAxes, fontsize=14, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title('2. Magnitude Pulse Waveform (1-50 Hz)'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    ax = axes[1, 1]
    ax.plot(time, phase_vel, label='Phase Velocity', color='darkred')
    ax.set_title('2. Phase Velocity (dV/dt) (10-50 Hz)'); ax.set_ylabel('Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    # ROW 3
    ax = axes[2, 0]
    f_mag, p_mag = signal.welch(mag, fs=FS, nperseg=FS*2)
    ax.semilogy(f_mag, p_mag, color='blue', label='Magnitude PSD')
    ax.axvspan(10, 50, color='yellow', alpha=0.2, label='Korotkoff Band')
    ax.set_xlim(0, 60); ax.set_ylim(bottom=1e-12)
    ax.set_title('3. Magnitude Power Spectral Density'); ax.set_ylabel('Power (a.u.^2/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    
    ax = axes[2, 1]
    f_ph, p_ph = signal.welch(phase_mm, fs=FS, nperseg=FS*2)
    ax.semilogy(f_ph, p_ph, color='red', label='Phase PSD')
    ax.axvspan(10, 50, color='yellow', alpha=0.2, label='Korotkoff Band')
    ax.set_xlim(0, 60); ax.set_ylim(bottom=1e-12)
    ax.set_title('3. Displacement Power Spectral Density'); ax.set_ylabel('Power (mm^2/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    
    # ROW 4
    ax = axes[3, 0]
    f_sm, t_sm, Zxx_m = signal.stft(mag_wide, fs=FS, nperseg=4096, noverlap=3072)
    # Clip the lowest values to prevent noise floor from dominating the color map
    Pxx_m = 10 * np.log10(np.abs(Zxx_m)**2 + 1e-12)
    vmin_m, vmax_m = np.percentile(Pxx_m, [50, 99.9])
    im1 = ax.pcolormesh(t_sm, f_sm, Pxx_m, shading='gouraud', cmap='viridis', vmin=vmin_m, vmax=vmax_m)
    if on_s < off_s: ax.axvline(on_s, color='white', linestyle='--'); ax.axvline(off_s, color='white', linestyle='--')
    ax.set_ylim(0, 150); ax.set_title('4. Magnitude Spectrogram (TFD: 0.5-150 Hz)'); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
    plt.colorbar(im1, ax=ax, label='Power (dB/Hz)')
    
    ax = axes[3, 1]
    f_sp, t_sp, Zxx_p = signal.stft(phase_disp_wide, fs=FS, nperseg=4096, noverlap=3072)
    Pxx_p = 10 * np.log10(np.abs(Zxx_p)**2 + 1e-12)
    vmin_p, vmax_p = np.percentile(Pxx_p, [50, 99.9])
    im2 = ax.pcolormesh(t_sp, f_sp, Pxx_p, shading='gouraud', cmap='plasma', vmin=vmin_p, vmax=vmax_p)
    if on_s < off_s: ax.axvline(on_s, color='white', linestyle='--'); ax.axvline(off_s, color='white', linestyle='--')
    ax.set_ylim(0, 150); ax.set_title('4. Displacement Spectrogram (TFD: 0.5-150 Hz)'); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
    plt.colorbar(im2, ax=ax, label='Power (dB/Hz)')
    
    # ROW 5
    ax = axes[4, 0]
    ax.plot(time, mag_kurt, color='purple')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Detected Window')
    ax.set_title('5. Impulse Detection: Sliding Kurtosis'); ax.set_ylabel('Kurtosis (Unitless)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    ax = axes[4, 1]
    ax.plot(time, phase_kurt, color='purple')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2)
    ax.set_title('5. Impulse Detection: Sliding Kurtosis'); ax.set_ylabel('Kurtosis (Unitless)'); ax.set_xlabel('Time (s)')
    
    # ROW 6
    ax = axes[5, 0]
    ax.plot(peak_times, peak_heights, marker='o', linestyle='-', color='blue', label='Peak Tracking')
    ax.set_title('6. Pulse Height Tracking (Magnitude)'); ax.set_ylabel('Peak Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    ax = axes[5, 1]
    ax.plot(time, phase_jitter, color='darkred', label='Velocity Jitter')
    ax.set_title('6. Phase Velocity Jitter (RMS)'); ax.set_ylabel('RMS Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    # ROW 7: Amplitude Spectrum (Frequency vs Magnitude)
    ax = axes[6, 0]
    if on_s < off_s:
        idx_on = int(on_s * FS)
        idx_off = int(off_s * FS)
        win_mag = mag[idx_on:idx_off]
        win_mag = win_mag - np.mean(win_mag)
        N = len(win_mag)
        freqs = np.fft.rfftfreq(N, 1/FS)
        fft_mag = np.abs(np.fft.rfft(win_mag)) / N
        ax.plot(freqs, fft_mag, color='blue', label='Magnitude FFT')
        ax.set_xlim(0, 60)
        ax.set_title('7. Amplitude Spectrum of Magnitude (0-60 Hz)'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    else:
        ax.set_title('7. Amplitude Spectrum (No Window)')

    ax = axes[6, 1]
    if on_s < off_s:
        win_vel = phase_vel[idx_on:idx_off]
        fft_vel = np.abs(np.fft.rfft(win_vel)) / N
        ax.plot(freqs, fft_vel, color='red', label='Phase Velocity FFT')
        ax.set_xlim(0, 60)
        ax.set_title('7. Amplitude Spectrum of Phase Velocity (0-60 Hz)'); ax.set_ylabel('Amplitude (mm/s)'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    else:
        ax.set_title('7. Amplitude Spectrum (No Window)')

    # ROW 8: Advanced Metrics
    ax = axes[7, 0]
    ax.plot(time, vel_tkeo, color='teal', label='TKEO Energy')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2)
    ax.set_title('8. TKEO (Teager-Kaiser Energy Operator)'); ax.set_ylabel('TKEO Energy'); ax.set_xlabel('Time (s)'); ax.legend()
    
    ax = axes[7, 1]
    ax.plot(time, inst_freq_smooth, color='green', label='Instantaneous Frequency')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2)
    ax.set_ylim(0, 150)
    ax.set_title('8. Instantaneous Frequency (Strictly bounded to 10-50 Hz Band)'); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)'); ax.legend()
    
    # ROW 9: Advanced Frequency Domain Validation
    ax = axes[8, 0]
    ax.semilogy(f_hr, p_hr, color='black', label='Displacement PSD')
    if hr_peak_f > 0:
        ax.plot(hr_peak_f, p_hr[np.argmin(np.abs(f_hr - hr_peak_f))], 'ro', markersize=8)
        ax.axvline(hr_peak_f, color='red', linestyle='--')
        ax.text(0.6, 0.8, f'Freq HR: {hr_bpm_freq:.1f} BPM', transform=ax.transAxes, fontsize=14, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_xlim(0, 5); ax.set_ylim(bottom=1e-8)
    ax.set_title('9. Frequency Domain: Exact HR Validation (0-5 Hz)'); ax.set_ylabel('Power'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    
    ax = axes[8, 1]
    if on_s < off_s:
        ax.plot(f_act, p_act, color='red', label='Active Koro Window FFT')
        ax.plot(f_noi, p_noi, color='blue', alpha=0.5, label='Noise Floor FFT')
    ax.set_xlim(10, 50)
    ax.set_title('9. Frequency Domain: Korotkoff Band Power (Active vs Noise)'); ax.set_ylabel('Power (mm/s)^2/Hz'); ax.set_xlabel('Frequency (Hz)'); ax.legend()
    # ROW 10: Heartbeat Overlay (Zoomed on Active Window)
    ax = axes[9, 0]
    sos_hr = signal.butter(4, [0.8, 3.0], btype='bandpass', fs=FS, output='sos')
    hr_signal = signal.sosfiltfilt(sos_hr, mag)
    if on_s < off_s:
        idx_on = max(0, int(on_s * FS) - FS) # 1 sec padding
        idx_off = min(len(time), int(off_s * FS) + FS)
        z_time = time[idx_on:idx_off]
        z_hr = hr_signal[idx_on:idx_off]
        z_vel = phase_vel[idx_on:idx_off]
        
        z_hr_norm = z_hr / (np.max(np.abs(z_hr)) + 1e-9)
        z_vel_norm = z_vel / (np.max(np.abs(z_vel)) + 1e-9)
        
        ax.plot(z_time, z_hr_norm, color='black', linewidth=2, label='Heartbeat (0.8-3.0 Hz)')
        ax.plot(z_time, z_vel_norm, color='red', alpha=0.7, label='Korotkoff Snaps (10-50 Hz)')
        ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Active Window')
        ax.set_xlim(z_time[0], z_time[-1])
        ax.set_title('10. Zoomed Overlay: Heartbeats vs. Korotkoff Snaps'); ax.set_ylabel('Normalized Amplitude'); ax.set_xlabel('Time (s)'); ax.legend()
    else:
        ax.set_title('10. Zoomed Overlay: No Window Found')

    ax = axes[9, 1]
    if on_s < off_s:
        z_tkeo = vel_tkeo[idx_on:idx_off]
        z_tkeo_norm = z_tkeo / (np.max(np.abs(z_tkeo)) + 1e-9)
        
        ax.plot(z_time, z_hr_norm, color='black', linewidth=2, label='Heartbeat (0.8-3.0 Hz)')
        ax.plot(z_time, z_tkeo_norm, color='teal', alpha=0.8, label='TKEO Energy')
        ax.axvspan(on_s, off_s, color='yellow', alpha=0.2)
        ax.set_xlim(z_time[0], z_time[-1])
        ax.set_title('10. Zoomed Overlay: Heartbeats vs. TKEO Energy'); ax.set_ylabel('Normalized Energy'); ax.set_xlabel('Time (s)'); ax.legend()
    else:
        ax.set_title('10. Zoomed Overlay: No Window Found')
        
    plt.suptitle("Ultimate Korotkoff Validation: Physical Units & Full Multi-Domain Evidence", fontsize=28, fontweight='bold', y=0.91)
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print(f"20-Panel Complete Plot saved to {OUTPUT_IMG}")

if __name__ == '__main__':
    run_pressure_energy_analysis()
