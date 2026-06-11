import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

# CONFIGURATION
FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_may12_2.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\advanced_koro_validation_v2_may12_2.png'

def sliding_kurtosis(x, window_size):
    return pd.Series(x).rolling(window=window_size).kurt().fillna(0).values

def sliding_rms(x, window_size):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=window_size).mean().fillna(0).values)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return tkeo

def notch_filter(x, f0, fs, Q=30):
    """Apply a notch filter at frequency f0 Hz."""
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

def run_pressure_energy_analysis():
    print(f"Starting Advanced Analysis (v2 - Improved) for: {os.path.basename(FILE_PATH)}")
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return
        
    with h5py.File(FILE_PATH, 'r') as f:
        data = f['data'][:]
        
    i_raw = data[0, :]
    q_raw = data[1, :]
    time = np.arange(len(i_raw)) / FS
    
    # 2. PRE-PROCESSING & DETRENDING
    i_c = i_raw - np.mean(i_raw)
    q_c = q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    
    mag = np.abs(iq)
    phase_rad = np.unwrap(np.angle(iq))
    phase_rad = signal.detrend(phase_rad)
    
    # Physical Unit Conversion (0.9 GHz radar)
    lambda_mm = (299792458 / 0.9e9) * 1000  # wavelength in mm
    phase_mm = (phase_rad * lambda_mm) / (4 * np.pi)  # displacement in mm
    
    # 2b. NOTCH FILTER - Remove 50 Hz powerline interference
    mag = notch_filter(mag, 50.0, FS)
    phase_mm = notch_filter(phase_mm, 50.0, FS)
    
    # 3. CUFF TREND PROXY (very low-pass on magnitude)
    sos_cuff = signal.butter(4, 0.2, btype='lowpass', fs=FS, output='sos')
    cuff_trend = signal.sosfiltfilt(sos_cuff, mag)
    
    # 4. FILTERING
    # Magnitude pulse: 1-49 Hz (below 50 Hz notch)
    sos_mag = signal.butter(4, [1, 49], btype='bandpass', fs=FS, output='sos')
    mag_pulse = signal.sosfiltfilt(sos_mag, mag)
    mag_env = np.abs(signal.hilbert(mag_pulse))
    
    # Phase velocity: derivative of displacement -> velocity (mm/s)
    phase_vel_raw = np.diff(phase_mm) * FS
    phase_vel_raw = np.append(phase_vel_raw, phase_vel_raw[-1])
    sos_vel = signal.butter(4, [10, 49], btype='bandpass', fs=FS, output='sos')
    phase_vel = signal.sosfiltfilt(sos_vel, phase_vel_raw)
    
    # 4b. WIDEBAND FILTERS FOR TFD (0.5 - 150 Hz)
    sos_wide = signal.butter(4, [0.5, 150], btype='bandpass', fs=FS, output='sos')
    mag_wide = signal.sosfiltfilt(sos_wide, mag)
    phase_disp_wide = signal.sosfiltfilt(sos_wide, phase_mm)
    
    # 5. TKEO + ACTIVE WINDOW DETECTION (before beat detection, so we can use quiet segment)
    vel_tkeo = calc_tkeo(phase_vel)
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
    
    # ENFORCE DURATION CONSTRAINTS (2 to 12 seconds - relaxed from original)
    dur = off_s - on_s
    if dur < 2.0:
        pad = (2.0 - dur) / 2.0
        on_s, off_s = max(0.0, on_s - pad), min(time[-1], off_s + pad)
    elif dur > 12.0:
        pad = (dur - 12.0) / 2.0
        on_s, off_s = on_s + pad, off_s - pad
    dur = off_s - on_s
    
    # 6. BEAT DETECTION (improved: threshold from quiet post-Koro segment)
    quiet_start_idx = int(off_s * FS) + int(2 * FS)  # 2s after Koro window ends
    quiet_end_idx = min(len(mag_pulse), quiet_start_idx + int(8 * FS))
    if quiet_end_idx > quiet_start_idx + int(2 * FS):
        quiet_segment = mag_pulse[quiet_start_idx:quiet_end_idx]
        prom_thresh = np.std(quiet_segment) * 1.5
    else:
        prom_thresh = np.std(mag_pulse) * 1.0  # fallback
    
    peaks, _ = signal.find_peaks(mag_pulse, distance=int(FS*0.4), prominence=prom_thresh)
    if len(peaks) > 1:
        intervals = np.diff(time[peaks])
        valid_intervals = intervals[(intervals > 0.4) & (intervals < 1.5)]
        hr_bpm_time = 60.0 / np.median(valid_intervals) if len(valid_intervals) > 0 else 0
    else:
        hr_bpm_time = 0
    peak_times = time[peaks]
    peak_heights = mag_pulse[peaks]
    
    # 7. SLIDING STATS (Window = 0.5s)
    win_len = int(FS * 0.5)
    mag_kurt = sliding_kurtosis(mag_pulse, win_len)
    phase_kurt = sliding_kurtosis(phase_vel, win_len)
    phase_jitter = sliding_rms(phase_vel, win_len)
    
    mag_energy = sliding_rms(mag_pulse, win_len)**2
    phase_energy = sliding_rms(phase_vel, win_len)**2
    
    # 8. INSTANTANEOUS FREQUENCY
    analytic_koro = signal.hilbert(phase_vel)
    inst_phase = np.unwrap(np.angle(analytic_koro))
    inst_freq = (np.diff(inst_phase) / (2.0*np.pi) * FS)
    inst_freq = np.append(inst_freq, inst_freq[-1])
    inst_freq_smooth = pd.Series(inst_freq).rolling(window=int(FS*0.05)).median().fillna(0).values
    inst_freq_smooth = np.clip(inst_freq_smooth, 0, 100)
    
    # 9. FREQUENCY DOMAIN ANALYSIS
    # HR PSD - computed on quiet post-Koro segment for reliable HR
    if quiet_end_idx > quiet_start_idx + int(2 * FS):
        quiet_phase = phase_mm[quiet_start_idx:quiet_end_idx]
        nperseg_hr = min(len(quiet_phase), int(FS * 5))
        f_hr, p_hr = signal.welch(quiet_phase, fs=FS, nperseg=nperseg_hr)
    else:
        f_hr, p_hr = signal.welch(phase_mm, fs=FS, nperseg=int(FS*10))
    hr_mask = (f_hr >= 0.8) & (f_hr <= 3.0)
    if np.any(hr_mask):
        hr_peak_f = f_hr[hr_mask][np.argmax(p_hr[hr_mask])]
        hr_bpm_freq = hr_peak_f * 60
    else:
        hr_peak_f, hr_bpm_freq = 0, 0
        
    # Active vs Noise Spectra (noise = post-Koro quiet segment)
    if on_s < off_s:
        idx_on = int(on_s * FS)
        idx_off = int(off_s * FS)
        active_vel = phase_vel[idx_on:idx_off]
        active_len = len(active_vel)
        # Use post-Koro segment as noise reference
        noise_start = min(idx_off + int(2 * FS), len(phase_vel) - active_len)
        noise_start = max(noise_start, 0)
        noise_end = min(noise_start + active_len, len(phase_vel))
        noise_vel = phase_vel[noise_start:noise_end]
        nperseg_spec = min(1024, len(active_vel))
        f_act, p_act = signal.welch(active_vel, fs=FS, nperseg=nperseg_spec)
        f_noi, p_noi = signal.welch(noise_vel, fs=FS, nperseg=nperseg_spec)
        # SNR in Korotkoff band
        koro_mask = (f_act >= 10) & (f_act <= 49)
        if np.any(koro_mask) and np.mean(p_noi[koro_mask]) > 0:
            snr_db = 10 * np.log10(np.mean(p_act[koro_mask]) / np.mean(p_noi[koro_mask]))
        else:
            snr_db = 0
    else:
        f_act, p_act = np.zeros(513), np.zeros(513)
        f_noi, p_noi = np.zeros(513), np.zeros(513)
        snr_db = 0
        
    # 10. KOROTKOFF ENERGY ENVELOPE (for replacement panel)
    koro_energy_env = sliding_rms(phase_vel, int(FS * 0.2))**2
    koro_thresh_line = np.mean(koro_energy_env) + 2 * np.std(koro_energy_env)
    
    # ===================== PLOTTING (18 Panels: 9 rows x 2 cols) =====================
    fig, axes = plt.subplots(9, 2, figsize=(22, 46))
    plt.subplots_adjust(hspace=0.50)
    
    # ROW 1: Full Recording Overview
    ax = axes[0, 0]
    ax.plot(time, mag, label='Raw Magnitude', color='blue', alpha=0.3)
    ax.plot(time, cuff_trend, label='Cuff Trend (0.2 Hz LP)', color='black', linewidth=2)
    if on_s < off_s: 
        ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label=f'Koro Window ({dur:.1f}s)')
        ax.axvline(on_s, color='red', linestyle='--', linewidth=2, label='SYS Point')
        ax.axvline(off_s, color='blue', linestyle='--', linewidth=2, label='DIA Point')
    ax.set_title('1. Full Recording: Magnitude + Cuff Trend')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    ax = axes[0, 1]
    ax.plot(time, phase_mm, label='Detrended Phase Displacement', color='red', alpha=0.8)
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_title('1. Full Recording: Physical Displacement')
    ax.set_ylabel('Displacement (mm)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    # ROW 2: Filtered Signals + Beat Detection
    ax = axes[1, 0]
    ax.plot(time, mag_pulse, label='Bandpass 1-49 Hz', color='blue', alpha=0.5)
    ax.plot(time, mag_env, label='Hilbert Envelope', color='gray', alpha=0.5)
    ax.plot(peak_times, peak_heights, 'ro', label=f'Beats ({len(peaks)} detected)', markersize=6)
    ax.text(0.05, 0.85, f'Time HR: {hr_bpm_time:.1f} BPM\n({len(peaks)} beats)',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title('2. Magnitude Pulse Waveform (1-49 Hz)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    ax = axes[1, 1]
    ax.plot(time, phase_vel, label='Phase Velocity (10-49 Hz)', color='darkred')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_title('2. Phase Velocity dφ/dt (10-49 Hz)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    # ROW 3: Power Spectral Density
    ax = axes[2, 0]
    f_mag, p_mag = signal.welch(mag, fs=FS, nperseg=FS*2)
    ax.semilogy(f_mag, p_mag, color='blue', label='Magnitude PSD')
    ax.axvspan(10, 49, color='yellow', alpha=0.2, label='Korotkoff Band (10-49 Hz)')
    ax.axvline(50, color='red', linestyle=':', alpha=0.5, label='50 Hz Notch')
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=1e-12)
    ax.set_title('3. Magnitude Power Spectral Density')
    ax.set_ylabel('PSD (a.u.²/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8)
    
    ax = axes[2, 1]
    f_ph, p_ph = signal.welch(phase_mm, fs=FS, nperseg=FS*2)
    ax.semilogy(f_ph, p_ph, color='red', label='Displacement PSD')
    ax.axvspan(10, 49, color='yellow', alpha=0.2, label='Korotkoff Band (10-49 Hz)')
    ax.axvline(50, color='red', linestyle=':', alpha=0.5, label='50 Hz Notch')
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=1e-12)
    ax.set_title('3. Displacement Power Spectral Density')
    ax.set_ylabel('PSD (mm²/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8)
    
    # ROW 4: Spectrograms (STFT)
    ax = axes[3, 0]
    f_sm, t_sm, Zxx_m = signal.stft(mag_wide, fs=FS, nperseg=4096, noverlap=3072)
    Pxx_m = 10 * np.log10(np.abs(Zxx_m)**2 + 1e-12)
    vmin_m, vmax_m = np.percentile(Pxx_m, [50, 99.9])
    im1 = ax.pcolormesh(t_sm, f_sm, Pxx_m, shading='gouraud', cmap='viridis', vmin=vmin_m, vmax=vmax_m)
    if on_s < off_s:
        ax.axvline(on_s, color='white', linestyle='--')
        ax.axvline(off_s, color='white', linestyle='--')
    ax.set_ylim(0, 150)
    ax.set_title('4. Magnitude Spectrogram (STFT: 0.5-150 Hz)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    plt.colorbar(im1, ax=ax, label='Power (dB)')
    
    ax = axes[3, 1]
    f_sp, t_sp, Zxx_p = signal.stft(phase_disp_wide, fs=FS, nperseg=4096, noverlap=3072)
    Pxx_p = 10 * np.log10(np.abs(Zxx_p)**2 + 1e-12)
    vmin_p, vmax_p = np.percentile(Pxx_p, [50, 99.9])
    im2 = ax.pcolormesh(t_sp, f_sp, Pxx_p, shading='gouraud', cmap='plasma', vmin=vmin_p, vmax=vmax_p)
    if on_s < off_s:
        ax.axvline(on_s, color='white', linestyle='--')
        ax.axvline(off_s, color='white', linestyle='--')
    ax.set_ylim(0, 150)
    ax.set_title('4. Displacement Spectrogram (STFT: 0.5-150 Hz)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    plt.colorbar(im2, ax=ax, label='Power (dB)')
    
    # ROW 5: Sliding Kurtosis
    ax = axes[4, 0]
    ax.plot(time, mag_kurt, color='purple')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Detected Window')
    ax.set_title('5. Impulse Detection: Magnitude Sliding Kurtosis')
    ax.set_ylabel('Kurtosis (unitless)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    ax = axes[4, 1]
    ax.plot(time, phase_kurt, color='purple')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Detected Window')
    ax.set_title('5. Impulse Detection: Phase Velocity Sliding Kurtosis')
    ax.set_ylabel('Kurtosis (unitless)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    # ROW 6: Korotkoff Energy Envelope + Phase Velocity Jitter
    # (Replaces old Pulse Height Tracking)
    ax = axes[5, 0]
    ax.plot(time, koro_energy_env, color='teal', label='Phase Vel Energy Envelope')
    ax.axhline(koro_thresh_line, color='red', linestyle='--', label=f'Threshold (mean+2σ)')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_title('6. Korotkoff Energy Envelope + Detection Threshold')
    ax.set_ylabel('Energy ((mm/s)²)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    ax = axes[5, 1]
    ax.plot(time, phase_jitter, color='darkred', label='Velocity Jitter (RMS)')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_title('6. Phase Velocity Jitter (Sliding RMS)')
    ax.set_ylabel('RMS Velocity (mm/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    # ROW 7: Amplitude Spectrum (Koro Window Only)
    ax = axes[6, 0]
    if on_s < off_s:
        idx_on_w = int(on_s * FS)
        idx_off_w = int(off_s * FS)
        win_mag = mag[idx_on_w:idx_off_w]
        win_mag = win_mag - np.mean(win_mag)
        N_w = len(win_mag)
        freqs_w = np.fft.rfftfreq(N_w, 1/FS)
        fft_mag = np.abs(np.fft.rfft(win_mag)) / N_w
        ax.plot(freqs_w, fft_mag, color='blue', label='Magnitude FFT (window)')
        ax.set_xlim(0, 60)
        ax.set_title('7. Amplitude Spectrum: Magnitude (Koro Window)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(fontsize=8)
    else:
        ax.set_title('7. Amplitude Spectrum (No Window)')

    ax = axes[6, 1]
    if on_s < off_s:
        win_vel = phase_vel[idx_on_w:idx_off_w]
        N_v = len(win_vel)
        freqs_v = np.fft.rfftfreq(N_v, 1/FS)
        fft_vel = np.abs(np.fft.rfft(win_vel)) / N_v
        ax.plot(freqs_v, fft_vel, color='red', label='Phase Velocity FFT (window)')
        ax.set_xlim(0, 60)
        ax.set_title('7. Amplitude Spectrum: Phase Velocity (Koro Window)')
        ax.set_ylabel('Amplitude (mm/s)')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(fontsize=8)
    else:
        ax.set_title('7. Amplitude Spectrum (No Window)')

    # ROW 8: TKEO + Instantaneous Frequency
    ax = axes[7, 0]
    ax.plot(time, vel_tkeo, color='teal', label='TKEO Energy')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_title('8. TKEO (Teager-Kaiser Energy Operator)')
    ax.set_ylabel('TKEO Energy ((mm/s)²)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    ax = axes[7, 1]
    ax.plot(time, inst_freq_smooth, color='green', label='Instantaneous Frequency')
    if on_s < off_s: ax.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax.set_ylim(0, 100)
    ax.set_title('8. Instantaneous Frequency (clipped 0-100 Hz)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    
    # ROW 9: HR Validation + Active vs Noise
    ax = axes[8, 0]
    ax.semilogy(f_hr, p_hr, color='black', label='Displacement PSD (quiet segment)')
    if hr_peak_f > 0:
        ax.plot(hr_peak_f, p_hr[np.argmin(np.abs(f_hr - hr_peak_f))], 'ro', markersize=8)
        ax.axvline(hr_peak_f, color='red', linestyle='--')
        ax.text(0.55, 0.80, f'Freq HR: {hr_bpm_freq:.1f} BPM\nTime HR: {hr_bpm_time:.1f} BPM',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_xlim(0, 5)
    ax.set_ylim(bottom=1e-8)
    ax.set_title('9. HR Validation: PSD of Quiet Post-Koro Segment (0-5 Hz)')
    ax.set_ylabel('PSD (mm²/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8)
    
    ax = axes[8, 1]
    if on_s < off_s:
        ax.semilogy(f_act, p_act, color='red', label='Active Koro Window')
        ax.semilogy(f_noi, p_noi, color='blue', alpha=0.5, label='Post-Koro Noise Floor')
        ax.text(0.55, 0.80, f'SNR (10-49 Hz): {snr_db:.1f} dB',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='lime', alpha=0.5))
    ax.set_xlim(10, 50)
    ax.set_title('9. Korotkoff Band: Active vs Noise PSD + SNR')
    ax.set_ylabel('PSD ((mm/s)²/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8)
    
    plt.suptitle("Korotkoff Validation Dashboard v2: Multi-Domain Evidence with SNR",
                 fontsize=24, fontweight='bold', y=0.92)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\n18-Panel Plot saved to {OUTPUT_IMG}")
    
    # REPORT
    print(f"\n{'='*50}")
    print(f"  KOROTKOFF ANALYSIS REPORT (v2 - Improved)")
    print(f"{'='*50}")
    print(f"  Korotkoff Window  : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)")
    print(f"  Time-Domain HR    : {hr_bpm_time:.1f} BPM ({len(peaks)} beats)")
    print(f"  Frequency-Domain HR: {hr_bpm_freq:.1f} BPM")
    print(f"  Korotkoff SNR     : {snr_db:.1f} dB (10-49 Hz)")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    run_pressure_energy_analysis()
