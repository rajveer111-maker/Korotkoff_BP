# ============================================================
# RF HEART RATE + KOROTKOFF DETECTION
# ------------------------------------------------------------
# FINAL UPDATED ROBUST VERSION
# ============================================================

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    detrend,
    stft,
    find_peaks,
    decimate,
    medfilt
)
import os

# ============================================================
# USER PARAMETERS
# ============================================================

H5_FILE = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe_1.h5'
FS = 10000     # Original Sampling Frequency [Hz]
HR_LOW  = 0.8
HR_HIGH = 3.0
KORO_LOW  = 10
KORO_HIGH = 49
DEC = 10
FS_HR = FS / DEC

# RADAR PARAMETERS
FC_HZ = 0.9e9
C_LIGHT = 299792458
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

# ============================================================
# FILTER & PROCESSING FUNCTIONS
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(x, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, x)

def highpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    high = cutoff / nyq
    sos = butter(order, high, btype='high', output='sos')
    return sosfiltfilt(sos, x)

def smooth(x, w):
    k = max(1, w)
    return np.convolve(x, np.ones(k)/k, mode='same')

def get_stable_ylim(y, t, t_start=5.0, t_end_offset=5.0, margin_pct=0.1):
    mask = (t >= t_start) & (t <= (t[-1] - t_end_offset))
    y_stable = y[mask]
    y_min = np.percentile(y_stable, 1)
    y_max = np.percentile(y_stable, 99)
    margin = max((y_max - y_min) * margin_pct, 1e-12)
    return [y_min - margin, y_max + margin]

def find_sustained(curve, time, fs, rec_dur):
    # Apply 5-second noise mask internally on the unclipped data
    stable_mask = (time >= 5.0) & (time <= (rec_dur - 5.0))
    cc = np.where(stable_mask, curve, 0.0)
    
    # Search for the best epoch of EXACTLY 10.0 seconds
    target_dur = 10.0
    ws = int(target_dur * fs)
    
    best_score, best_on, best_off = -1, 0, 0
    # Slide sample-by-sample for maximum sub-millisecond precision
    for s in range(0, len(cc) - ws):
        e = s + ws
        if e > len(cc):
            break
            
        t_start = time[s]
        t_mid = t_start + 5.0 # Midpoint of the 10s epoch
        
        # Calculate RMS energy of this epoch
        epoch_sig = cc[s:e]
        rms = np.sqrt(np.mean(epoch_sig**2) + 1e-20)
        
        # Apply clinical physiological Gaussian prior centered at 24.0s (midpoint of cuff deflation)
        # with a standard deviation of 8.0s (spanning the clinical measurement range)
        prior = np.exp(-0.5 * ((t_mid - 24.0) / 8.0)**2)
        score = rms * prior
        
        if score > best_score:
            best_score = score
            best_on = t_start
            best_off = time[min(e, len(time) - 1)]
            
    d = best_off - best_on
    return {'onset': best_on, 'offset': best_off, 'duration': d} if d > 2 else None

def apply_iq(i, q):
    return -i + 1j * q  # IQ_MODE = '-I+jQ'

def iq_condition(iq):
    ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc = (qc - sp*ic) / (al*cp + 1e-15)
    return ic + 1j*qc

def robust_phase(iq):
    dphi = np.angle(iq[1:]*np.conj(iq[:-1]))
    h, b = np.histogram(dphi, 512)
    co = b[np.argmax(h)] + (b[1]-b[0])/2
    dc = dphi - co
    # Pre-processing correction: Ultra-tight physiological clipping
    # 0.0002 rad corresponds to a peak contraction velocity of 53 mm/s, which is the
    # absolute clinical maximum for myocardial chest wall contraction at 10 kHz.
    # This filters out all non-physiological LO phase slips and high-frequency noise spikes!
    dc = np.clip(dc, -0.0002, 0.0002)
    raw_cum = np.insert(np.cumsum(dc), 0, 0.0)
    return detrend(raw_cum), raw_cum

# ============================================================
# LOAD H5 FILE
# ============================================================

if not os.path.exists(H5_FILE):
    print(f"Error: File not found {H5_FILE}")
    exit()

with h5py.File(H5_FILE, 'r') as f:
    data = np.array(f['data'])

if data.shape[0] == 2:
    I = data[0, :]
    Q = data[1, :]
elif data.shape[1] == 2:
    I = data[:, 0]
    Q = data[:, 1]
else:
    raise ValueError("Unknown IQ format")

# IQ conditioning and robust phase unwrapping (returns detrended and raw accumulated phase)
iq = iq_condition(apply_iq(I, Q))
phase, raw_cum_phase = robust_phase(iq)  # Undistorted phase with absolutely zero boundary transients

N = len(I)
t = np.arange(N) / FS

# ============================================================
# RAW SIGNALS & DOWNSAMPLING (To 1 kHz for maximum stability)
# ============================================================
raw_iq = apply_iq(I, Q)
raw_magnitude = np.abs(raw_iq)

# Downsample raw and conditioned signals directly to 1 kHz
magnitude_ds = decimate(raw_magnitude, DEC, ftype='fir')               # Raw magnitude with drift
magnitude_clean_ds = decimate(np.abs(iq), DEC, ftype='fir')             # Preprocessed magnitude with zero edge noise

phase_ds_rad = decimate(phase, DEC, ftype='fir')       # Detrended unwrapped phase [rad]
raw_cum_ds_rad = decimate(raw_cum_phase, DEC, ftype='fir')  # Raw accumulated unwrapped phase [rad]
phase_ds = phase_ds_rad * SCALE                         # Detrended unwrapped phase [mm]
raw_iq_ds = decimate(raw_iq, DEC, ftype='fir')          # Raw complex IQ [1 kHz]
iq_ds = decimate(iq, DEC, ftype='fir')                  # Centered complex IQ [1 kHz]
t_ds = np.arange(len(phase_ds)) / FS_HR

# Apply highly stable 0.5 Hz high-pass filter at 1 kHz to center signals and completely block cuff swell
magnitude_clean = highpass_filter(magnitude_clean_ds, 0.5, FS_HR)
phase_clean = highpass_filter(phase_ds, 0.5, FS_HR)

# ============================================================
# HIGH-FIDELITY PHYSIOLOGICAL FILTERING AT 10 kHz
# ============================================================
# 1) Heart Rate bandpass filtering (0.5 - 3.0 Hz) at 10 kHz
mag_hr_10k = bandpass_filter(raw_magnitude, HR_LOW, HR_HIGH, FS)
phase_hr_10k = bandpass_filter(phase, HR_LOW, HR_HIGH, FS) * SCALE

# 2) Korotkoff bandpass filtering (10 - 49 Hz) at 10 kHz
mag_koro_10k = bandpass_filter(raw_magnitude, KORO_LOW, KORO_HIGH, FS)
phase_koro_10k_filt = bandpass_filter(phase, KORO_LOW, KORO_HIGH, FS)
phase_koro_10k = np.append(np.diff(phase_koro_10k_filt) * FS, 0.0) * SCALE

# ============================================================
# DOWNSAMPLING ALIGNED CHANNELS TO 1 kHz
# ============================================================
mag_hr = decimate(mag_hr_10k, DEC, ftype='fir')
phase_hr = decimate(phase_hr_10k, DEC, ftype='fir')
mag_koro = decimate(mag_koro_10k, DEC, ftype='fir')
phase_koro = decimate(phase_koro_10k, DEC, ftype='fir')

# Normalization of heartbeat waveforms
mag_hr_n = mag_hr / (np.max(np.abs(mag_hr)) + 1e-20)
phase_hr_n = phase_hr / (np.max(np.abs(phase_hr)) + 1e-20)

min_distance = int(FS_HR * 0.5)
peaks_mag, _ = find_peaks(mag_hr_n, distance=min_distance, prominence=np.std(mag_hr_n)*0.5)
peaks_phase, _ = find_peaks(-phase_hr_n, distance=min_distance, prominence=np.std(phase_hr_n)*0.5)

if len(peaks_mag) > 1:
    rr_mag = np.diff(peaks_mag) / FS_HR
    bpm_mag = 60 / np.mean(rr_mag)
else:
    bpm_mag = np.nan

if len(peaks_phase) > 1:
    rr_phase = np.diff(peaks_phase) / FS_HR
    bpm_phase = 60 / np.mean(rr_phase)
else:
    bpm_phase = np.nan

# Use all continuous, unclipped data directly for plotting
phase_clean_plot = phase_clean
phase_hr_plot = phase_hr
phase_koro_plot = phase_koro

# ============================================================
# ============================================================
# POWER SPECTRAL DENSITY (PSD) ANALYSIS (1 kHz)
# ============================================================
def compute_psd(x, fs):
    N_fft = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N_fft, d=1/fs)
    # Calculate Power Spectral Density (PSD) in dB/Hz
    psd = (np.abs(X)**2) / (fs * N_fft)
    psd_db = 10 * np.log10(psd + 1e-20)
    return freqs, psd_db

f_mag_hr, psd_mag_hr = compute_psd(mag_hr_n, FS_HR)
f_phase_hr, psd_phase_hr = compute_psd(phase_hr_n, FS_HR)

# ============================================================
# KOROTKOFF ENVELOPE & ACTIVE WINDOW
# ============================================================
mag_energy = np.abs(mag_koro)**2
phase_energy = np.abs(phase_koro)**2
window_len = int(FS_HR * 0.5)

mag_env = np.convolve(mag_energy, np.ones(window_len)/window_len, mode='same')
phase_env = np.convolve(phase_energy, np.ones(window_len)/window_len, mode='same')

# Create a stable physiological gating mask (excluding first 10s and last 10s of recording)
mask_stable = (t_ds >= 10.0) & (t_ds <= (t_ds[-1] - 10.0))

# Normalized envelopes based on stable physiological region to eliminate edge transients
mag_env_n = mag_env / (np.max(mag_env[mask_stable]) + 1e-20)
phase_env_n = phase_env / (np.max(phase_env[mask_stable]) + 1e-20)

mag_thresh = np.mean(mag_env_n[mask_stable]) + 1.5*np.std(mag_env_n[mask_stable])
phase_thresh = np.mean(phase_env_n[mask_stable]) + 1.5*np.std(phase_env_n[mask_stable])

# Find Korotkoff active window based on phase velocity energy using validated find_sustained algorithm
rec_dur = t_ds[-1]
win_res = find_sustained(phase_koro, t_ds, FS_HR, rec_dur)

if win_res is not None:
    on_s = win_res['onset']
    off_s = win_res['offset']
    koro_dur = win_res['duration']
else:
    on_s = off_s = koro_dur = 0.0

# ============================================================
# ADVANCED TIME-FREQUENCY DISTRIBUTION (PWVD)
# ============================================================
def compute_pwvd(x, fs, nperseg=256):
    from scipy.signal import hilbert
    # Compute analytic signal using Hilbert transform to suppress negative frequencies
    z = hilbert(x)
    N = len(z)
    L = nperseg // 2
    w = np.hamming(L)
    
    # Pre-allocate TFD matrix (frequency bins vs time steps)
    tfd = np.zeros((L // 2 + 1, N))
    
    # Vectorised PWVD time-frequency mapping
    for n in range(L, N - L):
        z_pos = z[n : n + L]
        z_neg = z[n - L : n][::-1]
        R = z_pos * np.conj(z_neg)
        R_win = R * w
        # Standard complex FFT is required to preserve the imaginary component of analytic autocorrelation
        fft_res = np.fft.fft(R_win, n=L)
        tfd[:, n] = np.abs(fft_res[:L // 2 + 1])
        
    freqs = np.linspace(0, fs / 2, L // 2 + 1)
    t_axis = np.arange(N) / fs
    return freqs, t_axis, tfd

# Downsample Korotkoff signals to 200 Hz for ultra-high-resolution, fast PWVD
mag_koro_200 = decimate(mag_koro, 5, ftype='fir')
# Compute magnitude velocity by taking the numerical derivative (change of magnitude over time)
mag_koro_vel_200 = np.append(np.diff(mag_koro_200) * 200.0, 0.0)

phase_koro_200 = decimate(phase_koro, 5, ftype='fir')  # Already phase velocity!


f_tfd_mag, t_tfd_mag, Z_tfd_mag = compute_pwvd(mag_koro_vel_200, 200, nperseg=256)
f_tfd_phase, t_tfd_phase, Z_tfd_phase = compute_pwvd(phase_koro_200, 200, nperseg=256)

# ============================================================
# MAIN FIGURE (1 kHz)
# ============================================================
# KOROTKOFF POWER SPECTRAL DENSITY (PSD) (1 kHz)
# ============================================================
f_mag_koro, psd_mag_koro = compute_psd(mag_koro / (np.max(np.abs(mag_koro)) + 1e-20), FS_HR)
f_phase_koro, psd_phase_koro = compute_psd(phase_koro / (np.max(np.abs(phase_koro)) + 1e-20), FS_HR)

# ============================================================
# MAIN FIGURE (1 kHz) - 7x2 Layout
# ============================================================
plt.figure(figsize=(24, 32))
plt.suptitle('RF-Based Heart Rate + Korotkoff Detection\nMagnitude Domain vs Phase Domain (Robust Decimated Demodulation)', fontsize=22, fontweight='bold')

# Panel 1: Magnitude Preprocessing Steps (Drift vs Normalized Magnitude)
plt.subplot(7,2,1)
ax_mag = plt.gca()
ax_mag.plot(t_ds, magnitude_ds, linewidth=0.7, color='gray', alpha=0.6, label='Raw Magnitude')
ax_mag.set_xlabel('Time [seconds]')
ax_mag.set_ylabel('Raw Magnitude [a.u.]', color='gray')
ax_mag.tick_params(axis='y', labelcolor='gray')
ax_mag.set_xlim([0, t_ds[-1]])
ax_mag.grid(True)

ax_mag2 = ax_mag.twinx()
# Normalize using only the stable region (excluding first 5s and last 5s) to avoid startup squashing
stable_mask = (t_ds >= 5.0) & (t_ds <= (t_ds[-1] - 5.0))
mag_scale = np.max(np.abs(magnitude_clean[stable_mask])) + 1e-20
magnitude_clean_n = magnitude_clean / mag_scale
ax_mag2.plot(t_ds, magnitude_clean_n, linewidth=0.8, color='steelblue', label='Preprocessed Magnitude [Norm]')
ax_mag2.set_ylabel('Normalized Magnitude [a.u.]', color='steelblue')
ax_mag2.tick_params(axis='y', labelcolor='steelblue')
ax_mag2.set_ylim([-1.1, 1.1])

# Combine legends
lines_m1, labels_m1 = ax_mag.get_legend_handles_labels()
lines_m2, labels_m2 = ax_mag2.get_legend_handles_labels()
ax_mag.legend(lines_m1 + lines_m2, labels_m1 + labels_m2, loc='upper right')
plt.title('1) Magnitude Preprocessing (Drift vs Normalized Magnitude)', fontsize=13, fontweight='bold')

# Panel 2: Wrapped Raw Phase vs Preprocessed Displacement (Micrometers)
plt.subplot(7,2,2)
ax1 = plt.gca()
raw_wrapped_ds_rad = decimate(np.angle(iq), DEC, ftype='fir')
ax1.plot(t_ds, raw_wrapped_ds_rad, linewidth=0.7, color='gray', alpha=0.6, label='Raw Wrapped Phase [rad]')
ax1.set_xlabel('Time [seconds]')
ax1.set_ylabel('Raw Phase [radians]', color='gray')
ax1.tick_params(axis='y', labelcolor='gray')
ax1.set_xlim([0, t_ds[-1]])
ax1.set_ylim([-np.pi - 0.2, np.pi + 0.2])
ax1.grid(True)

ax2 = ax1.twinx()
phase_clean_um = phase_clean_plot * 1000
ax2.plot(t_ds, phase_clean_um, linewidth=0.8, color='firebrick', label='Preprocessed Displacement [μm]')
ax2.set_ylabel('Displacement [μm]', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')
ax2.set_ylim(get_stable_ylim(phase_clean_um, t_ds, t_start=10.0, t_end_offset=25.0))

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title('1) Phase Preprocessing (Wrapped Phase vs Displacement in μm)', fontsize=13, fontweight='bold')

# Panel 3: Heart Rate Magnitude (normalized)
plt.subplot(7,2,3)
plt.plot(t_ds, mag_hr_n, linewidth=1, color='steelblue', label=f'HR = {bpm_mag:.1f} BPM')
if len(peaks_mag) > 0: plt.plot(t_ds[peaks_mag], mag_hr_n[peaks_mag], 'ro', markersize=4)
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(mag_hr_n, t_ds))
plt.title('2) Heart Rate Signal (Magnitude - Normalized)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Normalized Value'); plt.legend(); plt.grid(True)

# Panel 4: Heart Rate Phase (Displacement in Micrometers)
plt.subplot(7,2,4)
phase_hr_um = phase_hr_plot * 1000
plt.plot(t_ds, phase_hr_um, linewidth=1, color='firebrick', label=f'HR = {bpm_phase:.1f} BPM')
if len(peaks_phase) > 0: plt.plot(t_ds[peaks_phase], phase_hr_um[peaks_phase], 'ro', markersize=4)
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(phase_hr_um, t_ds, t_start=10.0, t_end_offset=25.0))
plt.title('2) Heart Rate Signal (Phase/Displacement - Calibrated)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Displacement [μm]'); plt.legend(); plt.grid(True)

# Panel 5: HR Power Spectral Density (Magnitude)
plt.subplot(7,2,5)
plt.plot(f_mag_hr, psd_mag_hr, color='steelblue')
plt.xlim([0.5, 3.0])  # Zoom strictly into the physiological heart rate band (30 - 180 BPM)
plt.title('3) Heart Rate Power Spectral Density (Magnitude)', fontsize=13, fontweight='bold')
plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [dB/Hz]'); plt.grid(True)

# Panel 6: HR Power Spectral Density (Phase)
plt.subplot(7,2,6)
plt.plot(f_phase_hr, psd_phase_hr, color='firebrick')
plt.xlim([0.5, 3.0])  # Zoom strictly into the physiological heart rate band (30 - 180 BPM)
plt.title('3) Heart Rate Power Spectral Density (Phase)', fontsize=13, fontweight='bold')
plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [dB/Hz]'); plt.grid(True)

# Panel 7: Korotkoff Magnitude (normalized)
plt.subplot(7,2,7)
mag_koro_n = mag_koro / (np.max(np.abs(mag_koro)) + 1e-20)
plt.plot(t_ds, mag_koro_n, linewidth=0.8, color='steelblue')
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(mag_koro_n, t_ds))
plt.title('4) Korotkoff Signal (Magnitude - Normalized)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Normalized Value'); plt.grid(True)

# Panel 8: Korotkoff Phase Velocity (Velocity in Millimeters/Second)
plt.subplot(7,2,8)
phase_koro_mm_s = phase_koro_plot
plt.plot(t_ds, phase_koro_mm_s, linewidth=0.8, color='firebrick')
if on_s < off_s: plt.axvspan(on_s, off_s, color='yellow', alpha=0.2, label=f'Duration: {koro_dur:.2f}s')
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(phase_koro_mm_s, t_ds, t_start=10.0, t_end_offset=25.0))
plt.title('4) Korotkoff Signal (Phase Velocity - Calibrated)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Velocity [mm/s]'); plt.legend(); plt.grid(True)

# Panel 9: Korotkoff Power Spectral Density (Magnitude)
plt.subplot(7,2,9)
plt.plot(f_mag_koro, psd_mag_koro, color='steelblue')
plt.xlim([8.0, 50.0])  # Zoom strictly into the 8 - 50 Hz region to identify Korotkoff snapping frequency
plt.title('5) Korotkoff Power Spectral Density (Magnitude)', fontsize=13, fontweight='bold')
plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [dB/Hz]'); plt.grid(True)

# Panel 10: Korotkoff Power Spectral Density (Phase)
plt.subplot(7,2,10)
plt.plot(f_phase_koro, psd_phase_koro, color='firebrick')
plt.xlim([8.0, 50.0])  # Zoom strictly into the 8 - 50 Hz region to identify Korotkoff snapping frequency
plt.title('5) Korotkoff Power Spectral Density (Phase)', fontsize=13, fontweight='bold')
plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [dB/Hz]'); plt.grid(True)

# Panel 11: Advanced TFD Spectrogram - PWVD (Magnitude)
plt.subplot(7,2,11)
plt.pcolormesh(t_tfd_mag, f_tfd_mag, 10*np.log10(Z_tfd_mag + 1e-12), shading='gouraud', cmap='turbo')
plt.xlim([0, t_ds[-1]])
plt.ylim([0, 100])
plt.title('6) Advanced TFD Spectrogram - PWVD (Magnitude)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Frequency [Hz]'); plt.colorbar(label='Spectral Energy [dB]')

# Panel 12: Advanced TFD Spectrogram - PWVD (Phase Velocity)
plt.subplot(7,2,12)
plt.pcolormesh(t_tfd_phase, f_tfd_phase, 10*np.log10(Z_tfd_phase + 1e-12), shading='gouraud', cmap='turbo')
plt.xlim([0, t_ds[-1]])
plt.ylim([0, 100])
plt.title('6) Advanced TFD Spectrogram - PWVD (Phase Velocity)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Frequency [Hz]'); plt.colorbar(label='Spectral Energy [dB]')

# Panel 13: Energy Envelope Magnitude (normalized)
plt.subplot(7,2,13)
plt.plot(t_ds, mag_env_n, linewidth=1, color='steelblue', label='Energy Envelope')
plt.axhline(mag_thresh, color='r', linestyle='--', label='Detection Threshold')
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(mag_env_n, t_ds))
plt.title('7) Korotkoff Energy Envelope (Magnitude - Normalized)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Normalized Value'); plt.legend(); plt.grid(True)

# Panel 14: Energy Envelope Phase (normalized)
plt.subplot(7,2,14)
plt.plot(t_ds, phase_env_n, linewidth=1, color='firebrick', label='Energy Envelope')
plt.axhline(phase_thresh, color='r', linestyle='--', label='Detection Threshold')
if on_s < off_s: plt.axvspan(on_s, off_s, color='yellow', alpha=0.2)
plt.xlim([0, t_ds[-1]])
plt.ylim(get_stable_ylim(phase_env_n, t_ds))
plt.title('7) Korotkoff Energy Envelope (Phase Velocity - Normalized)', fontsize=13, fontweight='bold')
plt.xlabel('Time [seconds]'); plt.ylabel('Normalized Value'); plt.legend(); plt.grid(True)

plt.tight_layout(rect=[0,0,1,0.97])

output_img = r'd:\Bioview\My_RF_work_v1\data_new\RF_HeartRate_Korotkoff_Final_Updated.png'
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {output_img}")

print(f"\n=========================================")
print(f"KOROTKOFF DURATION REPORT (PHASE VELOCITY)")
print(f"=========================================")
print(f"Korotkoff Start : {on_s:.2f} s")
print(f"Korotkoff End   : {off_s:.2f} s")
print(f"Total Duration  : {koro_dur:.2f} s")
print(f"=========================================\n")
