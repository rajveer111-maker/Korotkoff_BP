import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    stft,
    find_peaks,
    decimate,
    hilbert
)

# ============================================================
# USER PARAMETERS
# ============================================================
H5_FILE = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
FS = 10000     # Original Sampling Frequency [Hz]
HR_LOW, HR_HIGH = 0.8, 3.0
KORO_LOW, KORO_HIGH = 8, 50
DEC = 100
FS_HR = FS / DEC

# ============================================================
# FILTER FUNCTIONS
# ============================================================
def bandpass_filter(x, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)

# ============================================================
# LOAD DATA
# ============================================================
with h5py.File(H5_FILE, 'r') as f:
    data = np.array(f['data'])
    if data.shape[0] == 2:
        I, Q = data[0, :], data[1, :]
    else:
        I, Q = data[:, 0], data[:, 1]

# ============================================================
# PRE-PROCESSING (RMG STYLE)
# ============================================================
# IQ Centering to prevent large phase accumulation
I_c, Q_c = I - np.mean(I), Q - np.mean(Q)
iq = I_c + 1j * Q_c
magnitude = np.abs(iq)
phase = np.unwrap(np.angle(iq))
phase_detrend = detrend(phase)

# PHASE DERIVATIVE (Better for transients/snaps)
velocity = np.diff(phase_detrend) * FS
velocity = np.append(velocity, velocity[-1])

# ============================================================
# HEART RATE (DOWNSAMPLED)
# ============================================================
phase_ds = decimate(phase_detrend, DEC, ftype='fir')
magnitude_ds = decimate(magnitude - np.mean(magnitude), DEC, ftype='fir')
t_ds = np.arange(len(phase_ds)) / FS_HR

# HR Bandpass
phase_hr = bandpass_filter(phase_ds, HR_LOW, HR_HIGH, FS_HR)
mag_hr = bandpass_filter(magnitude_ds, HR_LOW, HR_HIGH, FS_HR)

# HR Peaks
min_dist = int(FS_HR * 0.4)
peaks_p, _ = find_peaks(phase_hr, distance=min_dist, prominence=np.std(phase_hr)*0.4)
bpm_p = 60 / (np.mean(np.diff(peaks_p)) / FS_HR) if len(peaks_p) > 1 else 0

# ============================================================
# KOROTKOFF (HIGH RESOLUTION)
# ============================================================
# Filter the velocity signal for Koro
koro_sig = bandpass_filter(velocity, KORO_LOW, KORO_HIGH, FS)
koro_env = np.abs(hilbert(koro_sig))

# ============================================================
# IMPROVED TFD (STFT Tuning)
# ============================================================
# Using a smaller window (512 instead of 4096) for transient capture
f_s, t_s, Zxx = stft(koro_sig, fs=FS, nperseg=512, noverlap=256)
Sxx = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)

# ============================================================
# PLOTTING
# ============================================================
plt.figure(figsize=(20, 24))
t = np.arange(len(I)) / FS

# 1. Raw Complex signals
plt.subplot(6, 2, 1); plt.plot(t, I_c, color='blue', alpha=0.5); plt.title('Centered I'); plt.grid(True)
plt.subplot(6, 2, 2); plt.plot(t, Q_c, color='orange', alpha=0.5); plt.title('Centered Q'); plt.grid(True)

# 2. NCS signals
plt.subplot(6, 2, 3); plt.plot(t_ds, magnitude_ds, color='green'); plt.title('NCS_am (Downsampled)'); plt.grid(True)
plt.subplot(6, 2, 4); plt.plot(t_ds, phase_ds, color='red'); plt.title('NCS_ph (Downsampled)'); plt.grid(True)

# 3. Heart Rate Extraction
plt.subplot(6, 1, 3)
plt.plot(t_ds, phase_hr, color='brown', label=f'Heart Rate: {bpm_p:.1f} BPM')
plt.plot(t_ds[peaks_p], phase_hr[peaks_p], 'ro')
plt.title('Heart Rate Analysis (0.7-3.0 Hz Band)'); plt.legend(); plt.grid(True)

# 4. Korotkoff Time Domain
plt.subplot(6, 1, 4)
plt.plot(t, koro_sig, color='purple', alpha=0.6, label='Velocity Bandpass')
plt.plot(t, koro_env, color='orange', linewidth=1, label='Energy Envelope')
plt.title('Korotkoff Transient Detection (10-50 Hz)'); plt.legend(); plt.grid(True)

# 5. IMPROVED TFD (Spectrogram)
plt.subplot(6, 1, 5)
plt.pcolormesh(t_s, f_s, Sxx, shading='gouraud', cmap='magma', vmin=np.percentile(Sxx, 50), vmax=np.percentile(Sxx, 99.9))
plt.ylim(0, 60); plt.title('High-Resolution TFD Analysis (Spectrogram)'); plt.ylabel('Freq [Hz]'); plt.colorbar(label='Power [dB]')

# 6. FFT Summary
plt.subplot(6, 1, 6)
f_fft, p_fft = signal.welch(phase_hr, FS_HR, nperseg=len(phase_hr))
plt.semilogy(f_fft, p_fft, color='black')
plt.xlim(0, 5); plt.title('HR Spectral Confirmation'); plt.xlabel('Hz'); plt.grid(True)

plt.tight_layout()
plt.savefig('RF_Analysis_Robust_Improved.png', dpi=300)
print(f"Robust analysis saved to: RF_Analysis_Robust_Improved.png")
print(f"Detected Heart Rate: {bpm_p:.2f} BPM")
