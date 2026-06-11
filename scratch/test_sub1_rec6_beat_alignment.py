import h5py
import os
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF = 10_000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE = LAMBDA_MM / (4.0 * np.pi)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

print("Loading data...")
with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c  = i_raw - xc, q_raw - yc
phi_raw = robust_phase(i_c, q_c)

# Filter Phase Velocity
b, a = signal.iirnotch(100.71, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_raw)
b, a = signal.iirnotch(201.43, 30, FS_RF)
phi_clean = signal.filtfilt(b, a, phi_clean)

# Low-pass filter for compliance pulse (0.4-3 Hz)
sos_lp = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
phi_disp = sosfiltfilt(sos_lp, phi_clean)*SCALE

# Bandpass filter for velocity (10-80 Hz) to avoid the high freq noise
sos_vk = butter(4, [10, 80], btype='band', fs=FS_RF, output='sos')
phi_vel = np.append(np.diff(sosfiltfilt(sos_vk, phi_clean))*FS_RF, 0.0)*SCALE

# Load stethoscope
from scipy.io import wavfile
fs_a, audio = wavfile.read(WAV_PATH)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

# Filter stethoscope in the clinical band (30-200 Hz)
sos_st = butter(4, [30, 200], btype='band', fs=fs_a, output='sos')
audio_filt = sosfiltfilt(sos_st, audio)

t_rf = np.arange(len(phi_raw))/FS_RF
t_st = np.arange(len(audio))/fs_a

# Zoom window: 30.0 to 34.0 s (4 seconds, ~4-5 heartbeats)
w_start, w_end = 30.0, 34.0
m_rf = (t_rf >= w_start) & (t_rf <= w_end)
m_st = (t_st >= w_start) & (t_st <= w_end)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_rf[m_rf], phi_disp[m_rf], color='#C0392B', lw=1.5)
plt.title("RMG Arterial Displacement (0.4 - 3 Hz)")
plt.ylabel("Displacement (mm)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_rf[m_rf], phi_vel[m_rf], color='#2980B9', lw=1.0)
plt.title("RMG Phase Velocity (10 - 80 Hz BPF)")
plt.ylabel("Velocity (mm/s)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_st[m_st], audio_filt[m_st], color='#27AE60', lw=1.0)
plt.title("Stethoscope Acoustic Waveform (30 - 200 Hz BPF)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid(True)

plt.tight_layout()
plt.savefig("scratch/sub1_rec6_zoom_comparison.png", dpi=300)
print("Saved zoom comparison figure to scratch/sub1_rec6_zoom_comparison.png")
