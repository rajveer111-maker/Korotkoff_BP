import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
FS_RF   = 10_000

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

trim = int(5 * FS_RF)
if data.shape[1] > 2 * trim: data = data[:, trim:-trim]

# Raw I and Q signals (without mean subtraction or conditioning!)
I = data[0, :]
Q = data[1, :]

# Bandpass filter both channels in the Korotkoff band (10-50 Hz)
sos_koro = butter(4, [10, 50], btype='bandpass', fs=FS_RF, output='sos')
I_hf = sosfiltfilt(sos_koro, I)
Q_hf = sosfiltfilt(sos_koro, Q)

# Combine using the direct high-frequency energy formula
energy_hf = I_hf**2 + Q_hf**2

# Regions (Active: 12.15s - 17.95s, Noise: 5s - 10s)
# trimmed 5s:
idx_active_start = int((12.15 - 5.0) * FS_RF)
idx_active_end   = int((17.95 - 5.0) * FS_RF)
idx_noise_start  = 0
idx_noise_end    = int(5.0 * FS_RF)

active_energy = energy_hf[idx_active_start:idx_active_end]
noise_energy  = energy_hf[idx_noise_start:idx_noise_end]

rms_active = np.sqrt(np.mean(active_energy))
rms_noise  = np.sqrt(np.mean(noise_energy))
snr = 20 * np.log10(rms_active / rms_noise)

print("Direct IQ High-Frequency Energy Method:")
print(f"  Active RMS Energy: {rms_active:.6f}")
print(f"  Noise RMS Energy: {rms_noise:.6f}")
print(f"  SNR: {snr:.2f} dB")
