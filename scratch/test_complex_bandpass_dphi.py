import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
FS_RF   = 10_000
FC_HZ   = 0.9e9
C       = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE   = LAMBDA_MM / (4 * np.pi)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

trim = int(5 * FS_RF)
if data.shape[1] > 2 * trim: data = data[:, trim:-trim]

# Raw IQ
def apply_iq(i, q):
    return -i + 1j * q

iq = apply_iq(data[0, :], data[1, :])

# Bandpass filter the complex IQ signal in [0.5, 100] Hz
sos_complex = butter(4, [0.5, 100], btype='bandpass', fs=FS_RF, output='sos')
iq_filtered = sosfiltfilt(sos_complex, iq)

# Calculate phase differences (velocity)
dphi = np.angle(iq_filtered[1:] * np.conj(iq_filtered[:-1]))
dphi = np.append(dphi, dphi[-1])

# Convert to velocity in mm/s
vel_mm_s = dphi * FS_RF * SCALE

# Filter velocity to Korotkoff band (10-50 Hz)
sos_koro = butter(4, [10, 50], btype='bandpass', fs=FS_RF, output='sos')
vel_koro = sosfiltfilt(sos_koro, vel_mm_s)

# Regions (Active: 12.15s - 17.95s, Noise: 5s - 10s)
# trimmed 5s:
idx_active_start = int((12.15 - 5.0) * FS_RF)
idx_active_end   = int((17.95 - 5.0) * FS_RF)
idx_noise_start  = 0
idx_noise_end    = int(5.0 * FS_RF)

active_vel = vel_koro[idx_active_start:idx_active_end]
noise_vel  = vel_koro[idx_noise_start:idx_noise_end]

rms_active = np.sqrt(np.mean(active_vel**2))
rms_noise  = np.sqrt(np.mean(noise_vel**2))
snr = 20 * np.log10(rms_active / rms_noise)

print(f"Filtered dphi range: [{np.min(dphi):.6f}, {np.max(dphi):.6f}] rad")
print(f"Clean Velocity Range: [{np.min(vel_koro):.6f}, {np.max(vel_koro):.6f}] mm/s")
print(f"Active RMS: {rms_active:.6f} mm/s")
print(f"Noise RMS: {rms_noise:.6f} mm/s")
print(f"SNR: {snr:.2f} dB")
