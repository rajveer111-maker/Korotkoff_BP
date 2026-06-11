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

# IQ conditioning and centering
def apply_iq(i, q):
    return -i + 1j * q

def iq_condition(iq):
    ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc = (qc - sp*ic) / (al*cp + 1e-15)
    return ic + 1j*qc

iq = iq_condition(apply_iq(data[0, :], data[1, :]))

# Instantaneous phase difference
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
dphi = np.append(dphi, dphi[-1])

# Check for spikes (non-physiological jumps, e.g. when noise causes it to jump)
# Physiologically, displacement in 0.0001s is < 1 micrometer.
# 1 micrometer = 0.001 mm. Phase change for 0.001 mm is:
# dphi = 4 * pi * 0.001 / 333 = 0.000037 rad.
# Let's clip any spike larger than 0.01 rad (which corresponds to 0.26 mm in 0.0001s, i.e., 2.6 m/s!).
# This is extremely safe and will preserve all true physiological movements while completely removing phase jumps.
dphi_clean = np.clip(dphi, -0.01, 0.01)

# Convert to velocity in mm/s
vel_mm_s = dphi_clean * FS_RF * SCALE

# Filter to Korotkoff band (10-50 Hz)
sos_koro = butter(4, [10, 50], btype='band', fs=FS_RF, output='sos')
vel_koro = sosfiltfilt(sos_koro, vel_mm_s)

# Regions (Active: 12.15s - 17.95s, Noise: 5s - 10s)
# Since we trimmed 5s, we subtract 5s from the times:
idx_active_start = int((12.15 - 5.0) * FS_RF)
idx_active_end   = int((17.95 - 5.0) * FS_RF)
idx_noise_start  = 0
idx_noise_end    = int(5.0 * FS_RF)

active_vel = vel_koro[idx_active_start:idx_active_end]
noise_vel  = vel_koro[idx_noise_start:idx_noise_end]

rms_active = np.sqrt(np.mean(active_vel**2))
rms_noise  = np.sqrt(np.mean(noise_vel**2))
snr = 20 * np.log10(rms_active / rms_noise)

print(f"Clean Velocity Range: [{np.min(vel_koro):.6f}, {np.max(vel_koro):.6f}] mm/s")
print(f"Active RMS: {rms_active:.6f} mm/s")
print(f"Noise RMS: {rms_noise:.6f} mm/s")
print(f"SNR: {snr:.2f} dB")
