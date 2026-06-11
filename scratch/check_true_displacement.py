import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
FS_RF   = 10_000
FC_HZ   = 0.9e9
C       = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE   = LAMBDA_MM / (4 * np.pi)

def apply_iq(i, q):
    return -i + 1j * q

def robust_phase_unwrap_v5(iq):
    phase_unwrap = np.unwrap(np.angle(iq))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.median(dphi)
    dphi_clean = dphi - carrier_offset
    dphi_clean = np.clip(dphi_clean, -0.5, 0.5)
    phase_clean = np.insert(np.cumsum(dphi_clean), 0, 0.0)
    return signal.detrend(phase_clean)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]

# Get raw IQ directly (origin is far away, so it represents the true center of the RF circle)
iq_raw = apply_iq(i_raw, q_raw)

# Truncate to deflation period (t >= 20.0s)
start_sample = int(20.0 * FS_RF)
iq_raw_defl = iq_raw[start_sample:]

# Unwrap the raw IQ (WITHOUT AVERAGE CENTERING!)
phase_raw = robust_phase_unwrap_v5(iq_raw_defl)
disp_raw_mm = phase_raw * SCALE

# Filter heart rate pulse [0.7, 2.5] Hz
sos_hr = butter(4, [0.7, 2.5], btype='bandpass', fs=FS_RF, output='sos')
hr_sig_raw = sosfiltfilt(sos_hr, disp_raw_mm)

print(f"RAW (No-Center) Displacement: range [{np.min(disp_raw_mm):.4f}, {np.max(disp_raw_mm):.4f}] mm")
print(f"RAW (No-Center) Filtered Heart Rate Pulse (hr_sig): range [{np.min(hr_sig_raw):.6f}, {np.max(hr_sig_raw):.6f}] mm")
