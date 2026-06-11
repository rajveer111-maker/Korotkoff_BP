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

def iq_condition_keep_dc(iq):
    # Calculate DC offsets
    i_mean, q_mean = iq.real.mean(), iq.imag.mean()
    
    # Center for parameter estimation
    ic, qc = iq.real - i_mean, iq.imag - q_mean
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc_corr = (qc - sp*ic) / (al*cp + 1e-15)
    else:
        qc_corr = qc
        
    # Restore DC offsets!
    return (ic + i_mean) + 1j * (qc_corr + q_mean)

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
iq_raw = apply_iq(i_raw, q_raw)

# Condition and restore DC
iq_corr = iq_condition_keep_dc(iq_raw)

# Truncate to deflation period (t >= 20.0s)
start_sample = int(20.0 * FS_RF)
iq_corr_defl = iq_corr[start_sample:]

# Unwrap
phase_corr = robust_phase_unwrap_v5(iq_corr_defl)
disp_corr_mm = phase_corr * SCALE

# Filter heart rate pulse [0.7, 2.5] Hz
sos_hr = butter(4, [0.7, 2.5], btype='bandpass', fs=FS_RF, output='sos')
hr_sig_corr = sosfiltfilt(sos_hr, disp_corr_mm)

print(f"IQ-Corrected Keep-DC Displacement: range [{np.min(disp_corr_mm):.4f}, {np.max(disp_corr_mm):.4f}] mm")
print(f"IQ-Corrected Keep-DC Filtered Heart Rate Pulse (hr_sig): range [{np.min(hr_sig_corr):.6f}, {np.max(hr_sig_corr):.6f}] mm")
