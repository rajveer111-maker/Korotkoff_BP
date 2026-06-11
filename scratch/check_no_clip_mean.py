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

def iq_condition(iq):
    ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc = (qc - sp*ic) / (al*cp + 1e-15)
    return ic + 1j*qc

def robust_phase_unwrap_no_clip(iq):
    phase_unwrap = np.unwrap(np.angle(iq))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.mean(dphi)
    dphi_clean = dphi - carrier_offset
    phase_clean = np.insert(np.cumsum(dphi_clean), 0, 0.0)
    return signal.detrend(phase_clean)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq = iq_condition(apply_iq(i_raw, q_raw))

phase_clean = robust_phase_unwrap_no_clip(iq)
displacement_mm = phase_clean * SCALE

# Filter heart rate pulse [0.7, 2.5] Hz
sos_hr = butter(4, [0.7, 2.5], btype='bandpass', fs=FS_RF, output='sos')
hr_sig = sosfiltfilt(sos_hr, displacement_mm)

print(f"No-Clip Mean Displacement: range [{np.min(displacement_mm):.4f}, {np.max(displacement_mm):.4f}] mm")
print(f"No-Clip Mean Filtered Heart Rate Pulse (hr_sig): range [{np.min(hr_sig):.6f}, {np.max(hr_sig):.6f}] mm")
