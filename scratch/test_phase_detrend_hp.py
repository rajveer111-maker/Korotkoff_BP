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

def robust_phase_unwrap(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, b = np.histogram(dphi, 512)
    co = b[np.argmax(h)] + (b[1]-b[0])/2
    dc = dphi - co
    iqr = np.percentile(dc, 75) - np.percentile(dc, 25)
    dc = np.clip(dc, -max(3*iqr, 0.017), max(3*iqr, 0.017))
    return np.insert(np.cumsum(dc), 0, 0.0)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq = iq_condition(apply_iq(i_raw, q_raw))

# Unwrapped phase
phase_raw = robust_phase_unwrap(iq)

# 1. Detrend first to completely remove the linear slope
phase_detrended = signal.detrend(phase_raw)
print(f"Detrended phase range: [{np.min(phase_detrended):.2f}, {np.max(phase_detrended):.2f}] rad")

# 2. Now apply the 0.5 Hz highpass filter to remove remaining low frequency drifts
sos_hp = butter(4, 0.5, btype='highpass', fs=FS_RF, output='sos')
phase_clean = sosfiltfilt(sos_hp, phase_detrended)

# Physical conversion
disp_clean_mm = phase_clean * SCALE
print(f"Detrended + 0.5 Hz HP Filtered Displacement: range [{np.min(disp_clean_mm):.4f}, {np.max(disp_clean_mm):.4f}] mm")
