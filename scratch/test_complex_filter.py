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

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq = iq_condition(apply_iq(i_raw, q_raw))

# Bandpass filter the complex IQ signal in the physiological band [0.2, 4.0] Hz
sos_bp = butter(4, [0.2, 4.0], btype='bandpass', fs=FS_RF, output='sos')
iq_filtered = sosfiltfilt(sos_bp, iq)

# Extract phase from the filtered IQ signal
phase_filtered = np.unwrap(np.angle(iq_filtered))

# Detrend
phase_clean = signal.detrend(phase_filtered)

# Physical conversion
disp_mm = phase_clean * SCALE

print(f"Complex Filtered Phase Range: [{np.min(phase_clean):.4f}, {np.max(phase_clean):.4f}] rad")
print(f"Complex Filtered Displacement: [{np.min(disp_mm):.4f}, {np.max(disp_mm):.4f}] mm")
