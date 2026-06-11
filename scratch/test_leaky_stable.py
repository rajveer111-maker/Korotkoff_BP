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

# Calculate phase differences
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))

# Bandpass filter dphi in [0.7, 2.5] Hz (Heart Rate band)
sos_bp = butter(4, [0.7, 2.5], btype='bandpass', fs=FS_RF, output='sos')
dphi_filtered = sosfiltfilt(sos_bp, dphi)

# Leaky integration: y[n] = alpha * y[n-1] + x[n]
alpha = 0.999  # Time constant of 0.1 seconds at 10 kHz
phase_leaky = np.zeros_like(dphi_filtered)
current = 0.0
for i in range(len(dphi_filtered)):
    current = alpha * current + dphi_filtered[i]
    phase_leaky[i] = current

# Physical conversion
disp_mm = phase_leaky * SCALE

print(f"Leaky Integrated Phase Range: [{np.min(phase_leaky):.6f}, {np.max(phase_leaky):.6f}] rad")
print(f"Leaky Integrated Heartbeat Displacement: [{np.min(disp_mm):.6f}, {np.max(disp_mm):.6f}] mm")
