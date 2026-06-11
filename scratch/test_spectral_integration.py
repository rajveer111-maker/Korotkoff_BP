import h5py, numpy as np, os
from scipy import signal

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

# Wrap-free velocity (dphi * fs * SCALE)
vel = np.angle(iq[1:] * np.conj(iq[:-1])) * FS_RF * SCALE
vel = np.append(vel, vel[-1])  # maintain length

# Remove mean
vel = vel - np.mean(vel)

# Perform FFT
N = len(vel)
V = np.fft.rfft(vel)
freqs = np.fft.rfftfreq(N, 1/FS_RF)

# Frequency-domain integration: X(f) = V(f) / (2 * pi * j * f)
# Limit integration to the physiological band [0.15, 3.0] Hz to avoid low-frequency noise blowup
X = np.zeros_like(V, dtype=complex)
mask = (freqs >= 0.15) & (freqs <= 3.0)
X[mask] = V[mask] / (2 * np.pi * 1j * freqs[mask])

# Inverse FFT to get displacement
disp_reconstructed = np.fft.irfft(X, n=N)

print(f"Spectral Integrated Displacement: range [{np.min(disp_reconstructed):.6f}, {np.max(disp_reconstructed):.6f}] mm")
