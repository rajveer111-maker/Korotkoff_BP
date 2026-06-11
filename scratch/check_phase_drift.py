import h5py, numpy as np, os
from scipy import signal

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'

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
    return signal.detrend(np.insert(np.cumsum(dc), 0, 0.0))

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq = iq_condition(apply_iq(i_raw, q_raw))

# Standard NumPy unwrap
standard_phase = np.unwrap(np.angle(iq))

# Robust unwrap
robust_phase = robust_phase_unwrap(iq)

print(f"Standard unwrapped phase: range [{np.min(standard_phase):.2f}, {np.max(standard_phase):.2f}] rad, span: {np.max(standard_phase)-np.min(standard_phase):.2f} rad")
print(f"Robust unwrapped phase  : range [{np.min(robust_phase):.2f}, {np.max(robust_phase):.2f}] rad, span: {np.max(robust_phase)-np.min(robust_phase):.2f} rad")

# Check if there is a frequency offset
dphi_raw = np.angle(iq[1:] * np.conj(iq[:-1]))
print(f"Mean dphi: {np.mean(dphi_raw):.6f} rad/sample")
print(f"Median dphi: {np.median(dphi_raw):.6f} rad/sample")
