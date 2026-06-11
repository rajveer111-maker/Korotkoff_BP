import h5py, numpy as np, os

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
    return np.insert(np.cumsum(dc), 0, 0.0)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq = iq_condition(apply_iq(i_raw, q_raw))

# Unwrapped phase
phase_raw = robust_phase_unwrap(iq)

dphase = np.diff(phase_raw)
print(f"Max absolute phase difference between adjacent samples: {np.max(np.abs(dphase)):.6f} rad")
print(f"Mean of absolute phase differences: {np.mean(np.abs(dphase)):.6f} rad")
print(f"Standard deviation of phase differences: {np.std(dphase):.6f} rad")

# Check if there are any jumps larger than 0.1 rad
large_jumps = np.where(np.abs(dphase) > 0.1)[0]
print(f"Number of jumps larger than 0.1 rad: {len(large_jumps)}")
if len(large_jumps) > 0:
    print(f"Indices of first 10 large jumps:\n{large_jumps[:10]}")
    print(f"Values of first 10 large jumps:\n{dphase[large_jumps[:10]]}")
