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

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq_raw = apply_iq(i_raw, q_raw)
iq_cond = iq_condition(iq_raw)

print(f"Raw IQ: mean magnitude: {np.mean(np.abs(iq_raw)):.4e}, std magnitude: {np.std(np.abs(iq_raw)):.4e}")
print(f"Cond IQ: mean magnitude: {np.mean(np.abs(iq_cond)):.4e}, std magnitude: {np.std(np.abs(iq_cond)):.4e}")

# Check phase range of raw and conditioned IQ
phase_raw = np.angle(iq_raw)
phase_cond = np.angle(iq_cond)
print(f"Raw phase range: [{np.min(phase_raw):.4f}, {np.max(phase_raw):.4f}] rad")
print(f"Cond phase range: [{np.min(phase_cond):.4f}, {np.max(phase_cond):.4f}] rad")

# Check first 20 phase differences of conditioned IQ
dphi = np.angle(iq_cond[1:21] * np.conj(iq_cond[:20]))
print(f"First 20 phase differences of conditioned IQ:\n{dphi}")
