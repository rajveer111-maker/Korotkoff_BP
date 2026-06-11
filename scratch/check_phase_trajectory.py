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
iq = iq_condition(apply_iq(i_raw, q_raw))

standard_phase = np.unwrap(np.angle(iq))
N = len(standard_phase)

print("Standard unwrapped phase trajectory (at 10% steps):")
for pct in range(0, 101, 10):
    idx = min(int(N * pct / 100), N - 1)
    print(f"  {pct}%: sample {idx} -> {standard_phase[idx]:.2f} rad")
