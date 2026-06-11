import h5py, numpy as np, os

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
FS_RF   = 10_000

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

trim = int(5 * FS_RF)
if data.shape[1] > 2 * trim: data = data[:, trim:-trim]

# IQ conditioning and centering
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

iq = iq_condition(apply_iq(data[0, :], data[1, :]))
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))

print("dphi stats:")
print(f"  Min: {np.min(dphi):.6f}")
print(f"  Max: {np.max(dphi):.6f}")
print(f"  Mean: {np.mean(dphi):.6f}")
print(f"  Std: {np.std(dphi):.6f}")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(dphi, p):.6f}")
