import h5py, numpy as np, os

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
FS_RF   = 10_000

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

trim = int(5 * FS_RF)
if data.shape[1] > 2 * trim: data = data[:, trim:-trim]

# Raw IQ without any centering or mean subtraction
iq = -data[0, :] + 1j * data[1, :]

# Phase difference
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))

print("Raw dphi stats:")
print(f"  Min: {np.min(dphi):.6f}")
print(f"  Max: {np.max(dphi):.6f}")
print(f"  Mean: {np.mean(dphi):.6f}")
print(f"  Std: {np.std(dphi):.6f}")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(dphi, p):.6f}")
