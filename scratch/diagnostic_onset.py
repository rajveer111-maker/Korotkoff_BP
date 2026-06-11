import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
h5_path = os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5")

with h5py.File(h5_path, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = data[0], data[1]
t = np.arange(len(i_raw)) / 10000.0

# Let's compute raw signal energy or differential phase to see the inflation phase
iq = -i_raw + 1j * q_raw
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))

print(f"Data length: {len(t)} samples, {t[-1]:.2f} seconds")
print(f"Mean dphi: {np.mean(dphi):.6f}, Std dphi: {np.std(dphi):.6f}")

# Let's compute a running variance of dphi to find the high-frequency/high-amplitude pumping phase
# Pumping has massive phase changes (high variance of dphi), while holding/deflation has very low/stable variance.
w_size = 5000 # 0.5 seconds
dphi_var = np.convolve(dphi**2, np.ones(w_size)/w_size, mode='same')

# Let's find where the massive phase variance of inflation drops below a threshold
# Or let's see how dphi looks in different windows
for sec in [5, 10, 15, 20, 25, 30]:
    idx = int(sec * 10000)
    print(f"At {sec}s: local mean dphi: {np.mean(dphi[idx:idx+1000]):.6f}, local std: {np.std(dphi[idx:idx+100]):.6f}")
