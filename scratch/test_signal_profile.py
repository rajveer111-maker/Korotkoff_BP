import h5py
import numpy as np
import os

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_1.h5'
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = -data[0,:], data[1,:]
N = len(i_raw)
t = np.arange(N) / 10000

print(f"Data length: {t[-1]:.1f} s")
# Let's save a quick text profile of the raw I and Q signal energy to see where the pump inflation starts
step = 10000 * 5 # 5 second chunks
for i in range(0, N, step):
    seg_i = i_raw[i:i+step]
    seg_q = q_raw[i:i+step]
    std_i = np.std(seg_i)
    std_q = np.std(seg_q)
    print(f"Time {i/10000:.1f}s - {min(N, i+step)/10000:.1f}s: std(I)={std_i:.6f}, std(Q)={std_q:.6f}")
