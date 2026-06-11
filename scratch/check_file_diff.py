import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
f1 = os.path.join(ultra_dir, 'ultra_rfbody1.h5')
f2 = os.path.join(ultra_dir, 'ultra_rftable2.h5')

with h5py.File(f1, 'r') as f_b:
    d_b = f_b['data'][:, :40000]

with h5py.File(f2, 'r') as f_t:
    d_t = f_t['data'][:, :40000]

diff = np.sum(np.abs(d_b - d_t))
print("Diff between first 40000 samples:", diff)
print("Body shape:", d_b.shape, "Table shape:", d_t.shape)
print("Body sample mean:", np.mean(d_b), "Table sample mean:", np.mean(d_t))
