import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
files = [f for f in os.listdir(ultra_dir) if f.endswith('.h5')]

print("Comparing files:")
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        f1_path = os.path.join(ultra_dir, files[i])
        f2_path = os.path.join(ultra_dir, files[j])
        
        with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
            d1 = f1['data'][:]
            d2 = f2['data'][:]
            
            if d1.shape == d2.shape:
                are_identical = np.array_equal(d1, d2)
                mean_diff = np.mean(np.abs(d1 - d2))
                print(f"  {files[i]} vs {files[j]}: Identical={are_identical}, Mean Diff={mean_diff}")
            else:
                print(f"  {files[i]} vs {files[j]}: Shapes differ ({d1.shape} vs {d2.shape})")
