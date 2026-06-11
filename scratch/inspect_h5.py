import h5py
import numpy as np
import os

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

with h5py.File(file_path, 'r') as f:
    print(f"Keys: {list(f.keys())}")
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"Dataset '{key}' shape: {f[key].shape}, dtype: {f[key].dtype}")
        else:
            print(f"Group '{key}'")
    
    # Check attributes
    print(f"Attributes: {list(f.attrs.keys())}")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")
