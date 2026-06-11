import h5py
import os

filepath = r"d:\Bioview\My_RF_work_v1\data_new\Ultra\ultra_rfbody1.h5"
with h5py.File(filepath, 'r') as f:
    print("Keys in HDF5 file:")
    print(list(f.keys()))
    for key in f.keys():
        print(f"Shape of {key}: {f[key].shape}")
        print(f"Dtype of {key}: {f[key].dtype}")
        print(f"Attributes of {key}:")
        for attr in f[key].attrs:
            print(f"  {attr}: {f[key].attrs[attr]}")
