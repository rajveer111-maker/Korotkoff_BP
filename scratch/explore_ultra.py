import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
files = [f for f in os.listdir(ultra_dir) if f.endswith('.h5')]

print("Found HDF5 files in Ultra folder:")
for filename in files:
    filepath = os.path.join(ultra_dir, filename)
    print(f"\nFile: {filename} (Size: {os.path.getsize(filepath)/1e6:.2f} MB)")
    try:
        with h5py.File(filepath, 'r') as f:
            print("  Keys:", list(f.keys()))
            for key in f.keys():
                ds = f[key]
                print(f"    Dataset '{key}': shape={ds.shape}, dtype={ds.dtype}")
                # Print some basic stats of the data
                data_preview = ds[:]
                print(f"      Min: {data_preview.min()}, Max: {data_preview.max()}, Mean: {data_preview.mean()}, Std: {data_preview.std()}")
                if data_preview.ndim > 1:
                    print(f"      First few columns: {data_preview[:, :5]}")
    except Exception as e:
        print(f"  Error reading file: {e}")
