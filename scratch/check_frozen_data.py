import h5py, numpy as np, os

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
N = len(i_raw)

# Check variance in different segments
segments = [
    ("0% - 10%", 0, 59110),
    ("10% - 20%", 59110, 118220),
    ("20% - 30%", 118220, 177330),
    ("50% - 60%", 295550, 354660),
    ("90% - 100%", 531990, 591100)
]

for name, start, end in segments:
    i_seg = i_raw[start:end]
    q_seg = q_raw[start:end]
    print(f"Segment {name} (samples {start} to {end}):")
    print(f"  I: mean={np.mean(i_seg):.4f}, std={np.std(i_seg):.4e}, min={np.min(i_seg):.4f}, max={np.max(i_seg):.4f}")
    print(f"  Q: mean={np.mean(q_seg):.4f}, std={np.std(q_seg):.4e}, min={np.min(q_seg):.4f}, max={np.max(q_seg):.4f}")
