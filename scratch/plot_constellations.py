import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

# We'll plot all 8 files in a 2x4 grid
fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')

body_files = ['ultra_rfbody01.h5', 'ultra_rfbody1.h5', 'ultra_rfbody2.h5', 'ultra_rfbody3.h5']
table_files = ['ultra_rftable1.h5', 'ultra_rftable2.h5', 'ultra_rftable3.h5', 'ultra_rftable4.h5']

for col, fn in enumerate(body_files):
    filepath = os.path.join(ultra_dir, fn)
    with h5py.File(filepath, 'r') as f:
        # Decimate by 10 to speed up plotting
        data = f['data'][:, ::10]
    i_raw, q_raw = data[0, :], data[1, :]
    axes[0, col].scatter(i_raw, q_raw, s=1, alpha=0.5, color='red')
    axes[0, col].set_title(f"Body {col+1}: {fn}", fontsize=10, weight='bold')
    axes[0, col].set_xlabel("I")
    axes[0, col].set_ylabel("Q")
    axes[0, col].grid(True, alpha=0.3)
    axes[0, col].axis('equal')

for col, fn in enumerate(table_files):
    filepath = os.path.join(ultra_dir, fn)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, ::10]
    i_raw, q_raw = data[0, :], data[1, :]
    axes[1, col].scatter(i_raw, q_raw, s=1, alpha=0.5, color='blue')
    axes[1, col].set_title(f"Table {col+1}: {fn}", fontsize=10, weight='bold')
    axes[1, col].set_xlabel("I")
    axes[1, col].set_ylabel("Q")
    axes[1, col].grid(True, alpha=0.3)
    axes[1, col].axis('equal')

plt.suptitle("Raw I/Q Constellation Comparison: Body vs Table", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_all_constellations.png"), dpi=200)
plt.close()
print("Saved ultra_all_constellations.png")
