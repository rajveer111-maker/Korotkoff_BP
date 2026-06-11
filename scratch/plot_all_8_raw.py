import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    return np.insert(np.cumsum(dp), 0, 0.0)

all_files = [
    ('Table 1', 'ultra_rftable1.h5'),
    ('Body 1', 'ultra_rfbody01.h5'),
    ('Table 2', 'ultra_rftable2.h5'),
    ('Body 2', 'ultra_rfbody1.h5'),
    ('Table 3', 'ultra_rftable3.h5'),
    ('Body 3', 'ultra_rfbody2.h5'),
    ('Table 4', 'ultra_rftable4.h5'),
    ('Body 4', 'ultra_rfbody3.h5'),
]

fig, axes = plt.subplots(8, 2, figsize=(15, 20), facecolor='white')

for i, (name, filename) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        # Decimate to speed up
        data = f['data'][:, ::100]
    
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    mag = np.sqrt(i_c**2 + q_c**2)
    phase = robust_phase(i_c, q_c)
    
    t = np.arange(len(mag)) * (100 / FS)
    
    # Plot Magnitude
    axes[i, 0].plot(t, mag, color='purple', linewidth=1.2)
    axes[i, 0].set_title(f"{name} - Magnitude", fontsize=10, weight='bold')
    axes[i, 0].set_ylabel("Mag (V)")
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot Phase
    axes[i, 1].plot(t, phase, color='teal', linewidth=1.2)
    axes[i, 1].set_title(f"{name} - Unwrapped Phase", fontsize=10, weight='bold')
    axes[i, 1].set_ylabel("Phase (rad)")
    axes[i, 1].grid(True, alpha=0.3)

# Last row labels
axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")

plt.suptitle("Raw Signals comparison: All 8 Recordings", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_all_8_raw_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_all_8_raw_comparison.png")
