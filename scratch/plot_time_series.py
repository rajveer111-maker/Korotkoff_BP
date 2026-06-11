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

# We will load:
# Body 1, Body 2, Body 3
# Table 1, Table 2

files_to_plot = {
    'Body 1': 'ultra_rfbody1.h5',
    'Body 2': 'ultra_rfbody2.h5',
    'Body 3': 'ultra_rfbody3.h5',
    'Table 1': 'ultra_rftable1.h5',
    'Table 2': 'ultra_rftable2.h5',
}

fig, axes = plt.subplots(len(files_to_plot), 1, figsize=(12, 10), sharex=True, facecolor='white')

# 10 second window
start_sample = 20 * FS
end_sample = 30 * FS
t = np.arange(start_sample, end_sample) / FS

for i, (name, filename) in enumerate(files_to_plot.items()):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, start_sample:end_sample]
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    phase = robust_phase(i_c, q_c)
    # Detrend
    phase_detrend = phase - np.polyval(np.polyfit(np.arange(len(phase)), phase, 1), np.arange(len(phase)))
    
    # Scale phase to displacement (in micrometers)
    # Wavelength at 0.9 GHz is ~333.1 mm, so scaling is 26.51 mm/rad = 26510 micrometers/rad
    disp_um = phase_detrend * 26510
    
    color = 'red' if 'Body' in name else 'blue'
    axes[i].plot(t, disp_um, color=color, linewidth=1.5)
    axes[i].set_title(f"{name} ({filename}) - Phase Displacement", fontsize=11, weight='bold')
    axes[i].set_ylabel("Disp. (µm)")
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_time_series_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_time_series_comparison.png")
