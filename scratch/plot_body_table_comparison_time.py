import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
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

def bandpass_filter_sos(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

all_files = [
    ('Table 2', 'ultra_rftable2.h5'),
    ('Body 2', 'ultra_rfbody1.h5'),
    ('Table 3', 'ultra_rftable3.h5'),
    ('Body 3', 'ultra_rfbody2.h5'),
    ('Table 4', 'ultra_rftable4.h5'),
    ('Body 4', 'ultra_rfbody3.h5'),
]

fig, axes = plt.subplots(6, 2, figsize=(15, 18), facecolor='white')

for idx, (name, filename) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        # Load active segment 2s to 10s
        data = f['data'][:, 2*FS : 10*FS]
    
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    mag = np.sqrt(i_c**2 + q_c**2)
    phase = robust_phase(i_c, q_c)
    disp_um = phase * 26510
    
    # Bandpass filter (0.5 to 3.0 Hz) to isolate heartbeat
    disp_filt = bandpass_filter_sos(disp_um, 0.5, 3.0, FS)
    mag_filt = bandpass_filter_sos(mag, 0.5, 3.0, FS)
    
    t = np.arange(len(disp_filt)) / FS + 2.0
    
    # Plot Phase
    axes[idx, 0].plot(t[::10], disp_filt[::10], color='teal', linewidth=1.5)
    axes[idx, 0].set_title(f"{name} - Filtered Phase (0.5-3 Hz)", fontsize=11, weight='bold')
    axes[idx, 0].set_ylabel("Disp. (µm)")
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Plot Magnitude
    axes[idx, 1].plot(t[::10], mag_filt[::10], color='purple', linewidth=1.5)
    axes[idx, 1].set_title(f"{name} - Filtered Magnitude (0.5-3 Hz)", fontsize=11, weight='bold')
    axes[idx, 1].set_ylabel("Mag (V)")
    axes[idx, 1].grid(True, alpha=0.3)

# Last row labels
axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")

plt.suptitle("Ultra Modulated RF: Body vs Table Active Segment (2s - 10s)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_body_table_filtered_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_body_table_filtered_comparison.png")
