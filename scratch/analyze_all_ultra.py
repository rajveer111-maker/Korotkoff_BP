import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"
os.makedirs(out_dir, exist_ok=True)

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

body_files = [f for f in os.listdir(ultra_dir) if 'body' in f and f.endswith('.h5')]
table_files = [f for f in os.listdir(ultra_dir) if 'table' in f and f.endswith('.h5')]

body_files.sort()
table_files.sort()

print("Body files:", body_files)
print("Table files:", table_files)

# Setup plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')

# We'll plot:
# Row 0: Phase PSD (0-10 Hz Zoom) and (0-500 Hz Broad)
# Row 1: Magnitude PSD (0-10 Hz Zoom) and (0-500 Hz Broad)

colors_body = ['#C0392B', '#E74C3C', '#D35400', '#F39C12']
colors_table = ['#2980B9', '#3498DB', '#1ABC9C', '#2ECC71']

for i, fn in enumerate(body_files):
    filepath = os.path.join(ultra_dir, fn)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    mag = np.sqrt(i_c**2 + q_c**2)
    phase = robust_phase(i_c, q_c)
    phase_detrend = phase - np.polyval(np.polyfit(np.arange(len(phase)), phase, 1), np.arange(len(phase)))
    
    # Welch Phase
    f_p, psd_p = welch(phase_detrend, fs=FS, nperseg=8192)
    # Welch Magnitude
    f_m, psd_m = welch(mag, fs=FS, nperseg=8192)
    
    # Plot Phase Zoom (0-10 Hz)
    axes[0, 0].semilogy(f_p, psd_p, color=colors_body[i], label=f'Body {i+1}')
    # Plot Phase Broad (0-500 Hz)
    axes[0, 1].semilogy(f_p, psd_p, color=colors_body[i], label=f'Body {i+1}')
    
    # Plot Mag Zoom (0-10 Hz)
    axes[1, 0].semilogy(f_m, psd_m, color=colors_body[i], label=f'Body {i+1}')
    # Plot Mag Broad (0-500 Hz)
    axes[1, 1].semilogy(f_m, psd_m, color=colors_body[i], label=f'Body {i+1}')

for i, fn in enumerate(table_files):
    filepath = os.path.join(ultra_dir, fn)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    mag = np.sqrt(i_c**2 + q_c**2)
    phase = robust_phase(i_c, q_c)
    phase_detrend = phase - np.polyval(np.polyfit(np.arange(len(phase)), phase, 1), np.arange(len(phase)))
    
    # Welch Phase
    f_p, psd_p = welch(phase_detrend, fs=FS, nperseg=8192)
    # Welch Magnitude
    f_m, psd_m = welch(mag, fs=FS, nperseg=8192)
    
    # Plot Phase Zoom (0-10 Hz)
    axes[0, 0].semilogy(f_p, psd_p, color=colors_table[i], linestyle='--', alpha=0.8, label=f'Table {i+1}')
    # Plot Phase Broad (0-500 Hz)
    axes[0, 1].semilogy(f_p, psd_p, color=colors_table[i], linestyle='--', alpha=0.8, label=f'Table {i+1}')
    
    # Plot Mag Zoom (0-10 Hz)
    axes[1, 0].semilogy(f_m, psd_m, color=colors_table[i], linestyle='--', alpha=0.8, label=f'Table {i+1}')
    # Plot Mag Broad (0-500 Hz)
    axes[1, 1].semilogy(f_m, psd_m, color=colors_table[i], linestyle='--', alpha=0.8, label=f'Table {i+1}')

# Format plots
axes[0, 0].set_xlim([0.1, 10])
axes[0, 0].set_title("Phase PSD Zoom (0.1 - 10 Hz)")
axes[0, 0].set_xlabel("Frequency (Hz)")
axes[0, 0].set_ylabel("Power")
axes[0, 0].grid(True, which='both', alpha=0.5)
axes[0, 0].legend(loc='lower left', ncol=2)

axes[0, 1].set_xlim([0.1, 500])
axes[0, 1].set_title("Phase PSD Broad (0.1 - 500 Hz)")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Power")
axes[0, 1].grid(True, which='both', alpha=0.5)
axes[0, 1].legend(loc='upper right', ncol=2)

axes[1, 0].set_xlim([0.1, 10])
axes[1, 0].set_title("Magnitude PSD Zoom (0.1 - 10 Hz)")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Power")
axes[1, 0].grid(True, which='both', alpha=0.5)
axes[1, 0].legend(loc='lower left', ncol=2)

axes[1, 1].set_xlim([0.1, 500])
axes[1, 1].set_title("Magnitude PSD Broad (0.1 - 500 Hz)")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Power")
axes[1, 1].grid(True, which='both', alpha=0.5)
axes[1, 1].legend(loc='upper right', ncol=2)

plt.suptitle("Ultra Cohort PSD Comparison: Body vs Table", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_cohort_psd_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_cohort_psd_comparison.png")
