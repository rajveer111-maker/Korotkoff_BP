import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
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

files = {
    'Body 2': 'ultra_rfbody1.h5',
    'Body 4': 'ultra_rfbody3.h5',
    'Table 2': 'ultra_rftable2.h5',
    'Table 4': 'ultra_rftable4.h5',
}

fig, axes = plt.subplots(4, 2, figsize=(15, 12), facecolor='white')

for row_idx, (name, filename) in enumerate(files.items()):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        # Load first 30 seconds
        data = f['data'][:, :30 * FS]
    
    i_raw, q_raw = data[0, :], data[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    phase = robust_phase(i_c, q_c)
    
    # 0-10s segment (Active)
    p_active = phase[:10 * FS]
    p_active_detrend = p_active - np.polyval(np.polyfit(np.arange(len(p_active)), p_active, 1), np.arange(len(p_active)))
    
    # 10-30s segment (Quiet)
    p_quiet = phase[10 * FS:]
    p_quiet_detrend = p_quiet - np.polyval(np.polyfit(np.arange(len(p_quiet)), p_quiet, 1), np.arange(len(p_quiet)))
    
    t_active = np.arange(len(p_active)) / FS
    t_quiet = np.arange(len(p_quiet)) / FS + 10.0
    
    # Plot active vs quiet time series (decimated for clarity)
    axes[row_idx, 0].plot(t_active[::10], p_active_detrend[::10] * 26510, color='crimson', label='0-10s (Active)', alpha=0.9)
    axes[row_idx, 0].plot(t_quiet[::10], p_quiet_detrend[::10] * 26510, color='navy', label='10-30s (Quiet)', alpha=0.9)
    axes[row_idx, 0].set_title(f"{name} - Time Domain Comparison", fontsize=11, weight='bold')
    axes[row_idx, 0].set_ylabel("Disp. (µm)")
    axes[row_idx, 0].grid(True, alpha=0.3)
    axes[row_idx, 0].legend()
    
    # Compute PSD for active and quiet
    f_act, psd_act = welch(p_active_detrend, fs=FS, nperseg=8192)
    f_q, psd_q = welch(p_quiet_detrend, fs=FS, nperseg=8192)
    
    # Plot PSD comparison (0.1 to 10 Hz)
    axes[row_idx, 1].semilogy(f_act, psd_act, color='crimson', label='0-10s (Active)', linewidth=1.5)
    axes[row_idx, 1].semilogy(f_q, psd_q, color='navy', label='10-30s (Quiet)', linewidth=1.5)
    axes[row_idx, 1].set_xlim([0.1, 10])
    axes[row_idx, 1].set_title(f"{name} - PSD Comparison", fontsize=11, weight='bold')
    axes[row_idx, 1].set_xlabel("Frequency (Hz)")
    axes[row_idx, 1].set_ylabel("Power")
    axes[row_idx, 1].grid(True, which='both', alpha=0.3)
    axes[row_idx, 1].legend()

plt.suptitle("Ultra Signals Analysis: Active (0-10s) vs Quiet (10s+) Phase", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_active_vs_quiet_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_active_vs_quiet_comparison.png")
