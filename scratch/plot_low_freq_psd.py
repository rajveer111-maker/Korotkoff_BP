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

all_files = [
    ('Table 1', 'ultra_rftable1.h5', 5.0, 25.0, 0.0, 5.0),
    ('Body 1', 'ultra_rfbody01.h5', 28.0, 48.0, 2.0, 25.0),
    ('Table 2', 'ultra_rftable2.h5', 2.0, 10.0, 12.0, 30.0),
    ('Body 2', 'ultra_rfbody1.h5', 2.0, 10.0, 12.0, 30.0),
    ('Table 3', 'ultra_rftable3.h5', 2.0, 8.0, 10.0, 28.0),
    ('Body 3', 'ultra_rfbody2.h5', 38.0, 41.0, 2.0, 25.0),  # Body 3 has spike at 40s
    ('Table 4', 'ultra_rftable4.h5', 2.0, 8.0, 10.0, 28.0),
    ('Body 4', 'ultra_rfbody3.h5', 2.0, 10.0, 12.0, 30.0),
]

fig, axes = plt.subplots(8, 2, figsize=(15, 20), facecolor='white')

for i, (name, filename, act_start, act_end, q_start, q_end) in enumerate(all_files):
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        # Load entire signal to do circle fit
        data_all = f['data'][:, ::10] # decimate by 10 to speed up
    
    i_raw, q_raw = data_all[0, :], data_all[1, :]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Active Segment PSD
    idx_act = (np.arange(len(i_raw)) * (10 / FS) >= act_start) & (np.arange(len(i_raw)) * (10 / FS) <= act_end)
    if np.sum(idx_act) > 1000:
        act_phase = robust_phase(i_c[idx_act], q_c[idx_act])
        f_ap, p_ap = welch(act_phase * 26510, fs=FS/10, nperseg=min(len(act_phase), 2048))
        axes[i, 0].semilogy(f_ap, p_ap, color='teal', label='Active Phase')
        
        act_mag = np.sqrt(i_c[idx_act]**2 + q_c[idx_act]**2)
        f_am, p_am = welch(act_mag, fs=FS/10, nperseg=min(len(act_mag), 2048))
        axes[i, 1].semilogy(f_am, p_am, color='purple', label='Active Magnitude')

    # Quiet Segment PSD
    idx_q = (np.arange(len(i_raw)) * (10 / FS) >= q_start) & (np.arange(len(i_raw)) * (10 / FS) <= q_end)
    if np.sum(idx_q) > 1000:
        q_phase = robust_phase(i_c[idx_q], q_c[idx_q])
        f_qp, p_qp = welch(q_phase * 26510, fs=FS/10, nperseg=min(len(q_phase), 2048))
        axes[i, 0].semilogy(f_qp, p_qp, color='crimson', linestyle='--', alpha=0.7, label='Quiet Phase')
        
        q_mag = np.sqrt(i_c[idx_q]**2 + q_c[idx_q]**2)
        f_qm, p_qm = welch(q_mag, fs=FS/10, nperseg=min(len(q_mag), 2048))
        axes[i, 1].semilogy(f_qm, p_qm, color='navy', linestyle='--', alpha=0.7, label='Quiet Magnitude')
        
    axes[i, 0].set_xlim([0.1, 10.0])
    axes[i, 0].set_title(f"{name} - Phase PSD (0.1 - 10 Hz)", fontsize=10, weight='bold')
    axes[i, 0].set_ylabel("PSD (µm²/Hz)")
    axes[i, 0].legend(fontsize=8)
    axes[i, 0].grid(True, which="both", alpha=0.3)
    
    axes[i, 1].set_xlim([0.1, 10.0])
    axes[i, 1].set_title(f"{name} - Magnitude PSD (0.1 - 10 Hz)", fontsize=10, weight='bold')
    axes[i, 1].set_ylabel("PSD (V²/Hz)")
    axes[i, 1].legend(fontsize=8)
    axes[i, 1].grid(True, which="both", alpha=0.3)

# Last row labels
axes[-1, 0].set_xlabel("Frequency (Hz)")
axes[-1, 1].set_xlabel("Frequency (Hz)")

plt.suptitle("Low-Frequency PSD comparison: Active vs Quiet Segments", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_low_freq_psd_comparison.png"), dpi=200)
plt.close()
print("Saved ultra_low_freq_psd_comparison.png")
