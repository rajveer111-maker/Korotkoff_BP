import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, decimate
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'oscillometric_failure_proof.png')
FS_RF = 10000
DEC = 10
FS = FS_RF // DEC

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def process_subject(name, h5_file, sbp, map_p, dbp, k_on, k_off, defl_onset):
    rf_path = os.path.join(BASE, h5_file)
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Extract Low-Frequency Heartbeat Displacement Amplitude
    sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    rf_pulse = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
    t = np.arange(len(rf_pulse)) / FS
    
    # 4.0 second smoothing to perfectly mimic standard NIBP algorithms which look for a single broad peak
    rf_env = env_smooth(rf_pulse, 4.0, FS)
    
    # Calculate MAP time based on cuff pressure
    beta = (sbp - dbp) / (k_off - k_on)
    t_map = k_on + (sbp - map_p) / beta
    
    # Calculate false MAA peak
    mask = (t >= defl_onset + 1.0) & (t <= k_off + 2.0)
    t_false_peak = t[mask][np.argmax(rf_env[mask])]
    
    # Normalize envelope for plotting
    rf_env = rf_env / np.max(rf_env[mask])
    
    return t, rf_env, t_map, t_false_peak, k_on, k_off

subjects = [
    ('Sub 1 (Prof. Kan)', 'Sub_1_Prof_kan/Rec_6.h5', 125.0, 110.0, 75.0, 27.53, 43.33, 18.0),
    ('Sub 2 (Rajveer)', 'Sub_2_Rajveer/Rec_4.h5', 125.0, 92.0, 75.0, 27.38, 42.00, 18.6)
]

plt.rcParams.update({'font.family': 'sans-serif'})
fig, axs = plt.subplots(2, 1, figsize=(14, 10), dpi=300, facecolor='white')

for i, sub in enumerate(subjects):
    name = sub[0]
    t, env, t_map, t_false, c_kon, c_koff = process_subject(*sub)
    
    ax = axs[i]
    ax.plot(t, env, color='#880E4F', lw=3.0, label='Low-Frequency Skin Displacement (0.4-3.0 Hz)')
    
    # Highlight true Korotkoff window
    ax.axvspan(c_kon, c_koff, color='#27AE60', alpha=0.15, label='Clinical Korotkoff Window')
    
    # Mark True MAP
    ax.axvline(t_map, color='#27AE60', lw=2.5, ls='--')
    ax.plot(t_map, env[int(t_map*FS)], '*', color='#27AE60', ms=18, mec='white', mew=1.5, label=f'Clinical MAP Reference ({t_map:.1f}s)')
    
    # Mark False MAA Peak
    ax.axvline(t_false, color='#C0392B', lw=2.5, ls=':')
    ax.plot(t_false, 1.0, 'X', color='#C0392B', ms=14, mec='white', mew=1.5, label=f'Maximum Displacement ({t_false:.1f}s)')
    
    # Aesthetics
    ax.set_title(f"Continuous Displacement Growth Phenomenon: {name}", fontweight='bold', fontsize=16)
    ax.set_xlim([15, 48])
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Normalized Amplitude (a.u.)", fontsize=12, fontweight='bold')
    if i == 1:
        ax.set_xlabel("Time (Sec.)", fontsize=14, fontweight='bold')
    
    ax.grid(True, ls='-', color='#E5E5E5', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add text box explaining the failure
    props = dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9, edgecolor='#1ABC9C')
    ax.text(15.5, 0.45, f"Physical Phenomenon:\nUnlike air pressure, physical skin\ndisplacement continuously grows\nas the cuff deflates. This requires\nHigh-Frequency Korotkoff extraction\nfor accurate BP estimation.", fontsize=11, fontweight='bold', color='#117A65', bbox=props, va='center')

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved proof to {OUT}")
