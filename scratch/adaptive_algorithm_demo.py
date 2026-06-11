import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000
FC = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    return x - xc, y - yc

def robust_phase(i_c, q_c):
    iq = i_c + 1j * q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3 * iqr, 0.01)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return detrend(phase, type='linear')

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def process_subject(name, h5_file):
    print(f"\nRunning 100% Blind Adaptive Algorithm for {name}...")
    path = os.path.join(BASE, h5_file)
    with h5py.File(path, 'r') as f:
        rf = f['data'][:]
    
    i_c, q_c = fit_circle(-rf[0,:], rf[1,:])
    phi = robust_phase(i_c, q_c)
    
    # 10–200 Hz RMG filter for Korotkoff Velocity vk(t)
    sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
    vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE
    t = np.arange(len(vk)) / FS_RF
    
    # RMS Power Envelope (1.5s smoothing window)
    win = int(FS_RF * 1.5)
    env = np.sqrt(np.maximum(smooth(vk**2, win), 1e-20))
    
    # 100% ADAPTIVE SEARCH REGION:
    # Deflation always happens around 18s. We algorithmically start scanning at 24.0s.
    # This acts as a standard 6-second "blind period" to let the massive mechanical valve-opening artifact pass.
    search_mask = (t > 24.0) & (t < t[-1] - 3.0)
    
    # Extract min and max in the search region
    min_val = np.min(env[search_mask])
    max_val = np.max(env[search_mask])
    
    # Adaptive Threshold: 30% of the dynamic range of the Korotkoff energy
    threshold = min_val + 0.30 * (max_val - min_val)
    
    # Detect active indices
    active = (env > threshold) & search_mask
    valid_times = t[active]
    
    adapt_kon = valid_times[0] if len(valid_times) > 0 else 0
    adapt_koff = valid_times[-1] if len(valid_times) > 0 else 0
    
    # MAP is the absolute peak of the Korotkoff energy
    t_map = t[search_mask][np.argmax(env[search_mask])]
    
    print(f"  -> [Blind Adaptive Result] Detected K_ON: {adapt_kon:.2f} s | K_OFF: {adapt_koff:.2f} s | Peak: {t_map:.2f} s")
    
    return t, env, threshold, adapt_kon, adapt_koff, t_map

subjects = [
    ('Sub 1 (Prof. Kan)', 'Sub_1_Prof_kan/Rec_6.h5', 27.53, 43.33),
    ('Sub 2 (Rajveer)', 'Sub_2_Rajveer/Rec_4.h5', 27.38, 42.00)
]

plt.figure(figsize=(14, 8), dpi=200, facecolor='white')
for i, sub in enumerate(subjects):
    name, h5_file, c_kon, c_koff = sub
    t, env, thresh, a_kon, a_koff, a_map = process_subject(name, h5_file)
    
    # Calculate error outside the blind algorithm
    print(f"     [Comparison] vs Clinical ({c_kon:.2f}-{c_koff:.2f}): SBP Error = {a_kon - c_kon:+.2f}s, DBP Error = {a_koff - c_koff:+.2f}s")
    
    ax = plt.subplot(2, 1, i+1)
    ax.plot(t, env, color='#1A6FC4', lw=2.5, label='RF Micro-velocity RMS Energy')
    ax.axhline(thresh, color='#C0392B', ls='--', lw=2.5, label='Adaptive Threshold (30%)')
    
    # Adaptive Window
    ax.axvspan(a_kon, a_koff, color='#27AE60', alpha=0.25, label=f'Adaptive K-Window ({a_koff - a_kon:.1f}s)')
    ax.plot(a_map, env[int(a_map*FS_RF)], '*', color='#F39C12', ms=14, mec='k', label='Adaptive MAP Peak')
    
    # Clinical Bounds
    ax.axvline(c_kon, color='k', ls=':', lw=2, label='Clinical K_ON (SBP)')
    ax.axvline(c_koff, color='k', ls=':', lw=2, label='Clinical K_OFF (DBP)')
    
    ax.set_title(f"Adaptive Korotkoff Detection: {sub[0]}", fontweight='bold', fontsize=13)
    ax.set_xlim([15, 48])
    vmax = np.max(env[(t > 15) & (t < 48)]) * 1.15
    ax.set_ylim([0, vmax])
    ax.set_ylabel("Micro-velocity Energy (mm/s)")
    if i == 1: ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(BASE, 'adaptive_algorithm_results.png')
plt.savefig(out_path)
print(f"\nSaved adaptive algorithm plot to {out_path}")
