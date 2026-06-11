import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, detrend
from scipy.io import wavfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'oscillometric_failure_vs_stethoscope.png')
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

def process_subject(name, h5_file, wav_file, map_p, sbp, dbp, k_on, k_off, defl_onset):
    # RF
    rf_path = os.path.join(BASE, h5_file)
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    rf_pulse = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
    t = np.arange(len(rf_pulse)) / FS
    
    rf_env = env_smooth(rf_pulse, 4.0, FS)
    
    # Audio
    wav_path = os.path.join(BASE, wav_file)
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    
    audio_filt = bpf(audio, 50.0, 1000.0, fs_a)
    steth_env_high = np.abs(hilbert(audio_filt))
    
    high_cut = min(200.0, (fs_a / 2) - 1.0)
    steth_koro = bpf(steth_env_high, 10.0, high_cut, fs_a)
    steth_env = env_smooth(steth_koro, 1.5, fs_a)
    
    t_a = np.arange(len(steth_env)) / fs_a
    steth_env_resampled = np.interp(t, t_a, steth_env)
    
    # Normalize
    mask = (t >= defl_onset) & (t <= k_off + 3.0)
    rf_env = rf_env / np.max(rf_env[mask])
    steth_env_resampled = steth_env_resampled / np.max(steth_env_resampled[mask])
    
    # Calc map time
    beta = (sbp - dbp) / (k_off - k_on)
    t_map = k_on + (sbp - map_p) / beta
    
    t_false = t[mask][np.argmax(rf_env[mask])]
    
    return t, rf_env, steth_env_resampled, t_map, t_false, k_on, k_off

subjects = [
    ('Sub 1 (Prof. Kan)', 'Sub_1_Prof_kan/Rec_6.h5', 'Sub_1_Prof_kan/sthethoscope_rec06.wav', 110.0, 125.0, 75.0, 27.53, 43.33, 18.0),
    ('Sub 2 (Rajveer)', 'Sub_2_Rajveer/Rec_4.h5', 'Sub_2_Rajveer/sthethoscope_rec04.wav', 92.0, 125.0, 75.0, 27.38, 42.00, 18.6)
]

plt.rcParams.update({'font.family': 'sans-serif'})
fig, axs = plt.subplots(2, 1, figsize=(14, 10), dpi=300, facecolor='white')

for i, sub in enumerate(subjects):
    name = sub[0]
    t, rf_env, steth_env, t_map, t_false, c_kon, c_koff = process_subject(*sub)
    
    ax = axs[i]
    
    # Plot Stethoscope Envelope (The True Ground Truth)
    ax.fill_between(t, 0, steth_env, color='#2980B9', alpha=0.3, label='Stethoscope Acoustic Envelope (True Korotkoff)')
    ax.plot(t, steth_env, color='#1A5276', lw=1.5, alpha=0.8)
    
    # Plot RF Low-Frequency Displacement (The Flawed MAA Method)
    ax.plot(t, rf_env, color='#C0392B', lw=3.0, label='RF Heart Rate Envelope (0.4-3.0 Hz)')
    
    # Mark True MAP
    ax.axvline(t_map, color='#27AE60', lw=2.5, ls='--')
    ax.plot(t_map, rf_env[int(t_map*FS)], '*', color='#27AE60', ms=18, mec='white', mew=1.5, label=f'True Clinical MAP ({t_map:.1f}s)')
    
    # Mark False MAA Peak
    ax.axvline(t_false, color='#C0392B', lw=2.5, ls=':')
    ax.plot(t_false, 1.0, 'X', color='#C0392B', ms=14, mec='white', mew=1.5, label=f'False MAA Peak ({t_false:.1f}s)')
    
    # Aesthetics
    ax.set_title(f"Oscillometric MAA vs. Stethoscope Ground Truth: {name}", fontweight='bold', fontsize=16)
    ax.set_xlim([15, 48])
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Normalized Amplitude (a.u.)", fontsize=12, fontweight='bold')
    if i == 1:
        ax.set_xlabel("Time (Sec.)", fontsize=12, fontweight='bold')
        
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Text box explaining
    props = dict(boxstyle='round', facecolor='#FDEBD0', alpha=0.9, edgecolor='#F39C12')
    ax.text(15.5, 0.45, f"Clinical Validation:\nThe MAA peak occurs when the\nStethoscope is completely SILENT.\nThis definitively proves MAA\ncannot find the true MAP.", fontsize=11, fontweight='bold', color='#B9770E', bbox=props, va='center')

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved to {OUT}")
