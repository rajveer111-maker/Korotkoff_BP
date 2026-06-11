import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'cohort_rf_steth_overlay_20.png')
FS_RF = 10000; DEC = 10; FS = 1000
CP = '#C0392B' # Red RF
CS = '#2980B9' # Blue Steth

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

from scipy.signal import fftconvolve

def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -res[0]/2, -res[1]/2

fig, axes = plt.subplots(5, 4, figsize=(22, 16), dpi=200, facecolor='white')

subjects = [('Sub_1_Prof_kan', 'Sub 1 (Prof. Kan)'), ('Sub_2_Rajveer', 'Sub 2 (Rajveer)')]

for sub_idx, (sub_dir, sub_name) in enumerate(subjects):
    for rec_idx in range(1, 11):
        # Determine subplot index
        # Sub 1 goes to cols 0, 1
        # Sub 2 goes to cols 2, 3
        col = (sub_idx * 2) + ((rec_idx - 1) % 2)
        row = (rec_idx - 1) // 2
        ax = axes[row, col]
        
        rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
        
        if not os.path.exists(rf_path) or not os.path.exists(wav_path):
            ax.set_title(f"{sub_name} - Rec {rec_idx} (MISSING DATA)")
            ax.axis('off')
            continue
            
        print(f"Processing {sub_name} Rec {rec_idx}...")
        
        try:
            with h5py.File(rf_path, 'r') as f:
                rf = f['data'][:]
            i_raw, q_raw = -rf[0,:], rf[1,:]
            xc, yc = fit_circle(i_raw, q_raw)
            phi_raw = robust_phase(i_raw-xc, q_raw-yc)
            p = notch(notch(notch(phi_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
            vel = np.append(np.diff(bpf(p, 10, 200, FS_RF))*FS_RF, 0.0)
            tkeo_rf = decimate(smooth_energy(calc_tkeo(vel), 1.5, FS_RF), DEC, ftype='fir')
            t_rf = np.arange(len(tkeo_rf))/FS
            
            fs_a, audio = wavfile.read(wav_path)
            audio = audio.astype(np.float64)/32768.0
            if audio.ndim>1: audio = audio.mean(axis=1)
            audio_f = bpf(audio, 50, 1000, fs_a)
            steth_k = bpf(np.abs(audio_f), 20, min(200, (fs_a/2)-1), fs_a)
            steth_tkeo_r = smooth_energy(calc_tkeo(steth_k), 1.5, fs_a)
            tkeo_st = np.interp(t_rf, np.arange(len(steth_tkeo_r))/fs_a, steth_tkeo_r)
            
            # Normalize
            sm = (t_rf >= 20) & (t_rf <= 50)
            bm = (t_rf >= 5) & (t_rf <= 15)
            
            if np.sum(sm) == 0: continue
            
            def norm(env):
                s = np.maximum(env - np.percentile(env[bm], 5), 0)
                m = np.max(s[sm])
                return s / m if m > 0 else s
                
            rf_n = norm(tkeo_rf)
            st_n = norm(tkeo_st)
            
            ax.plot(t_rf, st_n, color=CS, lw=1.5, alpha=0.8, label='Stethoscope')
            ax.plot(t_rf, rf_n, color=CP, lw=1.5, alpha=0.8, label='RF Phase')
            
            ax.set_title(f"{sub_name} - Rec {rec_idx}")
            ax.set_xlim([15, 52])
            ax.set_ylim([-0.05, 1.1])
            ax.grid(True, alpha=0.3)
            
            if row == 4: ax.set_xlabel("Time (s)")
            if col == 0: ax.set_ylabel("Norm Energy")
            
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            print(f"Error on {sub_name} Rec {rec_idx}: {e}")
            ax.set_title(f"{sub_name} - Rec {rec_idx} (ERROR)")

plt.suptitle("Cohort Time-Domain Overlay Validation (RF Phase vs Stethoscope TKEO Energy)", fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"DONE: {OUT}")
