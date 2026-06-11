import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'font.weight': 'bold',
    'axes.labelsize': 11, 'axes.labelweight': 'bold',
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'legend.fontsize': 9, 'lines.linewidth': 1.5,
    'axes.grid': True, 'grid.color': '#EEEEEE', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'cohort_adaptive_detection.png')
FS_RF = 10000; DEC = 10; FS = 1000

CP = '#C0392B' # Red RF Phase
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

def smooth(x, w_sec, fs):
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

def detect_bounds(env, t, fs):
    # Search from 20s (Deflation start) to 50s
    mask = (t >= 20.0) & (t <= 50.0)
    env_s = env[mask]
    t_s = t[mask]
    if len(env_s) == 0: return 0.0, 0.0
    
    # Macro smoothing (1.5s) to connect bursts into a solid block
    macro_env = smooth(env_s, 1.5, fs)
    peak = np.max(macro_env)
    
    # Adaptive threshold: 15% of peak macro-energy
    thresh = 0.15 * peak
    active = macro_env > thresh
    
    if not np.any(active): return 0.0, 0.0
    idx = np.where(active)[0]
    return t_s[idx[0]], t_s[idx[-1]]

fig, axes = plt.subplots(5, 4, figsize=(22, 16), dpi=200, facecolor='white')
subjects = [('Sub_1_Prof_kan', 'Sub 1'), ('Sub_2_Rajveer', 'Sub 2')]

for sub_idx, (sub_dir, sub_name) in enumerate(subjects):
    for rec_idx in range(1, 11):
        col = (sub_idx * 2) + ((rec_idx - 1) % 2)
        row = (rec_idx - 1) // 2
        ax = axes[row, col]
        
        rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
        
        if not os.path.exists(rf_path) or not os.path.exists(wav_path):
            ax.set_title(f"{sub_name} - Rec {rec_idx} (MISSING)")
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
            vel_dec = decimate(vel, DEC, ftype='fir')
            t_rf = np.arange(len(vel_dec))/FS
            
            fs_a, audio = wavfile.read(wav_path)
            audio = audio.astype(np.float64)/32768.0
            if audio.ndim>1: audio = audio.mean(axis=1)
            audio_f = bpf(audio, 50, 1000, fs_a)
            audio_k = bpf(np.abs(audio_f), 20, min(200, (fs_a/2)-1), fs_a)
            audio_k_dec = signal.resample(audio_k, int(len(audio_k) * (FS/fs_a)))
            if len(audio_k_dec) > len(t_rf): audio_k_dec = audio_k_dec[:len(t_rf)]
            else: audio_k_dec = np.pad(audio_k_dec, (0, len(t_rf)-len(audio_k_dec)))
            
            rf_tkeo = smooth(calc_tkeo(vel_dec), 1.5, FS)
            st_tkeo = smooth(calc_tkeo(audio_k_dec), 1.5, FS)
            
            # Independent Adaptive Detection!
            rf_on, rf_off = detect_bounds(rf_tkeo, t_rf, FS)
            st_on, st_off = detect_bounds(st_tkeo, t_rf, FS)
            
            # Normalize
            def norm(env):
                e = np.maximum(env - np.percentile(env[(t_rf>5)&(t_rf<15)], 5), 0)
                m = np.max(e[(t_rf>20)&(t_rf<50)])
                return e / m if m > 0 else e
                
            rf_n = norm(rf_tkeo)
            st_n = norm(st_tkeo)
            
            ax.plot(t_rf, st_n, color=CS, lw=2.0, alpha=0.9, label=f'Steth (Dur: {st_off-st_on:.1f}s)')
            ax.plot(t_rf, rf_n, color=CP, lw=2.0, alpha=0.8, label=f'RF (Dur: {rf_off-rf_on:.1f}s)')
            
            # Plot Detected Bounds
            ax.axvline(st_on, color=CS, ls='--', lw=1.5, alpha=0.8)
            ax.axvline(st_off, color=CS, ls='--', lw=1.5, alpha=0.8)
            
            ax.axvline(rf_on, color=CP, ls=':', lw=2.0, alpha=0.9)
            ax.axvline(rf_off, color=CP, ls=':', lw=2.0, alpha=0.9)
            
            err_on = abs(rf_on - st_on)
            err_off = abs(rf_off - st_off)
            
            ax.set_title(f"{sub_name} Rec {rec_idx} | $\Delta$On: {err_on:.1f}s, $\Delta$Off: {err_off:.1f}s")
            ax.set_xlim([18, 52])
            ax.set_ylim([-0.05, 1.2])
            
            if row == 4: ax.set_xlabel("Time (s)")
            if col == 0: ax.set_ylabel("Norm Energy")
            
            ax.legend(loc='upper right', fontsize=8)
            
        except Exception as e:
            print(f"Error on {sub_name} Rec {rec_idx}: {e}")
            ax.set_title(f"{sub_name} Rec {rec_idx} (ERROR)")

fig.suptitle('Automated Adaptive Validation: Independent Korotkoff Boundary Detection (Deflation > 20s)\nComparing Independent Stethoscope Bounds (Blue --) vs RF Phase Bounds (Red :)', fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"DONE: {OUT}")
