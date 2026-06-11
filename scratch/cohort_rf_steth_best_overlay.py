import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── PUBLICATION STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'font.weight':       'bold',
    'axes.labelsize':    14,
    'axes.labelweight':  'bold',
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   12,
    'legend.framealpha': 0.92,
    'lines.linewidth':   2.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#E0E0E0',
    'grid.linewidth':    0.8,
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'cohort_rf_steth_best_overlay.png')
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

# Best 8 Sessions (4 from Sub 1, 4 from Sub 2)
best_sessions = [
    ('Sub_1_Prof_kan', 'Sub 1 (Prof. Kan)', 4),
    ('Sub_2_Rajveer', 'Sub 2 (Rajveer)', 4),
    ('Sub_1_Prof_kan', 'Sub 1 (Prof. Kan)', 5),
    ('Sub_2_Rajveer', 'Sub 2 (Rajveer)', 6),
    ('Sub_1_Prof_kan', 'Sub 1 (Prof. Kan)', 6),
    ('Sub_2_Rajveer', 'Sub 2 (Rajveer)', 8),
    ('Sub_1_Prof_kan', 'Sub 1 (Prof. Kan)', 7),
    ('Sub_2_Rajveer', 'Sub 2 (Rajveer)', 10)
]

fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=300, facecolor='white')
axes = axes.flatten()

for idx, (sub_dir, sub_name, rec_idx) in enumerate(best_sessions):
    ax = axes[idx]
    
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    
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
        sm = (t_rf >= 15) & (t_rf <= 50)
        bm = (t_rf >= 5) & (t_rf <= 15)
        
        def norm(env):
            s = np.maximum(env - np.percentile(env[bm], 5), 0)
            m = np.max(s[sm])
            return s / m if m > 0 else s
            
        rf_n = norm(tkeo_rf)
        st_n = norm(tkeo_st)
        
        ax.plot(t_rf, st_n, color=CS, lw=2.5, alpha=0.9, label='Stethoscope Acoustic Envelope')
        ax.plot(t_rf, rf_n, color=CP, lw=2.5, alpha=0.8, ls='-', label='RF Phase Velocity Envelope')
        
        ax.set_title(f"{sub_name} | Session {rec_idx:02d}")
        ax.set_xlim([15, 50])
        ax.set_ylim([-0.05, 1.1])
        ax.grid(True, alpha=0.4, ls='--')
        
        if idx >= 6: ax.set_xlabel("Time (s)")
        if idx % 2 == 0: ax.set_ylabel("Normalized TKEO Energy")
        
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.95)
            
    except Exception as e:
        print(f"Error on {sub_name} Rec {rec_idx}: {e}")
        ax.set_title(f"{sub_name} | Session {rec_idx:02d} (ERROR)")

fig.suptitle(
    "Radiomyography Multi-Session Validation: Cross-Modality Korotkoff Envelopes\n"
    "Comparing RF Phase Velocity vs Stethoscope Acoustic Energy Across 8 Selected High-Fidelity Sessions",
    fontsize=22, fontweight='bold', color='#1C1C1C', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.subplots_adjust(hspace=0.25, wspace=0.10)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nDONE: {OUT}")
