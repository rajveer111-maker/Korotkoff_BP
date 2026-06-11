import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'font.weight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'legend.fontsize': 10, 'lines.linewidth': 1.8,
    'axes.grid': True, 'grid.color': '#EEEEEE', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'multi_algo_validation_6_best.png')
FS_RF = 10000; DEC = 10; FS = 1000
CP = '#C0392B' # Red RF Phase
CS = '#2980B9' # Blue Steth

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

# ── APPROACHES ───────────────────────────────────────────────────────────
def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def calc_rms(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.maximum(fftconvolve(x**2, np.ones(k)/k, mode='same'), 0))

def calc_shannon(x):
    x_norm = x / (np.max(np.abs(x)) + 1e-10)
    se = -(x_norm**2) * np.log(x_norm**2 + 1e-10)
    return np.maximum(se, 0)

def smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')
# ────────────────────────────────────────────────────────────────────────

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

# 6 Best Sessions (sub_dir, sub_name, rec_idx, k_on, k_off, lag, defl, notches)
sessions = [
    ('Sub_1_Prof_kan', 'Subject 1', 4, 29.5, 44.75, 0.875, 18.0, [100.71, 201.43, 302.14, 402.86]),
    ('Sub_1_Prof_kan', 'Subject 1', 6, 27.53, 43.33, 1.7083, 18.0, [100.71, 201.43, 302.14, 402.86]),
    ('Sub_1_Prof_kan', 'Subject 1', 7, 24.96875, 39.91875, -1.3271, 18.0, [100.71, 201.43, 302.14, 402.86]),
    ('Sub_2_Rajveer', 'Subject 2', 4, 27.38, 42.00, 2.6042, 18.6, [50.0, 64.0, 100.6, 201.2]),
    ('Sub_2_Rajveer', 'Subject 2', 8, 27.1875, 42.1875, 3.9375, 18.6, [50.0, 64.0, 100.6, 201.2]),
    ('Sub_2_Rajveer', 'Subject 2', 10, 25.575, 40.375, 5.975, 18.6, [50.0, 64.0, 100.6, 201.2])
]

fig, axes = plt.subplots(6, 3, figsize=(18, 20), dpi=250, facecolor='white')

for row_idx, (sub_dir, sub_name, rec_idx, k_on, k_off, lag, defl, notches) in enumerate(sessions):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    print(f"Processing {sub_name} Rec {rec_idx}...")
    
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    phi_raw = robust_phase(i_raw-xc, q_raw-yc)
    
    # Notch filter the raw phase
    p = phi_raw.copy()
    for freq in notches:
        p = notch(p, freq, FS_RF)
        
    # Bandpass filter and derivative for RMG Korotkoff band (30-180 Hz)
    vel = np.append(np.diff(bpf(p, 30, 180, FS_RF))*FS_RF, 0.0)
    
    # Zero out outside deflation clean window to remove pump and cuff dump transients
    t_rf_full = np.arange(len(vel))/FS_RF
    t_start_clean = defl + 3.0
    t_end_clean = k_off + 1.2
    vel[(t_rf_full < t_start_clean) | (t_rf_full > t_end_clean)] = 0.0
    
    vel_dec = decimate(vel, DEC, ftype='fir')
    t_rf = np.arange(len(vel_dec))/FS
    
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64)/32768.0
    if audio.ndim>1: audio = audio.mean(axis=1)
    audio_f = bpf(audio, 50, 1000, fs_a)
    audio_k = bpf(np.abs(audio_f), 20, min(200, (fs_a/2)-1), fs_a)
    
    # Decimate audio slightly to speed up hilbert
    DEC_A = 10
    fs_ad = fs_a // DEC_A
    audio_k_d = decimate(audio_k, DEC_A, ftype='fir')
    
    # Zero out stethoscope audio outside clean deflation window
    t_a_full = (np.arange(len(audio_k_d)) / fs_ad) + lag
    audio_k_d[(t_a_full < t_start_clean) | (t_a_full > t_end_clean)] = 0.0
    
    # Calc 3 Approaches
    # TKEO
    r_tkeo = smooth(calc_tkeo(vel_dec), 1.5, FS)
    s_tkeo = smooth(calc_tkeo(audio_k_d), 1.5, fs_ad)
    # RMS
    r_rms = smooth(calc_rms(vel_dec, 0.25, FS), 1.0, FS)
    s_rms = smooth(calc_rms(audio_k_d, 0.25, fs_ad), 1.0, fs_ad)
    # Shannon
    r_shan = smooth(calc_shannon(vel_dec), 1.5, FS)
    s_shan = smooth(calc_shannon(audio_k_d), 1.5, fs_ad)
    
    # Interpolate steth back to FS using the subject-specific alignment lag
    t_a = (np.arange(len(s_tkeo)) / fs_ad) + lag
    s_tkeo = np.interp(t_rf, t_a, s_tkeo, left=0.0, right=0.0)
    s_rms = np.interp(t_rf, t_a, s_rms, left=0.0, right=0.0)
    s_shan = np.interp(t_rf, t_a, s_shan, left=0.0, right=0.0)
    
    approaches = [(r_tkeo, s_tkeo, 'TKEO Energy'), 
                  (r_rms, s_rms, 'RMS Energy'), 
                  (r_shan, s_shan, 'Shannon Energy')]
    
    sm = (t_rf >= k_on-3) & (t_rf <= k_off+3)
    bm = (t_rf >= 5) & (t_rf <= k_on-3)
    
    def norm(env):
        s = np.maximum(env - np.percentile(env[bm], 5), 0)
        m = np.max(s[sm])
        return s / m if m > 0 else s
        
    for col_idx, (rf_env, st_env, algo_name) in enumerate(approaches):
        ax = axes[row_idx, col_idx]
        rf_n = norm(rf_env)
        st_n = norm(st_env)
        
        ax.plot(t_rf, st_n, color=CS, lw=2.0, alpha=0.9, label='Steth (Acoustic)')
        ax.plot(t_rf, rf_n, color=CP, lw=2.0, alpha=0.8, label='RF (Phase Velocity)')
        
        if row_idx == 0: ax.set_title(algo_name, fontsize=16)
        ax.set_xlim([k_on - 3, k_off + 3])
        ax.set_ylim([-0.05, 1.1])
        
        # Overlay a subtle background for the active region
        ax.axvspan(k_on, k_off, color='#F39C12', alpha=0.08, zorder=0)
        ax.axvline(k_on, color='#F39C12', lw=1.5, ls='--', zorder=2)
        ax.axvline(k_off, color='#F39C12', lw=1.5, ls='--', zorder=2)
        
        if col_idx == 0: ax.set_ylabel(f"{sub_name} Rec {rec_idx}\nNorm Energy", fontsize=11)
        if row_idx == 5: ax.set_xlabel("Time (s)", fontsize=12)
        if row_idx == 0 and col_idx == 0: ax.legend(loc='upper right')

fig.suptitle('Algorithmic Invariance Validation: 3 Signal Processing Methods Across 6 High-Fidelity Sessions', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.subplots_adjust(hspace=0.35, wspace=0.15)

# Save to base and RMG paper folders
OUT2 = r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\figures\diagnostic\multi_algo_validation_6_best.png'
os.makedirs(os.path.dirname(OUT2), exist_ok=True)
plt.savefig(OUT, dpi=250, bbox_inches='tight')
plt.savefig(OUT2, dpi=250, bbox_inches='tight')
print(f"DONE: {OUT}")
print(f"DONE: {OUT2}")
