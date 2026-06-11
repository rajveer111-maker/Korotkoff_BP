import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'font.weight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'legend.fontsize': 11, 'lines.linewidth': 2.0,
    'axes.grid': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'multi_algo_validation_best_session.png')
FS_RF = 10000; DEC = 10; FS = 1000
CP = '#C0392B' # Red RF Phase
CS = '#2980B9' # Blue Steth

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

# ── 4 APPROACHES ─────────────────────────────────────────────────────────
def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def calc_rms(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.maximum(fftconvolve(x**2, np.ones(k)/k, mode='same'), 0))

def calc_shannon(x):
    # Normalized Shannon Energy
    x_norm = x / (np.max(np.abs(x)) + 1e-10)
    se = -(x_norm**2) * np.log(x_norm**2 + 1e-10)
    return np.maximum(se, 0)

def calc_analytic(x):
    return np.abs(hilbert(x))

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

# ── LOAD BEST SESSION ──
rf_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
wav_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

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
# Resample Stethoscope to match RF length for direct processing
audio_k_dec = signal.resample(audio_k, int(len(audio_k) * (FS/fs_a)))
# Pad or truncate to match exactly
if len(audio_k_dec) > len(t_rf): audio_k_dec = audio_k_dec[:len(t_rf)]
else: audio_k_dec = np.pad(audio_k_dec, (0, len(t_rf)-len(audio_k_dec)))

# ── CALCULATE THE 4 APPROACHES ──
# 1. TKEO
rf_tkeo = smooth(calc_tkeo(vel_dec), 1.5, FS)
st_tkeo = smooth(calc_tkeo(audio_k_dec), 1.5, FS)

# 2. RMS
rf_rms = calc_rms(vel_dec, 0.25, FS)
rf_rms = smooth(rf_rms, 1.0, FS)
st_rms = calc_rms(audio_k_dec, 0.25, FS)
st_rms = smooth(st_rms, 1.0, FS)

# 3. Shannon Energy
rf_shannon = smooth(calc_shannon(vel_dec), 1.5, FS)
st_shannon = smooth(calc_shannon(audio_k_dec), 1.5, FS)

# 4. Analytic Envelope (Absolute Hilbert)
rf_hilbert = smooth(calc_analytic(vel_dec), 1.5, FS)
st_hilbert = smooth(calc_analytic(audio_k_dec), 1.5, FS)

# Normalize
sm = (t_rf >= 15) & (t_rf <= 50)
bm = (t_rf >= 5) & (t_rf <= 15)

def norm(env):
    s = np.maximum(env - np.percentile(env[bm], 5), 0)
    m = np.max(s[sm])
    return s / m if m > 0 else s

# ── PLOTTING ──
fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=300, facecolor='white')

approaches = [
    (rf_tkeo, st_tkeo, '(A) Teager-Kaiser Energy Operator (TKEO)'),
    (rf_rms, st_rms, '(B) Root Mean Square (RMS) Energy'),
    (rf_shannon, st_shannon, '(C) Shannon Energy'),
    (rf_hilbert, st_hilbert, '(D) Absolute Analytic Envelope (Hilbert)')
]

for idx, (rf_env, st_env, title) in enumerate(approaches):
    ax = axes[idx]
    rf_n = norm(rf_env)
    st_n = norm(st_env)
    
    ax.plot(t_rf, st_n, color=CS, lw=2.5, alpha=0.9, label='Stethoscope Ground Truth')
    ax.plot(t_rf, rf_n, color=CP, lw=2.5, alpha=0.8, label='RF Phase Velocity')
    
    ax.set_title(title)
    ax.set_xlim([22, 47])
    ax.set_ylim([-0.05, 1.1])
    ax.set_ylabel('Normalized Energy')
    if idx == 3: ax.set_xlabel('Time (s)')
    if idx == 0: ax.legend(loc='upper right')
    
    # Highlight clinical window
    ax.axvspan(27.53, 43.33, color='#F39C12', alpha=0.1, zorder=0)

fig.suptitle('Algorithmic Robustness: Validating RF Phase Korotkoff Bursts Across 4 Signal Processing Methods\nSubject 1 (Prof. Kan) | Best Session (Rec 06)', fontsize=16, y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.35)
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"DONE: {OUT}")
