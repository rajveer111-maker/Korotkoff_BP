"""
Definitive Korotkoff Duration Validation Dashboard
- Uses the 2 best known sessions (Sub1 Rec6, Sub2 Rec4) with ground-truth windows
- Computes 4 energy approaches for BOTH modalities side-by-side
- Explicitly computes and annotates Korotkoff duration from each approach
- Layout: 4 rows (approaches) x 2 columns (subjects) = 8 panels
- The stethoscope sets the "Ground Truth" duration; RF must match it
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11.5, 'font.weight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'axes.titlesize': 12.5, 'axes.titleweight': 'bold',
    'legend.fontsize': 10, 'lines.linewidth': 2.0,
    'axes.grid': True, 'grid.color': '#EEEEEE', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = os.path.join(BASE, 'korotkoff_duration_validation_300dpi.png')
FS_RF = 10000; DEC = 10; FS = 1000
CP = '#C0392B'  # Red – RF
CS = '#2980B9'  # Blue – Steth
CA = '#F39C12'  # Amber – duration annotation

# ── Known best sessions with manually-verified clinical windows ──────────────
SESSIONS = [
    dict(sub_dir='Sub_1_Prof_kan', label='Subject 1 (Prof. Kan)\nRec 06',
         rec=6, k_on=27.53, k_off=43.33, zoom=(22, 48)),
    dict(sub_dir='Sub_2_Rajveer',  label='Subject 2 (Rajveer)\nRec 04',
         rec=4, k_on=27.38, k_off=42.00, zoom=(22, 47)),
]

# ── DSP helpers ──────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

def smooth_box(x, w_sec, fs):
    k = max(1, int(w_sec*fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

# ── 4 Energy Approaches ──────────────────────────────────────────────────────
def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def rms_env(x, fs, w=0.25):
    k = max(1, int(w*fs))
    return np.sqrt(np.maximum(fftconvolve(x**2, np.ones(k)/k, mode='same'), 0))

def shannon_env(x):
    n = x / (np.max(np.abs(x)) + 1e-12)
    se = -(n**2) * np.log(n**2 + 1e-12)
    return np.maximum(se, 0)

def hilbert_env(x):
    return np.abs(hilbert(x))

ALGOS = [
    ('TKEO',           lambda v, fs: smooth_box(tkeo(v), 1.5, fs)),
    ('RMS Energy',     lambda v, fs: smooth_box(rms_env(v, fs, 0.25), 1.0, fs)),
    ('Shannon Energy', lambda v, fs: smooth_box(shannon_env(v), 1.5, fs)),
    ('Hilbert Env',    lambda v, fs: smooth_box(hilbert_env(v), 1.0, fs)),
]

# ── Adaptive Duration Detector (runs on steth to find ground truth, ──────────
# ── then RF must confirm it)  ─────────────────────────────────────────────────
def detect_duration(env, t, search_on, search_off, thresh_pct=0.20):
    mask = (t >= search_on) & (t <= search_off)
    ev = smooth_box(env[mask], 2.0, FS)  # extra-wide smooth for macro shape
    ts = t[mask]
    if len(ev) == 0 or np.max(ev) == 0: return search_on, search_off
    thr = thresh_pct * np.max(ev)
    active = ev >= thr
    if not np.any(active): return search_on, search_off
    idx = np.where(active)[0]
    return float(ts[idx[0]]), float(ts[idx[-1]])

# ── Normalise to 0-1 within the Korotkoff window ─────────────────────────────
def normalise(env, t, k_on, k_off):
    base_mask = (t >= 5) & (t <= 18)
    k_mask    = (t >= k_on) & (t <= k_off)
    baseline  = np.percentile(env[base_mask], 5) if base_mask.any() else 0.0
    e = np.maximum(env - baseline, 0)
    peak = np.max(e[k_mask]) if k_mask.any() else 1.0
    return e / (peak + 1e-12)

# ── Pre-load all sessions ─────────────────────────────────────────────────────
print("Loading sessions...")
sessions_data = []
for s in SESSIONS:
    print(f"  {s['label'].replace(chr(10),' ')}...")
    rf_path  = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    wav_path = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")

    with h5py.File(rf_path, 'r') as f:
        raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc)
    phi = robust_phase(ic-xc, qc-yc)
    phi = notch(notch(notch(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    vel_hi = np.append(np.diff(bpf(phi, 10, 200, FS_RF))*FS_RF, 0.0)
    vel_dec = decimate(vel_hi, DEC, ftype='fir')
    t_rf = np.arange(len(vel_dec)) / FS

    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(1)
    audio_bp = bpf(audio, 50, 1000, fs_a)
    steth_k  = bpf(np.abs(hilbert(audio_bp)), 20, min(200, fs_a/2-1), fs_a)
    steth_k_dec = signal.resample(steth_k, int(len(steth_k)*(FS/fs_a)))
    n = len(t_rf)
    if len(steth_k_dec) > n: steth_k_dec = steth_k_dec[:n]
    else: steth_k_dec = np.pad(steth_k_dec, (0, n-len(steth_k_dec)))

    sessions_data.append(dict(
        **s,
        t=t_rf,
        vel=vel_dec,
        steth=steth_k_dec,
    ))

# ── Figure: 4 rows × 2 cols ───────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(20, 18), dpi=300, facecolor='white',
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.12})

for col, sd in enumerate(sessions_data):
    t      = sd['t']
    vel    = sd['vel']
    steth  = sd['steth']
    k_on   = sd['k_on']
    k_off  = sd['k_off']
    zl, zr = sd['zoom']

    for row, (algo_name, algo_fn) in enumerate(ALGOS):
        ax = axes[row, col]

        # --- Compute envelopes ---
        rf_env    = algo_fn(vel,   FS)
        steth_env = algo_fn(steth, FS)

        # --- Normalise ---
        rf_n    = normalise(rf_env,    t, k_on, k_off)
        steth_n = normalise(steth_env, t, k_on, k_off)

        # --- Adaptive detection using Stethoscope as reference ---
        st_on, st_off = detect_duration(steth_env, t, 20, 50, thresh_pct=0.20)
        rf_on, rf_off = detect_duration(rf_env,    t, 20, 50, thresh_pct=0.20)

        st_dur = st_off - st_on
        rf_dur = rf_off - rf_on
        dur_err = abs(rf_dur - st_dur)

        # --- Plot ---
        ax.plot(t, steth_n, color=CS, lw=2.2, alpha=0.9,
                label=f'Steth GT  ({st_dur:.1f}s)')
        ax.plot(t, rf_n,    color=CP, lw=2.2, alpha=0.85,
                label=f'RF Radar  ({rf_dur:.1f}s)')

        # Ground-truth shaded region (stethoscope)
        ax.axvspan(st_on, st_off, color=CS, alpha=0.10, zorder=0)
        ax.axvline(st_on,  color=CS, ls='--', lw=1.8)
        ax.axvline(st_off, color=CS, ls='--', lw=1.8)

        # RF detected region
        ax.axvspan(rf_on, rf_off, color=CP, alpha=0.10, zorder=0)
        ax.axvline(rf_on,  color=CP, ls=':', lw=2.0)
        ax.axvline(rf_off, color=CP, ls=':', lw=2.0)

        # Duration annotation arrows
        y_annot = 1.05
        ax.annotate('', xy=(st_off, y_annot), xytext=(st_on, y_annot),
                    arrowprops=dict(arrowstyle='<->', color=CS, lw=2))
        ax.text((st_on+st_off)/2, y_annot+0.04,
                f'GT: {st_dur:.1f}s', ha='center', va='bottom',
                color=CS, fontsize=9.5, fontweight='bold')

        ax.annotate('', xy=(rf_off, y_annot+0.15), xytext=(rf_on, y_annot+0.15),
                    arrowprops=dict(arrowstyle='<->', color=CP, lw=2))
        ax.text((rf_on+rf_off)/2, y_annot+0.19,
                f'RF: {rf_dur:.1f}s', ha='center', va='bottom',
                color=CP, fontsize=9.5, fontweight='bold')

        ax.set_xlim(zl, zr)
        ax.set_ylim(-0.05, 1.40)
        panel_letter = chr(65 + row * 2 + col)
        sub_label    = sd["label"].replace('\n', ' | ')
        ax.set_title(f'({panel_letter}) {sub_label} – {algo_name}'
                     f'\n|Duration Error|: {dur_err:.1f}s', fontsize=12.5)
        if row == 3: ax.set_xlabel('Time (s)', fontsize=13)
        if col == 0: ax.set_ylabel('Normalized Energy', fontsize=13)
        if row == 0 and col == 0:
            ax.legend(loc='upper right', framealpha=0.9)

# Global title
fig.suptitle(
    'Korotkoff Duration Validation: RF Phase vs Stethoscope Ground Truth\n'
    '4 Signal Processing Approaches x 2 Best Sessions (300 DPI)',
    fontsize=20, y=0.995
)

plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f'DONE -> {OUT}')
