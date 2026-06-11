import h5py, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, fftconvolve

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = os.path.join(BASE, 'preprocessing_variations.png')
FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

# DSP Helpers
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch_f(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth_box(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def normalise(env, t, k_on, k_off, pct=5):
    base_mask  = (t >= 25.0) & (t <= k_on - 1.0)
    koro_mask  = (t >= k_on) & (t <= k_off)
    baseline   = np.percentile(env[base_mask], pct) if base_mask.any() else 0.0
    e          = np.maximum(env - baseline, 0)
    peak       = np.max(e[koro_mask]) if koro_mask.any() else 1.0
    return e / (peak + 1e-12)

SESSIONS = [
    dict(sub_dir='Sub_1_Prof_kan', label='Sub 1 (Prof. Kan) | Rec 06',
         rec=6,  k_on=27.53, k_off=43.33, notches=[100.71, 201.43, 302.14, 402.86, 50.0]),
    dict(sub_dir='Sub_2_Rajveer',  label='Sub 2 (Rajveer) | Rec 04',
         rec=4,  k_on=27.38, k_off=42.00, notches=[50.0, 64.0, 100.6, 201.2]),
]

fig, axes = plt.subplots(4, 2, figsize=(18, 16), facecolor='white')

for col, s in enumerate(SESSIONS):
    rf_path = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    with h5py.File(rf_path, 'r') as f:
        raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc)
    ic -= xc; qc -= yc
    
    # 1. Phase raw
    phi_raw = robust_phase(ic, qc)
    phi_clean = phi_raw.copy()
    for f0 in s['notches']:
        phi_clean = notch_f(phi_clean, f0, FS_RF)
        
    # 2. Magnitude raw
    mag_raw = np.sqrt(ic**2 + qc**2)
    mag_clean = mag_raw.copy()
    for f0 in s['notches']:
        mag_clean = notch_f(mag_clean, f0, FS_RF)
        
    # Evaluate 3 filter bands:
    # A: 10-200 Hz
    # B: 30-180 Hz
    # C: 40-150 Hz (with derivative for Magnitude)
    
    # Bands to plot:
    bands = [
        (10, 200, False, '10-200 Hz (no deriv)'),
        (30, 200, True, '30-200 Hz (with deriv)'),
        (40, 180, True, '40-180 Hz (with deriv)'),
        (40, 150, True, '40-150 Hz (with deriv)')
    ]
    
    t = np.arange(len(ic)) / FS_RF
    t_ds = decimate(t, DEC, ftype='fir')
    
    for row_idx, (lo, hi, use_deriv, name) in enumerate(bands):
        # Phase velocity env
        vel_hi = np.append(np.diff(bpf(phi_clean, lo, hi, FS_RF)) * FS_RF, 0.0) * SCALE
        vel_env_hi = smooth_box(tkeo(vel_hi), 1.5, FS_RF)
        vel_env = decimate(vel_env_hi, DEC, ftype='fir')
        vel_n = normalise(vel_env, t_ds, s['k_on'], s['k_off'], pct=5)
        
        # Magnitude env
        if use_deriv:
            mag_filt = bpf(mag_clean, lo, hi, FS_RF)
            mag_vel = np.append(np.diff(mag_filt)*FS_RF, 0.0)
            mag_env_hi = smooth_box(tkeo(mag_vel), 1.5, FS_RF)
        else:
            mag_filt = bpf(mag_clean, lo, hi, FS_RF)
            mag_env_hi = smooth_box(tkeo(mag_filt), 1.5, FS_RF)
        mag_env = decimate(mag_env_hi, DEC, ftype='fir')
        mag_n = normalise(mag_env, t_ds, s['k_on'], s['k_off'], pct=5)
        
        ax = axes[row_idx, col]
        ax.plot(t_ds, vel_n, color='#C0392B', lw=1.5, label='Phase Vel')
        ax.plot(t_ds, mag_n, color='#8E44AD', lw=1.5, label='Magnitude')
        ax.axvspan(s['k_on'], s['k_off'], color='#EAECEE', alpha=0.5, zorder=0)
        ax.axvline(s['k_on'], color='#F39C12', ls='--')
        ax.axvline(s['k_off'], color='#F39C12', ls='--')
        ax.set_xlim(20, 50)
        ax.set_ylim(-0.05, 1.45)
        ax.set_title(f"{s['label']} | {name}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=200)
plt.close()
print("Saved comparison dashboard to:", OUT)
