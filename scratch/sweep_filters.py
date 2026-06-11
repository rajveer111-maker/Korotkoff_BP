"""
Parameter sweep to find the best bandpass filter range and smoothing width
to get the cleanest alignment between RF phase velocity and Stethoscope envelopes.
We test:
- Cutoff bands: 30-200 Hz, 50-200 Hz, 60-250 Hz, 80-200 Hz
- Smooth widths: 0.5s, 1.0s, 1.5s
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, iirnotch, filtfilt, fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_1_Prof_kan'
rf_path = os.path.join(BASE, sub_dir, 'Rec_6.h5')
wav_path = os.path.join(BASE, sub_dir, 'sthethoscope_rec06.wav')

FS_RF = 10000; DEC = 10; FS = 1000
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    k = max(1, int(w * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

# Load raw
with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
xc, yc = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
phi = robust_phase(i_c, q_c)
phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)

# Steth
fs_a, audio = wavfile.read(wav_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)
st_bp = bpf(audio, 30, 1000, fs_a)
st_hilb = np.abs(hilbert(st_bp))
st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)

fig, axs = plt.subplots(4, 3, figsize=(15, 12))
bands = [(30, 200), (50, 200), (60, 250), (80, 250)]
widths = [0.5, 1.0, 1.5]

for r_idx, (lo, hi) in enumerate(bands):
    vel_hi = np.append(np.diff(bpf(phi_clean, lo, hi, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
    
    for c_idx, w in enumerate(widths):
        rf_env = smooth(np.maximum(vel_dec, 0), w, FS)
        t_rf = np.arange(len(rf_env)) / FS
        
        steth_env = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)
        
        mask = (t_rf >= 22.0) & (t_rf <= 45.0)
        
        # Norm
        def norm(env):
            base = np.percentile(env[mask], 5)
            env_c = np.maximum(env - base, 0)
            return env_c / (np.max(env_c[mask]) + 1e-12)
            
        s_n = norm(steth_env)
        r_n = norm(rf_env)
        
        ax = axs[r_idx, c_idx]
        ax.plot(t_rf, s_n, label='Steth')
        ax.plot(t_rf, r_n, label='RF')
        ax.set_xlim([22, 45])
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.set_title(f"Smooth = {w}s")
        if c_idx == 0:
            ax.set_ylabel(f"Band: {lo}-{hi} Hz")

axs[0, 0].legend()
plt.tight_layout()
plt.savefig(r'C:\Users\rajve\.gemini\antigravity\brain\455975f3-9b17-4899-a08c-147aeebc3fe5\artifacts\sweep_filters.png', dpi=150)
print("Sweep done!")
