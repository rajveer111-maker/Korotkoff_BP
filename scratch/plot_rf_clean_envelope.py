"""
Generate a test plot to compare the raw and baseline-subtracted envelopes
for Phase and Magnitude against the Stethoscope envelope.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, detrend, iirnotch, filtfilt, fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_1_Prof_kan'
rf_path = os.path.join(BASE, sub_dir, 'Rec_1.h5')
wav_path = os.path.join(BASE, sub_dir, 'sthethoscope_rec01.wav')

FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
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

# Load and process
with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
xc, yc = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc

phi = robust_phase(i_c, q_c)
phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
vel_hi = np.append(np.diff(bpf(phi_clean, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
vel_env = smooth(np.maximum(vel_dec, 0), 1.5, FS)
t_rf = np.arange(len(vel_env)) / FS

mag_raw = np.sqrt(i_c**2 + q_c**2)
mag_clean = nf(nf(nf(mag_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
mag_hi = bpf(mag_clean, 30, 200, FS_RF)
mag_dec = decimate(smooth(tkeo(mag_hi), 0.15, FS_RF), DEC, ftype='fir')
mag_env = smooth(np.maximum(mag_dec, 0), 1.5, FS)

fs_a, audio = wavfile.read(wav_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)
st_bp = bpf(audio, 30, 1000, fs_a)
st_hilb = np.abs(hilbert(st_bp))
st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)
steth_env = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)

mask = (t_rf >= 22.0) & (t_rf <= 45.0)

# Normalize functions
def get_clean_norm(env):
    base = np.percentile(env[mask], 5)
    env_clean = np.maximum(env - base, 0)
    return env_clean / (np.max(env_clean[mask]) + 1e-12)

s_norm = get_clean_norm(steth_env)
v_norm_raw = vel_env / np.max(vel_env[mask])
v_norm_clean = get_clean_norm(vel_env)
m_norm_clean = get_clean_norm(mag_env)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(t_rf, s_norm, label='Steth Envelope (Clean Norm)', color='#2980B9', lw=2)
plt.plot(t_rf, v_norm_raw, label='Phase Env (Raw Norm - Flat)', color='#BDC3C7', lw=1.5, ls='--')
plt.plot(t_rf, v_norm_clean, label='Phase Env (Clean Norm)', color='#E74C3C', lw=2)
plt.plot(t_rf, m_norm_clean, label='Mag Env (Clean Norm)', color='#9B59B6', lw=2)

plt.xlim([22, 45])
plt.ylim([0, 1.15])
plt.title("Comparison of Normalization Methods for Session 1")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Envelope")
plt.legend()
plt.grid(True, alpha=0.3)

out_fig = r'C:\Users\rajve\.gemini\antigravity\brain\455975f3-9b17-4899-a08c-147aeebc3fe5\artifacts\overlay_test.png'
plt.savefig(out_fig, dpi=150)
print(f"Saved test plot to {out_fig}")
