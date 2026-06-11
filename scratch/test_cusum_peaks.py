"""
Test script to verify automated CUSUM windowing and peak finding.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks, fftconvolve
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_2_Rajveer'
rec_idx = 4
rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')

FS_RF = 10000; DEC = 10; FS = 1000; FS_100 = 100
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

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

def cusum_detect(env, t, lo=0.08, hi=0.92):
    mask = (t >= 20.0) & (t <= 52.0)
    ev   = smooth(env[mask], 3.0, FS)
    ts   = t[mask]
    if len(ev) == 0 or ev.max() == 0: return 22.0, 42.0
    cs   = np.cumsum(ev) / (ev.sum() + 1e-12)
    ion  = np.where(cs >= lo)[0]
    ioff = np.where(cs >= hi)[0]
    return (float(ts[ion[0]])  if len(ion)  else 22.0,
            float(ts[ioff[0]]) if len(ioff) else 42.0)

with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
# circle fit
A = np.column_stack([i_raw, q_raw, np.ones_like(i_raw)])
B = -(i_raw**2 + q_raw**2)
res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
xc, yc = -res[0]/2, -res[1]/2
i_c, q_c = i_raw - xc, q_raw - yc

# Filter phase
phi = robust_phase(i_c, q_c)
phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)

phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
t_100 = np.arange(len(phi_100)) / FS_100
phi_hr = bpf(phi_100, 0.9, 2.5, FS_100)

# Audio
fs_a, audio = wavfile.read(wav_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)
st_bp = bpf(audio, 30, 1000, fs_a)
st_hilb = np.abs(signal.hilbert(st_bp))
st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)
t_rf = np.arange(len(phi_100)*10)/FS_RF
st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
st_hr = bpf(st_env_100, 0.9, 2.5, FS_100)
steth_env = np.interp(t_100, np.arange(len(st_wide_a))/fs_a, st_wide_a)

k_on, k_off = cusum_detect(steth_env, t_100)
print(f"CUSUM Window: [{k_on:.2f}, {k_off:.2f}]")

# Peak finding in window
mask_k = (t_100 >= k_on) & (t_100 <= k_off)
st_k = st_hr[mask_k]
phi_k = phi_hr[mask_k]

st_k_n = st_k / np.max(np.abs(st_k))
phi_k_n = -phi_k / np.max(np.abs(phi_k))

min_dist = int(FS_100 * 0.5)
pks_s, _ = find_peaks(st_k_n, distance=min_dist, prominence=0.12)
pks_p, _ = find_peaks(phi_k_n, distance=min_dist, prominence=0.12)

t_k = t_100[mask_k]
print("Steth peaks:", t_k[pks_s])
print("RF peaks:", t_k[pks_p])
