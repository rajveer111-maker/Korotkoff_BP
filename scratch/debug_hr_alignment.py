"""
Test script to check peak alignment between all three heartbeat waveforms
in a zoomed-in window (e.g. 30s to 36s).
"""
import h5py, os, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, find_peaks, detrend
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC1 = 10; DEC2 = 10
FS_100 = 100
FC    = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000.0) / (4.0 * np.pi)

s = dict(sub_dir='Sub_1_Prof_kan', rec=6, k_on=27.53, k_off=43.33)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

# Load RF
rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
with h5py.File(rp, 'r') as f: raw = f['data'][:]
ic, qc = -raw[0,:], raw[1,:]
xc, yc = fit_circle(ic, qc); ic -= xc; qc -= yc

phi = robust_phase(ic, qc) * SCALE
mag_raw = np.sqrt(ic**2 + qc**2)

# Downsample to 100 Hz
phi_100 = decimate(decimate(phi, DEC1, ftype='fir'), DEC2, ftype='fir')
mag_100 = decimate(decimate(mag_raw, DEC1, ftype='fir'), DEC2, ftype='fir')

# Filter at 100 Hz
phi_hr = bpf(phi_100, 1.0, 2.5, FS_100)
mag_hr = bpf(mag_100, 1.0, 2.5, FS_100)

# Load Stethoscope
wp = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")
fs_a, aud = wavfile.read(wp)
aud = aud.astype(np.float64) / 32768.0
if aud.ndim > 1: aud = aud.mean(1)

st_bp = bpf(aud, 30, 1000, fs_a)
st_env_a = np.abs(hilbert(st_bp))

t_rf = np.arange(len(phi_100)) / FS_100
t_a = np.arange(len(st_env_a)) / fs_a
st_env_100 = np.interp(t_rf, t_a, st_env_a)

st_hr = bpf(st_env_100, 1.0, 2.5, FS_100)

# Normalise
phi_hr_n = phi_hr / np.max(np.abs(phi_hr))
mag_hr_n = mag_hr / np.max(np.abs(mag_hr))
st_hr_n  = st_hr / np.max(np.abs(st_hr))

plt.figure(figsize=(10, 6))
mask = (t_rf >= 30.0) & (t_rf <= 36.0)
plt.plot(t_rf[mask], st_hr_n[mask], label='Steth Envelope Modulation (0.5-3Hz)', color='blue')
# Phase and Mag might be inverted depending on signal polarity.
# Let's plot both signs or check how they line up.
plt.plot(t_rf[mask], phi_hr_n[mask], label='RF Phase Heartbeat (0.5-3Hz)', color='red')
plt.plot(t_rf[mask], mag_hr_n[mask], label='RF Magnitude Heartbeat (0.5-3Hz)', color='purple', linestyle='--')
plt.legend()
plt.title("Heartbeat Alignment (Zoomed: 30s to 36s)")
plt.savefig('d:\\Bioview\\My_RF_work_v1\\scratch\\debug_hr_alignment.png')
print("Saved debug alignment plot")
