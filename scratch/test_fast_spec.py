"""
Test fast decimated STFT spectrogram calculation.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, spectrogram, detrend

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; FC = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000.0) / (4.0 * np.pi)

s = dict(sub_dir='Sub_1_Prof_kan', rec=6)

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

phi = robust_phase(ic, qc)
vel_hi = np.append(np.diff(bpf(phi, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE

# Decimate to 1 kHz first
vel_hi_1k = decimate(vel_hi, 10, ftype='fir')

# Calculate STFT spectrogram
f, t_spec, Sxx = spectrogram(vel_hi_1k, fs=1000, nperseg=128, noverlap=110)

print("Spectrogram shape:", Sxx.shape)
print("Frequencies range:", f[0], "to", f[-1])
