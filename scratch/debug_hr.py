"""
Save a debug plot of the raw and filtered signals to see why peak detection is failing.
"""
import h5py, os, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, detrend
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC1 = 10; DEC2 = 10
FS_100 = 100

s = dict(sub_dir='Sub_1_Prof_kan', rec=6, k_on=27.53, k_off=43.33)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def robust_phase_clipping(ic, qc):
    iq = ic + 1j*qc
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dc = dphi - co
    dc = np.clip(dc, -0.0002, 0.0002)
    raw_cum = np.insert(np.cumsum(dc), 0, 0.0)
    return detrend(raw_cum)

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

phi = robust_phase_clipping(ic, qc)
mag_raw = np.sqrt(ic**2 + qc**2)

phi_1k = decimate(phi, DEC1, ftype='fir')
phi_100 = decimate(phi_1k, DEC2, ftype='fir')
mag_1k = decimate(mag_raw, DEC1, ftype='fir')
mag_100 = decimate(mag_1k, DEC2, ftype='fir')

phi_hr = bpf(phi_100, 0.5, 3.0, FS_100)
mag_hr = bpf(mag_100, 0.5, 3.0, FS_100)

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(phi_100, label='phi_100')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(phi_hr, label='phi_hr')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(mag_100, label='mag_100')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(mag_hr, label='mag_hr')
plt.legend()
plt.savefig('d:\\Bioview\\My_RF_work_v1\\scratch\\debug_hr.png')
print("Saved debug plot")
