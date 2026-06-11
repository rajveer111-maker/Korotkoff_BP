import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy import signal

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_1.h5'
FS = 10000

with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = -data[0,:], data[1,:]

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
iq_c = i_c + 1j * q_c

dphi = np.angle(iq_c[1:] * np.conj(iq_c[:-1]))
hist, bins = np.histogram(dphi, bins=512)
co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
dphi_c = dphi - co
iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
clip = max(3 * iqr, 0.01)
dphi_c = np.clip(dphi_c, -clip, clip)
phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
ph = signal.detrend(phase, type='linear')

t = np.arange(len(ph)) / FS
SCALE = (299792458.0 / 0.9e9) * 1000 / (4 * np.pi)

# Test displacement RMS and SNR in 10-30 Hz
sos = butter(4, [10, 30], btype='band', fs=FS, output='sos')
pk = sosfiltfilt(sos, ph)
disp_k = pk * SCALE

rms_k = np.sqrt(np.mean(disp_k[(t >= 24.0) & (t <= 41.5)]**2))
rms_b = np.sqrt(np.mean(disp_k[(t >= 46.0) & (t <= 53.0)]**2))
print(f"Displacement 10-30 Hz: RMS_k={rms_k:.6f} mm, RMS_b={rms_b:.6f} mm, SNR={rms_k/rms_b:.2f}x")

# Let's check 15-40 Hz displacement
sos2 = butter(4, [15, 40], btype='band', fs=FS, output='sos')
pk2 = sosfiltfilt(sos2, ph)
disp_k2 = pk2 * SCALE
rms_k2 = np.sqrt(np.mean(disp_k2[(t >= 24.0) & (t <= 41.5)]**2))
rms_b2 = np.sqrt(np.mean(disp_k2[(t >= 46.0) & (t <= 53.0)]**2))
print(f"Displacement 15-40 Hz: RMS_k={rms_k2:.6f} mm, RMS_b={rms_b2:.6f} mm, SNR={rms_k2/rms_b2:.2f}x")
