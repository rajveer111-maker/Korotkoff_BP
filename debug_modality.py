import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
FS_RF = 10_000
DEC = 10
FS = 1000
SCALE = ((299_792_458.0 / 0.9e9) * 1000) / (4.0 * np.pi)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])
def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
phi = robust_phase(i_c, q_c)

sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE

tkeo = np.zeros_like(vk)
tkeo[1:-1] = vk[1:-1]**2 - vk[:-2] * vk[2:]
tkeo[0] = tkeo[1]
tkeo[-1] = tkeo[-2]
m3 = smooth(np.abs(tkeo), max(1, int(FS_RF * 0.5)))

t_rf = np.arange(len(vk))/FS_RF
mask_koro = (t_rf > 27.38) & (t_rf < 42.0)
mask_base = (t_rf > 20.0) & (t_rf < 25.38)
mask_infl = (t_rf > 5.0) & (t_rf < 18.0)

print(f"Modality Dashboard TKEO (M3):")
print(f"Max in Inflation: {np.max(m3[mask_infl]):.2f}")
print(f"Max in Baseline : {np.max(m3[mask_base]):.2f}")
print(f"Mean in Baseline: {np.mean(m3[mask_base]):.2f}")
print(f"Max in Korotkoff: {np.max(m3[mask_koro]):.2f}")
print(f"Mean in Korotkoff:{np.mean(m3[mask_koro]):.2f}")
