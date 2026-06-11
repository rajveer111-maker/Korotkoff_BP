import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch, decimate
import h5py

SUB1_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_6.h5"
SCALE = (299792458.0 / 0.9e9 * 1000.0) / (4.0 * np.pi)

# Load RF
with h5py.File(SUB1_RF, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = -data[0, :], data[1, :]
A = np.column_stack([i_raw, q_raw, np.ones_like(i_raw)])
B = -(i_raw**2 + q_raw**2)
res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
xc, yc = -res[0]/2, -res[1]/2
i_c = i_raw - xc
q_c = q_raw - yc
iq = i_c + 1j*q_c

# Get raw phase difference (already a velocity!)
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
hist, bins = np.histogram(dphi, bins=512)
co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
dphi -= co
iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))

# Scale to velocity in mm/s
v_raw = dphi * 10000 * SCALE # velocity at 10 kHz

# Old method: vk_old = diff(bpf(phi))
phi = signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')
sos_vk_old = butter(4, [30, 180], btype='band', fs=10000, output='sos')
vk_old = np.append(np.diff(sosfiltfilt(sos_vk_old, phi)) * 10000, 0) * SCALE

# New method: decimate v_raw to 1000 Hz, then bandpass filter
v_raw_1k = decimate(v_raw, 10, ftype='fir')
sos_vk_new = butter(4, [30, 180], btype='band', fs=1000, output='sos')
vk_new = sosfiltfilt(sos_vk_new, v_raw_1k)

print("Old velocity range:", np.min(vk_old), np.max(vk_old))
print("New velocity range (decimate-first):", np.min(vk_new), np.max(vk_new))
print("Old velocity std:", np.std(vk_old))
print("New velocity std (decimate-first):", np.std(vk_new))
