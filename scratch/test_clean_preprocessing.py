import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, decimate
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
dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
hist, bins = np.histogram(dphi, bins=512)
co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
dphi -= co
iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
phi = signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

FS_RF = 10000

# 1. Old method: Filter at 10 kHz, then decimate to 1 kHz
sos_dh_old = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
dh_old = decimate(sosfiltfilt(sos_dh_old, phi) * SCALE, 10, ftype='fir')

# 2. New method: Decimate to 1 kHz, then Filter at 1 kHz
phi_ds = decimate(phi, 10, ftype='fir')
sos_dh_new = butter(4, [0.4, 3.0], btype='band', fs=1000, output='sos')
dh_new = sosfiltfilt(sos_dh_new, phi_ds) * SCALE

print("Old range:", np.min(dh_old), np.max(dh_old))
print("New range (decimate then filter):", np.min(dh_new), np.max(dh_new))
print("Old std:", np.std(dh_old))
print("New std (decimate then filter):", np.std(dh_new))

# Check middle part (excluding start/end transients)
mid = slice(1000, -1000)
print("\nExcluding edge transients (middle 50s):")
print("Old range (mid):", np.min(dh_old[mid]), np.max(dh_old[mid]))
print("New range (mid):", np.min(dh_new[mid]), np.max(dh_new[mid]))
print("Old std (mid):", np.std(dh_old[mid]))
print("New std (mid):", np.std(dh_new[mid]))
