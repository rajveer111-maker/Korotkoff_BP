import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy import signal

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
FS = 10000
FC = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000
SCALE = LAMBDA / (4 * np.pi)

print("Loading data...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = -data[0,:], data[1,:] # apply USRP sign convention

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    R = np.sqrt(xc**2 + yc**2 - c)
    return xc, yc, R

xc, yc, R = fit_circle(i_raw, q_raw)
print(f"Circle center: ({xc:.4f}, {yc:.4f}), Radius: {R:.4f}")

# Center the IQ signal
i_c = i_raw - xc
q_c = q_raw - yc
iq_c = i_c + 1j * q_c

# Use robust phase estimation on the centered IQ signal
def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    # Remove clutter offset (mode of dphi hist)
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    
    # Adaptive thresholding for phase steps
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3 * iqr, 0.01)
    dphi_c = np.clip(dphi_c, -clip, clip)
    
    # Reconstruct phase
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return signal.detrend(phase, type='linear')

print("Extracting robust phase from centered IQ...")
phi_robust = robust_phase(iq_c)
print(f"Robust phase range: {phi_robust.min():.4f} to {phi_robust.max():.4f} rad")

# Displacement (mm)
disp = phi_robust * SCALE
print(f"Robust displacement range: {disp.min():.4f} to {disp.max():.4f} mm")

# Heartbeat displacement (0.4-3 Hz)
sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS, output='sos')
dh = sosfiltfilt(sos_h, phi_robust) * SCALE
print(f"Heartbeat displacement range: {dh.min():.4f} to {dh.max():.4f} mm, std: {np.std(dh):.4f} mm")

# Korotkoff velocity (10-100 Hz)
sos_k = butter(4, [10, 100], btype='band', fs=FS, output='sos')
pk = sosfiltfilt(sos_k, phi_robust)

vk = np.append(np.diff(pk) * FS, 0) * SCALE
print(f"Korotkoff velocity (10-100 Hz) range: {vk.min():.4f} to {vk.max():.4f} mm/s, RMS: {np.sqrt(np.mean(vk**2)):.4f} mm/s")

# Let's try 10-50 Hz (standard Korotkoff fundamental band in literature)
sos_k50 = butter(4, [10, 50], btype='band', fs=FS, output='sos')
pk50 = sosfiltfilt(sos_k50, phi_robust)
vk50 = np.append(np.diff(pk50) * FS, 0) * SCALE
print(f"Korotkoff velocity (10-50 Hz) range: {vk50.min():.4f} to {vk50.max():.4f} mm/s, RMS: {np.sqrt(np.mean(vk50**2)):.4f} mm/s")
