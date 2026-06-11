import h5py, os, numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_1_Prof_kan'
rec_idx = 1
rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')

FS_RF = 10000; FS_100 = 100
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

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

with h5py.File(rf_path, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0,:], rf[1,:]
xc, yc = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc

phi = robust_phase(i_c, q_c)
phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
t_100 = np.arange(len(phi_100)) / FS_100
phi_hr = bpf(phi_100, 0.9, 2.5, FS_100)

fs_a, audio = wavfile.read(wav_path)
if audio.ndim > 1: audio = audio.mean(axis=1)
st_bp = bpf(audio.astype(np.float64)/32768.0, 30, 1000, fs_a)
st_hilb = np.abs(signal.hilbert(st_bp))
st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
st_hr = bpf(st_env_100, 0.9, 2.5, FS_100)

mask_w = (t_100 >= 30.0) & (t_100 <= 39.0)
t_w = t_100[mask_w]
s_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
p_local = -phi_hr[mask_w] / np.max(np.abs(phi_hr[mask_w]))

plt.figure(figsize=(10, 4))
plt.plot(t_w, s_local, label='Stethoscope')
plt.plot(t_w, p_local, label='RF Phase (Inv)')
plt.grid(True)
plt.legend()
plt.title("Subject 1 Rec 1 Zoomed Heartbeats [30, 39]s")
plt.savefig(os.path.join(BASE, 'rec1_zoom.png'), dpi=150)
print("Done!")
