"""
Test script to calculate and plot instantaneous HR tracking
with local normalization inside the [22, 45] window.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_2_Rajveer'
rec_idx = 4
rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')

FS_RF = 10000; DEC = 10; FS = 1000; FS_100 = 100
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
st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
st_hr = bpf(st_env_100, 0.9, 2.5, FS_100)

# Local normalization inside [22, 45]
mask = (t_100 >= 22.0) & (t_100 <= 45.0)
s_hr_local = st_hr / np.max(np.abs(st_hr[mask]))
p_hr_local = -phi_hr / np.max(np.abs(phi_hr[mask]))

# Peak finding on local signals
min_dist = int(FS_100 * 0.5) # Min 0.5s between beats
p_peaks, _ = find_peaks(p_hr_local, distance=min_dist, prominence=0.25)
s_peaks, _ = find_peaks(s_hr_local, distance=min_dist, prominence=0.25)

# Limit peaks to the [22, 45] window
p_peaks = p_peaks[(t_100[p_peaks] >= 22.0) & (t_100[p_peaks] <= 45.0)]
s_peaks = s_peaks[(t_100[s_peaks] >= 22.0) & (t_100[s_peaks] <= 45.0)]

p_times = t_100[p_peaks]
s_times = t_100[s_peaks]

p_hr_vals = 60.0 / np.diff(p_times)
p_hr_t = (p_times[:-1] + p_times[1:]) / 2.0

s_hr_vals = 60.0 / np.diff(s_times)
s_hr_t = (s_times[:-1] + s_times[1:]) / 2.0

print("RF peak times:", p_times)
print("Steth peak times:", s_times)
print("RF HR values:", p_hr_vals)
print("Steth HR values:", s_hr_vals)

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(s_hr_t, s_hr_vals, 'o-', color='#2980B9', label='Stethoscope Ground Truth HR')
ax.plot(p_hr_t, p_hr_vals, 's-', color='#C0392B', label='Near-field RF (USRP) HR')
ax.set_xlim([22, 45])
ax.set_ylim([50, 90])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title('Beat-by-Beat Heart Rate Tracking inside Korotkoff Window')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\rajve\.gemini\antigravity\brain\455975f3-9b17-4899-a08c-147aeebc3fe5\artifacts\test_hr_tracking.png', dpi=150)
print("Done!")
