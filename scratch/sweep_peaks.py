import h5py, os, numpy as np
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
s_hr_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
p_hr_local = -phi_hr[mask_w] / np.max(np.abs(phi_hr[mask_w]))

print("Sweep on distance (in samples at 100Hz) and prominence:")
for dist in [50, 60, 65, 70, 75]:
    for prom in [0.1, 0.15, 0.2, 0.25]:
        pks_s, _ = find_peaks(s_hr_local, distance=dist, prominence=prom)
        pks_p, _ = find_peaks(p_hr_local, distance=dist, prominence=prom)
        
        s_times = t_w[pks_s]
        p_times = t_w[pks_p]
        
        s_hr = 60.0 / np.diff(s_times) if len(s_times) > 1 else []
        p_hr = 60.0 / np.diff(p_times) if len(p_times) > 1 else []
        
        mean_s_hr = np.mean(s_hr) if len(s_hr) > 0 else 0
        mean_p_hr = np.mean(p_hr) if len(p_hr) > 0 else 0
        std_p_hr = np.std(p_hr) if len(p_hr) > 0 else 0
        
        print(f"dist={dist}, prom={prom} | Steth: {len(pks_s)} peaks (mean HR={mean_s_hr:.1f}) | RF: {len(pks_p)} peaks (mean HR={mean_p_hr:.1f}, std={std_p_hr:.1f})")
