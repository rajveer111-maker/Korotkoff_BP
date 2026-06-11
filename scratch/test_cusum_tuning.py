import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000.0
SCALE     = LAMBDA_MM / (4.0 * np.pi)

# DSP Helpers
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch_f(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth_box(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def cusum_detect(env, t, search_on=20.0, search_off=52.0, w_smooth=1.5, lower=0.08, upper=0.92):
    mask = (t >= search_on) & (t <= search_off)
    base_val = np.percentile(env, 15)
    ev_corr = np.maximum(env[mask] - base_val, 0)
    ev   = smooth_box(ev_corr, w_smooth, FS)
    ts   = t[mask]
    if len(ev) == 0 or np.max(ev) == 0: return search_on, search_off
    cs = np.cumsum(ev)
    cs = cs / cs[-1]
    i_on  = np.where(cs >= lower)[0]
    i_off = np.where(cs >= upper)[0]
    on  = float(ts[i_on[0]])  if len(i_on)  else search_on
    off = float(ts[i_off[0]]) if len(i_off) else search_off
    return on, off

# Test Subject 1 Rec 6
k_on_1, k_off_1 = 27.530, 43.330
rf_path_1  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
with h5py.File(rf_path_1, 'r') as f:
    raw_1 = f['data'][:]
ic_1, qc_1 = -raw_1[0,:], raw_1[1,:]
# Wait, len(raw_1) is 2, so raw_1[1,:]
ic_1, qc_1 = -raw_1[0,:], raw_1[1,:]
xc_1, yc_1 = fit_circle(ic_1, qc_1)
ic_1 -= xc_1; qc_1 -= yc_1

phi_raw_1  = robust_phase(ic_1, qc_1)
phi_clean_1 = phi_raw_1.copy()
for f0 in [100.71, 201.43, 302.14, 402.86]:
    phi_clean_1 = notch_f(phi_clean_1, f0, FS_RF)

# Test Subject 2 Rec 4
k_on_2, k_off_2 = 27.380, 42.000
rf_path_2  = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
with h5py.File(rf_path_2, 'r') as f:
    raw_2 = f['data'][:]
ic_2, qc_2 = -raw_2[0,:], raw_2[1,:]
xc_2, yc_2 = fit_circle(ic_2, qc_2)
ic_2 -= xc_2; qc_2 -= yc_2

phi_raw_2  = robust_phase(ic_2, qc_2)
phi_clean_2 = phi_raw_2.copy()
for f0 in [50.0, 64.0, 100.6, 201.2]:
    phi_clean_2 = notch_f(phi_clean_2, f0, FS_RF)

# Sweep
for w_smooth in [1.0, 1.2, 1.5]:
    for low_f in [10, 15, 20]:
        for lower_pct, upper_pct in [(0.06, 0.94), (0.08, 0.92), (0.10, 0.90)]:
            # Sub 1
            vel_hi_1 = np.append(np.diff(bpf(phi_clean_1, low_f, 200, FS_RF)) * FS_RF, 0.0) * SCALE
            vel_env_hi_1 = smooth_box(tkeo(vel_hi_1), 0.4, FS_RF)
            vel_env_1 = decimate(vel_env_hi_1, DEC, ftype='fir')
            t_rf_1 = np.arange(len(vel_env_1)) / FS
            
            rf_on_1, rf_off_1 = cusum_detect(vel_env_1, t_rf_1, w_smooth=w_smooth, lower=lower_pct, upper=upper_pct)
            err_1 = abs((rf_off_1 - rf_on_1) - (k_off_1 - k_on_1))
            
            # Sub 2
            vel_hi_2 = np.append(np.diff(bpf(phi_clean_2, low_f, 200, FS_RF)) * FS_RF, 0.0) * SCALE
            vel_env_hi_2 = smooth_box(tkeo(vel_hi_2), 0.4, FS_RF)
            vel_env_2 = decimate(vel_env_hi_2, DEC, ftype='fir')
            t_rf_2 = np.arange(len(vel_env_2)) / FS
            
            rf_on_2, rf_off_2 = cusum_detect(vel_env_2, t_rf_2, w_smooth=w_smooth, lower=lower_pct, upper=upper_pct)
            err_2 = abs((rf_off_2 - rf_on_2) - (k_off_2 - k_on_2))
            
            print(f"Smooth={w_smooth}s | LowF={low_f}Hz | Thresh=({lower_pct},{upper_pct}) | Sub1 Err={err_1:.2f}s | Sub2 Err={err_2:.2f}s")
