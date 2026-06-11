import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000.0
SCALE     = LAMBDA_MM / (4.0 * np.pi)

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

def get_rf_env(sub):
    if sub == 1:
        rf_path  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
        notches = [100.71, 201.43, 302.14, 402.86]
    else:
        rf_path  = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
        notches = [50.0, 64.0, 100.6, 201.2]
        
    with h5py.File(rf_path, 'r') as f:
        raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc)
    ic -= xc; qc -= yc

    phi_raw  = robust_phase(ic, qc)
    phi_clean = phi_raw.copy()
    for f0 in notches:
        phi_clean = notch_f(phi_clean, f0, FS_RF)

    vel_hi = np.append(np.diff(bpf(phi_clean, 10, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_env_hi = smooth_box(tkeo(vel_hi), 1.5, FS_RF)
    vel_env = decimate(vel_env_hi, DEC, ftype='fir')
    return vel_env

def get_steth_env(sub):
    if sub == 1:
        wav_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')
        lag = 1.7083
    else:
        wav_path = os.path.join(BASE, 'Sub_2_Rajveer', 'sthethoscope_rec04.wav')
        lag = 2.6042
        
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(1)
    audio_bp = bpf(audio, 50, 1000, fs_a)
    steth_env_a = bpf(np.abs(hilbert(audio_bp)), 20, min(200, fs_a/2-1), fs_a)
    steth_tkeo_a = smooth_box(tkeo(steth_env_a), 1.5, fs_a)
    
    t_rf = np.arange(len(get_rf_env(sub))) / FS
    t_a = np.arange(len(steth_tkeo_a)) / fs_a
    steth_env = np.interp(t_rf, t_a + lag, steth_tkeo_a)
    return steth_env

def cusum_detect(env, t, search_on=20.0, search_off=52.0, w_smooth=1.5, lower=0.15, upper=0.85):
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

rf_env_1 = get_rf_env(1)
st_env_1 = get_steth_env(1)
rf_env_2 = get_rf_env(2)
st_env_2 = get_steth_env(2)

t_grid = np.arange(len(rf_env_1)) / FS

for lower in np.arange(0.10, 0.35, 0.02):
    for upper in np.arange(0.65, 0.90, 0.02):
        rf_on1, rf_off1 = cusum_detect(rf_env_1, t_grid, lower=lower, upper=upper)
        st_on1, st_off1 = cusum_detect(st_env_1, t_grid, lower=lower, upper=upper)
        err_rf1 = abs((rf_off1 - rf_on1) - 15.80)
        err_st1 = abs((st_off1 - st_on1) - 15.80)
        
        # Sub 2
        t_grid_2 = np.arange(len(rf_env_2)) / FS
        rf_on2, rf_off2 = cusum_detect(rf_env_2, t_grid_2, lower=lower, upper=upper)
        st_on2, st_off2 = cusum_detect(st_env_2, t_grid_2, lower=lower, upper=upper)
        err_rf2 = abs((rf_off2 - rf_on2) - 14.62)
        err_st2 = abs((st_off2 - st_on2) - 14.62)
        
        total_err = err_rf1 + err_st1 + err_rf2 + err_st2
        print(f"Thresh=({lower:.2f}, {upper:.2f}) | Sub1 RF={err_rf1:.2f}s, St={err_st1:.2f}s | Sub2 RF={err_rf2:.2f}s, St={err_st2:.2f}s | Total={total_err:.2f}s")
