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
    vel_env_hi = smooth_box(tkeo(vel_hi), 1.5, FS_RF) # 1.5s smooth like dashboard!
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

def normalise(env, t, k_on, k_off):
    base_mask  = (t >= 20.0) & (t <= k_on - 2.0) # pre-Korotkoff quiet window (deflation ramp)
    koro_mask  = (t >= k_on) & (t <= k_off)
    baseline   = np.percentile(env[base_mask], 5) if base_mask.any() else 0.0
    e          = np.maximum(env - baseline, 0)
    peak       = np.max(e[koro_mask]) if koro_mask.any() else 1.0
    return e / (peak + 1e-12)

def detect_onset_offset(env_norm, t, thresh=0.15):
    # Search between 20s and 52s
    mask = (t >= 20.0) & (t <= 52.0)
    ts = t[mask]
    ev = env_norm[mask]
    
    # Smooth slightly more to bridge beats
    ev_smooth = smooth_box(ev, 1.0, FS)
    
    on_idx = np.where(ev_smooth >= thresh)[0]
    if len(on_idx) > 0:
        on_t = ts[on_idx[0]]
    else:
        on_t = 20.0
        
    off_idx = np.where(ev_smooth >= thresh)[0]
    if len(off_idx) > 0:
        off_t = ts[off_idx[-1]]
    else:
        off_t = 52.0
        
    return on_t, off_t

rf_env_1 = get_rf_env(1)
st_env_1 = get_steth_env(1)
rf_env_2 = get_rf_env(2)
st_env_2 = get_steth_env(2)

t_1 = np.arange(len(rf_env_1)) / FS
t_2 = np.arange(len(rf_env_2)) / FS

rf_norm_1 = normalise(rf_env_1, t_1, 27.53, 43.33)
st_norm_1 = normalise(st_env_1, t_1, 27.53, 43.33)

rf_norm_2 = normalise(rf_env_2, t_2, 27.38, 42.00)
st_norm_2 = normalise(st_env_2, t_2, 27.38, 42.00)

for thresh in np.arange(0.05, 0.35, 0.05):
    # Detect Sub 1
    rf_on_1, rf_off_1 = detect_onset_offset(rf_norm_1, t_1, thresh)
    st_on_1, st_off_1 = detect_onset_offset(st_norm_1, t_1, thresh)
    dur_rf1 = rf_off_1 - rf_on_1
    dur_st1 = st_off_1 - st_on_1
    
    # Detect Sub 2
    rf_on_2, rf_off_2 = detect_onset_offset(rf_norm_2, t_2, thresh)
    st_on_2, st_off_2 = detect_onset_offset(st_norm_2, t_2, thresh)
    dur_rf2 = rf_off_2 - rf_on_2
    dur_st2 = st_off_2 - st_on_2
    
    print(f"Thresh={thresh:.2f} | Sub1 RF={dur_rf1:.2f}s, Steth={dur_st1:.2f}s | Sub2 RF={dur_rf2:.2f}s, Steth={dur_st2:.2f}s")
