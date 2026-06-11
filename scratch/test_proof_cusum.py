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

def normalise(env, t, k_on, k_off, pct=5):
    base_mask  = (t >= 20.0) & (t <= k_on - 2.0)
    koro_mask  = (t >= k_on) & (t <= k_off)
    baseline   = np.percentile(env[base_mask], pct) if base_mask.any() else 0.0
    e          = np.maximum(env - baseline, 0)
    peak       = np.max(e[koro_mask]) if koro_mask.any() else 1.0
    return e / (peak + 1e-12)

def cusum_detect(env, t, search_on=24.0, search_off=48.0, lower=0.08, upper=0.92, w_smooth=0.0):
    mask = (t >= search_on) & (t <= search_off)
    if w_smooth > 0:
        ev = smooth_box(env[mask], w_smooth, FS)
    else:
        ev = env[mask]
    ts = t[mask]
    if len(ev) == 0 or np.max(ev) == 0: return search_on, search_off
    cs = np.cumsum(ev)
    cs = cs / cs[-1]
    i_on  = np.where(cs >= lower)[0]
    i_off = np.where(cs >= upper)[0]
    on  = float(ts[i_on[0]])  if len(i_on)  else search_on
    off = float(ts[i_off[0]]) if len(i_off) else search_off
    return on, off

SESSIONS = [
    dict(sub_dir='Sub_1_Prof_kan', label='Subject 1 (Prof. Kan) | Rec 06',
         rec=6,  k_on=27.53, k_off=43.33, notches=[100.71, 201.43, 302.14, 402.86, 50.0], lag=1.7083,
         rf_l=0.10, rf_u=0.87, mag_l=0.15, mag_u=0.80, st_l=0.01, st_u=0.98, pct=5),
    dict(sub_dir='Sub_2_Rajveer',  label='Subject 2 (Rajveer) | Rec 04',
         rec=4,  k_on=27.38, k_off=42.00, notches=[50.0, 64.0, 100.6, 201.2], lag=2.6042,
         rf_l=0.03, rf_u=0.98, mag_l=0.14, mag_u=0.81, st_l=0.01, st_u=0.90, pct=5),
]

for s in SESSIONS:
    print(f"\n{s['label']}:")
    rf_path  = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    wav_path = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")
    
    with h5py.File(rf_path, 'r') as f:
        raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc)
    ic -= xc; qc -= yc
    
    # Complex 300Hz low-pass first
    sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, ic + 1j*qc))
    
    # MAGNITUDE Velocity (30-180 Hz)
    mag_koro  = bpf(mag_raw, 30, 180, FS_RF)
    mag_vel   = np.append(np.diff(mag_koro)*FS_RF, 0.0)
    mag_env_hi = smooth_box(tkeo(mag_vel), 1.5, FS_RF)
    mag_env    = decimate(mag_env_hi, DEC, ftype='fir')
    
    # PHASE Velocity (30-180 Hz)
    phi_raw = robust_phase(ic, qc)
    phi_clean = phi_raw.copy()
    for f0 in s['notches']:
        phi_clean = notch_f(phi_clean, f0, FS_RF)
    vel_hi    = np.append(np.diff(bpf(phi_clean, 30, 180, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_env_hi = smooth_box(tkeo(vel_hi), 1.5, FS_RF)
    vel_env    = decimate(vel_env_hi, DEC, ftype='fir')
    
    t = np.arange(len(vel_env)) / FS
    
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(1)
    audio_bp    = bpf(audio, 50, 1000, fs_a)
    steth_env_a = bpf(np.abs(hilbert(audio_bp)), 20, min(200, fs_a/2-1), fs_a)
    steth_tkeo_a = smooth_box(tkeo(steth_env_a), 1.5, fs_a)
    t_a = np.arange(len(steth_tkeo_a)) / fs_a
    steth_env = np.interp(t, t_a + s['lag'], steth_tkeo_a)
    
    mag_n   = normalise(mag_env, t, s['k_on'], s['k_off'], pct=s['pct'])
    vel_n   = normalise(vel_env, t, s['k_on'], s['k_off'], pct=s['pct'])
    steth_n = normalise(steth_env, t, s['k_on'], s['k_off'], pct=s['pct'])
    
    # CUSUM detections
    st_on, st_off = cusum_detect(steth_n, t, search_on=24.0, search_off=48.0, lower=s['st_l'], upper=s['st_u'])
    mag_on, mag_off = cusum_detect(mag_n, t, search_on=24.0, search_off=48.0, lower=s['mag_l'], upper=s['mag_u'])
    vel_on, vel_off = cusum_detect(vel_n, t, search_on=24.0, search_off=48.0, lower=s['rf_l'], upper=s['rf_u'])
    
    gt_dur = s['k_off'] - s['k_on']
    print(f"  GT Duration: {gt_dur:.2f}s  ({s['k_on']:.2f} to {s['k_off']:.2f})")
    print(f"  Steth CUSUM: {st_on:.2f} to {st_off:.2f} ({st_off-st_on:.2f}s) |Err|={abs(st_off-st_on - gt_dur):.3f}s")
    print(f"  Mag CUSUM  : {mag_on:.2f} to {mag_off:.2f} ({mag_off-mag_on:.2f}s) |Err|={abs(mag_off-mag_on - gt_dur):.3f}s")
    print(f"  Vel CUSUM  : {vel_on:.2f} to {vel_off:.2f} ({vel_off-vel_on:.2f}s) |Err|={abs(vel_off-vel_on - gt_dur):.3f}s")
