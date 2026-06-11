import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt

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
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

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

SESSIONS = [
    dict(sub=1, sub_dir='Sub_1_Prof_kan', label='Subject 1',
         rec=6,  k_on=27.53, k_off=43.33, notches=[100.71, 201.43, 302.14, 402.86, 50.0]),
    dict(sub=2, sub_dir='Sub_2_Rajveer',  label='Subject 2',
         rec=4,  k_on=27.38, k_off=42.00, notches=[50.0, 64.0, 100.6, 201.2]),
]

for s in SESSIONS:
    rf_path = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
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
    mask_act = (t >= s['k_on']) & (t <= s['k_off'])
    mask_bas = (t >= 20.0) & (t <= s['k_on'] - 2.0)
    
    # Baseline subtraction (5th percentile)
    def calc_snr(env):
        b_min = np.percentile(env[mask_bas], 5)
        e_shifted = np.maximum(env - b_min, 0)
        e_norm = e_shifted / (np.max(e_shifted[mask_act]) + 1e-10)
        peak = 1.0
        noise = np.mean(e_norm[mask_bas])
        return 10 * np.log10(peak / (noise + 1e-10))
        
    snr_m = calc_snr(mag_env)
    snr_p = calc_snr(vel_env)
    
    print(f"\nSubject {s['sub']} (30-180 Hz):")
    print(f"  Magnitude SNR: {snr_m:+.1f} dB")
    print(f"  Phase Vel SNR: {snr_p:+.1f} dB")
