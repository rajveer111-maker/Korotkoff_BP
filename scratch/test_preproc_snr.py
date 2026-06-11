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
    
    # MAGNITUDE Velocity (40-180 Hz)
    mag_raw = np.sqrt(ic**2 + qc**2)
    mag_clean = mag_raw.copy()
    for f0 in s['notches']:
        mag_clean = notch_f(mag_clean, f0, FS_RF)
    mag_koro  = bpf(mag_clean, 40, 180, FS_RF)
    mag_vel   = np.append(np.diff(mag_koro)*FS_RF, 0.0)
    mag_env_hi = smooth_box(tkeo(mag_vel), 1.5, FS_RF)
    mag_env    = decimate(mag_env_hi, DEC, ftype='fir')
    
    # PHASE Velocity (40-180 Hz)
    phi_raw = robust_phase(ic, qc)
    phi_clean = phi_raw.copy()
    for f0 in s['notches']:
        phi_clean = notch_f(phi_clean, f0, FS_RF)
    vel_hi    = np.append(np.diff(bpf(phi_clean, 40, 180, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_env_hi = smooth_box(tkeo(vel_hi), 1.5, FS_RF)
    vel_env    = decimate(vel_env_hi, DEC, ftype='fir')
    
    t = np.arange(len(vel_env)) / FS
    mask_act = (t >= s['k_on']) & (t <= s['k_off'])
    mask_bas = (t >= 20.0) & (t <= s['k_on'] - 2.0)
    
    # Magnitude SNR
    peak_m = np.max(mag_env[mask_act])
    noise_m = np.mean(mag_env[mask_bas])
    snr_m = 10 * np.log10(peak_m / (noise_m + 1e-10))
    
    # Phase SNR
    peak_p = np.max(vel_env[mask_act])
    noise_p = np.mean(vel_env[mask_bas])
    snr_p = 10 * np.log10(peak_p / (noise_p + 1e-10))
    
    print(f"\nSubject {s['sub']}:")
    print(f"  Magnitude SNR: {snr_m:+.1f} dB (vs original +5.7/+5.8 dB)")
    print(f"  Phase Vel SNR: {snr_p:+.1f} dB (vs original +6.4/+4.8 dB)")
