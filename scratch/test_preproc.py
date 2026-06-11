import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve

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

def get_snr_and_out_of_window_ratio(env, t, k_on, k_off):
    # koro window mask
    koro_mask = (t >= k_on) & (t <= k_off)
    # pre-deflation + post-deflation out-of-window mask
    # search window is 20 to 52 seconds
    out_mask = (t >= 20.0) & (t <= 52.0) & ~((t >= k_on - 1.0) & (t <= k_off + 1.0))
    
    peak_val = np.max(env[koro_mask]) if koro_mask.any() else 1e-10
    out_mean = np.mean(env[out_mask]) if out_mask.any() else 1e-10
    out_max  = np.max(env[out_mask]) if out_mask.any() else 1e-10
    
    snr_db = 10 * np.log10(peak_val / (out_mean + 1e-10))
    peak_to_out_max_ratio = peak_val / (out_max + 1e-10)
    
    return snr_db, peak_to_out_max_ratio

def test_preprocessing(bp_low, bp_high, use_mag_deriv):
    # Test for Subject 1
    rf_path1  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
    notches1 = [100.71, 201.43, 302.14, 402.86, 50.0]  # Add 50Hz notch to Sub 1 as well!
    
    # Test for Subject 2
    rf_path2  = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
    notches2 = [50.0, 64.0, 100.6, 201.2]
    
    results = []
    
    for sub, rf_path, notches, k_on, k_off in [
        (1, rf_path1, notches1, 27.53, 43.33),
        (2, rf_path2, notches2, 27.38, 42.00)
    ]:
        with h5py.File(rf_path, 'r') as f:
            raw = f['data'][:]
        ic, qc = -raw[0,:], raw[1,:]
        xc, yc = fit_circle(ic, qc)
        ic -= xc; qc -= yc
        
        # MAGNITUDE
        mag_raw = np.sqrt(ic**2 + qc**2)
        mag_clean = mag_raw.copy()
        for f0 in notches:
            mag_clean = notch_f(mag_clean, f0, FS_RF)
        
        if use_mag_deriv:
            # Derivative of magnitude (Magnitude Velocity)
            mag_filt = bpf(mag_clean, bp_low, bp_high, FS_RF)
            mag_vel = np.append(np.diff(mag_filt) * FS_RF, 0.0)
            mag_tkeo_env = smooth_box(tkeo(mag_vel), 1.5, FS_RF)
        else:
            mag_filt = bpf(mag_clean, bp_low, bp_high, FS_RF)
            mag_tkeo_env = smooth_box(tkeo(mag_filt), 1.5, FS_RF)
            
        mag_env = decimate(mag_tkeo_env, DEC, ftype='fir')
        
        # PHASE VELOCITY
        phi_raw = robust_phase(ic, qc)
        phi_clean = phi_raw.copy()
        for f0 in notches:
            phi_clean = notch_f(phi_clean, f0, FS_RF)
            
        vel_hi = np.append(np.diff(bpf(phi_clean, bp_low, bp_high, FS_RF)) * FS_RF, 0.0) * SCALE
        vel_tkeo_env = smooth_box(tkeo(vel_hi), 1.5, FS_RF)
        vel_env = decimate(vel_tkeo_env, DEC, ftype='fir')
        
        t = np.arange(len(vel_env)) / FS
        
        mag_snr, mag_peak_ratio = get_snr_and_out_of_window_ratio(mag_env, t, k_on, k_off)
        vel_snr, vel_peak_ratio = get_snr_and_out_of_window_ratio(vel_env, t, k_on, k_off)
        
        results.append({
            'sub': sub,
            'mag_snr': mag_snr,
            'mag_peak_ratio': mag_peak_ratio,
            'vel_snr': vel_snr,
            'vel_peak_ratio': vel_peak_ratio
        })
        
    return results

print("Starting sweeps...")
print(f"{'BP Low':<8} {'BP High':<8} {'MagDeriv':<8} | {'Sub1 MagSNR':<11} {'Sub1 MagRatio':<13} {'Sub1 VelSNR':<11} {'Sub1 VelRatio':<13} | {'Sub2 MagSNR':<11} {'Sub2 MagRatio':<13} {'Sub2 VelSNR':<11} {'Sub2 VelRatio':<13}")
print("-" * 150)

for bp_low in [10, 20, 30, 40]:
    for bp_high in [150, 180, 200]:
        for use_mag_deriv in [False, True]:
            res = test_preprocessing(bp_low, bp_high, use_mag_deriv)
            r1 = res[0]
            r2 = res[1]
            print(f"{bp_low:<8} {bp_high:<8} {str(use_mag_deriv):<8} | "
                  f"{r1['mag_snr']:>10.2f} {r1['mag_peak_ratio']:>12.2f} {r1['vel_snr']:>10.2f} {r1['vel_peak_ratio']:>12.2f} | "
                  f"{r2['mag_snr']:>10.2f} {r2['mag_peak_ratio']:>12.2f} {r2['vel_snr']:>10.2f} {r2['vel_peak_ratio']:>12.2f}")
