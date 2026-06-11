import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch, decimate
import h5py

SUB1_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_6.h5"
SUB2_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\Rec_4.h5"
SCALE = (299792458.0 / 0.9e9 * 1000.0) / (4.0 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
    return tkeo

def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def test_preprocessing(rf_path, notches, k_on, k_off, defl):
    with h5py.File(rf_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = -data[0, :], data[1, :]
    
    A = np.column_stack([i_raw, q_raw, np.ones_like(i_raw)])
    B = -(i_raw**2 + q_raw**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    i_c = i_raw - xc
    q_c = q_raw - yc
    
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    phi_raw = signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')
    
    fs_rf = 10000
    sos_lp = butter(4, 300.0, btype='low', fs=fs_rf, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    
    # 1. Old preprocessing (30-180 Hz, notches only on Phase, Magnitude unnotched)
    phi_clean_old = phi_raw.copy()
    for freq in notches:
        phi_clean_old = notch(phi_clean_old, freq, fs_rf)
    
    mag_vel_old = np.append(np.diff(bpf(mag_raw, 30, 180, fs_rf))*fs_rf, 0.0)
    phi_vel_old = np.append(np.diff(bpf(phi_clean_old, 30, 180, fs_rf))*fs_rf, 0.0)
    
    # 2. New preprocessing (30-180 Hz, notches on BOTH Phase and Magnitude)
    phi_clean_new = phi_raw.copy()
    mag_clean_new = mag_raw.copy()
    for freq in notches:
        phi_clean_new = notch(phi_clean_new, freq, fs_rf)
        mag_clean_new = notch(mag_clean_new, freq, fs_rf)
        
    mag_vel_new = np.append(np.diff(bpf(mag_clean_new, 30, 180, fs_rf))*fs_rf, 0.0)
    phi_vel_new = np.append(np.diff(bpf(phi_clean_new, 30, 180, fs_rf))*fs_rf, 0.0)
    
    # Zero outside clean deflation window
    t_rf = np.arange(len(mag_vel_old)) / fs_rf
    t_start = defl + 3.0
    t_end = k_off + 1.2
    
    mag_vel_old[(t_rf < t_start) | (t_rf > t_end)] = 0.0
    phi_vel_old[(t_rf < t_start) | (t_rf > t_end)] = 0.0
    mag_vel_new[(t_rf < t_start) | (t_rf > t_end)] = 0.0
    phi_vel_new[(t_rf < t_start) | (t_rf > t_end)] = 0.0
    
    # Calculate SNR
    mag_tkeo_old = smooth_energy(calc_tkeo(mag_vel_old), 1.5, fs_rf)
    phi_tkeo_old = smooth_energy(calc_tkeo(phi_vel_old), 1.5, fs_rf)
    mag_tkeo_new = smooth_energy(calc_tkeo(mag_vel_new), 1.5, fs_rf)
    phi_tkeo_new = smooth_energy(calc_tkeo(phi_vel_new), 1.5, fs_rf)
    
    # Decimate to 1000 Hz for SNR calc
    mag_tkeo_old_ds = decimate(mag_tkeo_old, 10, ftype='fir')
    phi_tkeo_old_ds = decimate(phi_tkeo_old, 10, ftype='fir')
    mag_tkeo_new_ds = decimate(mag_tkeo_new, 10, ftype='fir')
    phi_tkeo_new_ds = decimate(phi_tkeo_new, 10, ftype='fir')
    
    t_ds = np.arange(len(mag_tkeo_old_ds)) / 1000.0
    mask_act = (t_ds >= k_on) & (t_ds <= k_off)
    mask_bas = (t_ds >= 22.0) & (t_ds <= k_on - 2.0)
    
    def get_snr(env):
        b_min = np.percentile(env[mask_bas], 5)
        e_shifted = np.maximum(env - b_min, 0)
        peak = np.max(e_shifted[mask_act])
        noise = np.mean(e_shifted[mask_bas])
        return 10 * np.log10(peak / (noise + 1e-10))
        
    print(f"Old Mag SNR: {get_snr(mag_tkeo_old_ds):+.2f} dB | New Mag SNR: {get_snr(mag_tkeo_new_ds):+.2f} dB")
    print(f"Old Phi SNR: {get_snr(phi_tkeo_old_ds):+.2f} dB | New Phi SNR: {get_snr(phi_tkeo_new_ds):+.2f} dB")

print("Subject 1:")
test_preprocessing(SUB1_RF, [100.71, 201.43, 302.14, 402.86], 27.53, 43.33, 18.0)

print("\nSubject 2:")
test_preprocessing(SUB2_RF, [50.0, 64.0, 100.6, 201.2], 27.38, 42.00, 18.6)
