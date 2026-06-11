import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
FS_RF = 10000

def load_rf(path):
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
    ir, qr = data[0, :], data[1, :]
    fs = FS_RF
    t = np.arange(len(ir)) / fs
    
    # simple IQ conditioning and phase unwrapping
    ic = ir - ir.mean()
    qc = qr - qr.mean()
    iq = ic + 1j * qc
    phase = np.unwrap(np.angle(iq))
    
    # Korotkoff velocity (10-49 Hz)
    sos_k = butter(4, [10, 49], btype='band', fs=fs, output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vel_koro = np.append(np.diff(pk) * fs, 0)
    return t, vel_koro, fs

def find_robust_stethoscope_window_adaptive(koro_aud, t_aud, fs_aud, max_search_s=34.0):
    aud_env = np.abs(hilbert(koro_aud))
    min_onset_s = 20.0
    
    idx = (t_aud >= min_onset_s) & (t_aud <= max_search_s)
    if not np.any(idx):
        return min_onset_s, min_onset_s + 5.0, None, None
        
    t_defl = t_aud[idx]
    env_defl = aud_env[idx]
    
    # Detect peaks with low prominence to capture all clicks
    peaks_idx, _ = find_peaks(env_defl, distance=int(fs_aud * 0.4), prominence=0.005)
    if len(peaks_idx) < 2:
        return min_onset_s, min_onset_s + 5.0, None, None
        
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    # Adaptive thresholds
    max_all_h = np.max(peaks_h)
    median_h = np.median(peaks_h)
    
    upper_th = 0.40 * max_all_h
    lower_th = 1.5 * median_h
    
    upper_th = max(0.25, min(upper_th, 0.6))
    lower_th = max(0.04, min(lower_th, 0.15))
    
    valid_idx = (peaks_h >= lower_th) & (peaks_h <= upper_th)
    filtered_t = peaks_t[valid_idx]
    filtered_h = peaks_h[valid_idx]
    
    n_peaks = len(filtered_t)
    best_seq = []
    
    for i in range(n_peaks):
        seq = [i]
        curr = i
        for j in range(i + 1, n_peaks):
            diff = filtered_t[j] - filtered_t[curr]
            if 0.65 <= diff <= 1.35:
                seq.append(j)
                curr = j
        if len(seq) > len(best_seq):
            best_seq = seq
            
    if len(best_seq) >= 2:
        st_on = filtered_t[best_seq[0]]
        st_off = filtered_t[best_seq[-1]]
        st_on = max(min_onset_s, st_on - 0.3)
        st_off = min(t_aud[-1], st_off + 0.3)
    else:
        if len(filtered_t) > 0:
            p_max = filtered_t[np.argmax(filtered_h)]
            st_on = max(min_onset_s, p_max - 1.5)
            st_off = min(t_aud[-1], p_max + 1.5)
        else:
            st_on, st_off = 23.0, 27.0
            
    return st_on, st_off, filtered_t, filtered_h

def main():
    t_rf, vel, fs_rf = load_rf(RF_PATH)
    
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    # 1. Compute lag
    rf_env_full = np.convolve(vel**2, np.ones(int(fs_rf * 2.0))/(fs_rf * 2.0), mode='same')
    rf_env_n = rf_env_full / (np.max(rf_env_full) + 1e-20)
    
    aud_env_full = np.convolve(koro_aud**2, np.ones(int(fs_aud * 2.0))/(fs_aud * 2.0), mode='same')
    aud_env_n = aud_env_full / (np.max(aud_env_full) + 1e-20)
    aud_env_rf = np.interp(t_rf, t_aud, aud_env_n)
    
    s_idx, e_idx = int(5 * fs_rf), min(int(50 * fs_rf), len(rf_env_n))
    cc = np.correlate(rf_env_n[s_idx:e_idx], aud_env_rf[s_idx:e_idx], mode='full')
    lag_samples = np.argmax(cc) - len(rf_env_n[s_idx:e_idx]) + 1
    lag_sec = lag_samples / fs_rf
    
    # 2. Get true stethoscope window restricted to overlapping region
    max_search_s = max(25.0, t_rf[-1] - lag_sec)
    
    st_on, st_off, filt_t, filt_h = find_robust_stethoscope_window_adaptive(
        koro_aud, t_aud, fs_aud, max_search_s=max_search_s
    )
    
    print("=" * 60)
    print("CROSS-CORRELATION LAG & DYNAMIC WINDOW ANALYSIS")
    print("=" * 60)
    print(f"RF signal duration (t_rf[-1]): {t_rf[-1]:.2f}s")
    print(f"Computed lag_sec: {lag_sec:.2f}s")
    print(f"Computed max_search_s (t_rf[-1] - lag_sec): {max_search_s:.2f}s")
    print(f"Steth window with max_search_s={max_search_s:.2f}s:")
    print(f"  Onset: {st_on:.2f}s, Offset: {st_off:.2f}s, Duration: {st_off - st_on:.2f}s")
    print(f"Filtered peaks within max_search_s={max_search_s:.2f}s:")
    if filt_t is not None:
        for i in range(len(filt_t)):
            print(f"  Peak {i+1}: time = {filt_t[i]:.2f}s, height = {filt_h[i]:.4f}")
            
    # Let's also check what happens if we search up to a lower limit, or what are the peaks in [20, 30]
    st_on_30, st_off_30, filt_t_30, filt_h_30 = find_robust_stethoscope_window_adaptive(
        koro_aud, t_aud, fs_aud, max_search_s=29.0
    )
    print(f"\nSteth window with max_search_s=29.0s:")
    print(f"  Onset: {st_on_30:.2f}s, Offset: {st_off_30:.2f}s, Duration: {st_off_30 - st_on_30:.2f}s")
    print(f"Filtered peaks within max_search_s=29.0s:")
    if filt_t_30 is not None:
        for i in range(len(filt_t_30)):
            print(f"  Peak {i+1}: time = {filt_t_30[i]:.2f}s, height = {filt_h_30[i]:.4f}")

if __name__ == '__main__':
    main()
