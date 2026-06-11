import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
FS_RF = 10000

def main():
    # 1. Load RF
    print("Loading RF signal...")
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
    ir, qr = data[0, :], data[1, :]
    t_rf = np.arange(len(ir)) / FS_RF
    
    # Preprocess RF
    ic = ir - ir.mean()
    qc = qr - qr.mean()
    iq = ic + 1j * qc
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_clean = sosfiltfilt(sos_lp, iq)
    idx_defl = int(20.0 * FS_RF)
    
    phase_unwrap = np.unwrap(np.angle(iq_clean[idx_defl:]))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.median(dphi)
    dphi_clean = dphi - carrier_offset
    dphi_clean = np.clip(dphi_clean, -0.5, 0.5)
    phase_clean_def = np.insert(np.cumsum(dphi_clean), 0, 0)
    
    t_idx = np.arange(len(phase_clean_def))
    t_norm = (t_idx - np.mean(t_idx)) / (np.max(t_idx) - np.min(t_idx) + 1e-9)
    poly = np.polyfit(t_norm, phase_clean_def, 2)
    phase_clean_def = phase_clean_def - np.polyval(poly, t_norm)
    
    phase_clean_inf = np.angle(iq_clean[:idx_defl])
    phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(FS_RF*1.0), center=True).mean().bfill().ffill().values
    shift = phase_clean_def[0] - phase_clean_inf[-1]
    phase_clean_inf = phase_clean_inf + shift
    
    phase_clean = np.zeros(len(iq))
    phase_clean[:idx_defl] = phase_clean_inf
    phase_clean[idx_defl:] = phase_clean_def
    
    LAMBDA_MM = (299792458 / 0.9e9) * 1000
    SCALE = LAMBDA_MM / (4 * np.pi)
    
    # 2. Extract RF Heartbeat Peaks
    # Filter 0.5 - 3.0 Hz
    sos_hr = butter(4, [0.5, 3.0], btype='band', fs=FS_RF, output='sos')
    rf_hr = sosfiltfilt(sos_hr, phase_clean)
    peaks_rf, _ = signal.find_peaks(rf_hr, distance=int(FS_RF*0.5), prominence=np.std(rf_hr)*0.4)
    rf_peaks_t = peaks_rf / FS_RF
    
    # Only keep peaks during deflation (after 20s)
    rf_peaks_t = rf_peaks_t[rf_peaks_t >= 20.0]
    print(f"Extracted {len(rf_peaks_t)} RF heartbeat peaks in deflation period.")
    
    # 3. Load and Filter Stethoscope Audio
    print("Loading Stethoscope Audio...")
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # Filter Audio in 80-200 Hz band
    sos_k_aud = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k_aud, audio)
    aud_env = np.abs(hilbert(koro_aud))
    
    # Find all audio envelope peaks after 20s with low prominence (capture all clicks)
    idx_defl_aud = t_aud >= 20.0
    t_defl_aud = t_aud[idx_defl_aud]
    env_defl_aud = aud_env[idx_defl_aud]
    
    peaks_aud_idx, _ = signal.find_peaks(env_defl_aud, distance=int(fs_aud*0.4), prominence=0.005)
    aud_peaks_t = t_defl_aud[peaks_aud_idx]
    aud_peaks_h = env_defl_aud[peaks_aud_idx]
    print(f"Extracted {len(aud_peaks_t)} Audio peak candidates in deflation period.")
    
    # 4. Search for the best alignment (lag) between RF peaks and Audio peaks
    # Let's test lags from -10s to +10s in steps of 0.01s
    lags = np.arange(-15.0, 15.0, 0.02)
    best_lag = 0
    max_coincident = 0
    best_matching_pairs = []
    
    for lag in lags:
        # Shift audio peaks: rf_t_est = aud_t - lag
        rf_est = aud_peaks_t - lag
        
        # Count how many shifted audio peaks align with an RF peak within a tolerance of 0.15s
        coincident = 0
        pairs = []
        for i, ae in enumerate(rf_est):
            # Find closest RF peak
            closest_idx = np.argmin(np.abs(rf_peaks_t - ae))
            diff = np.abs(rf_peaks_t[closest_idx] - ae)
            if diff <= 0.15:
                coincident += 1
                pairs.append((aud_peaks_t[i], rf_peaks_t[closest_idx], diff))
        
        if coincident > max_coincident:
            max_coincident = coincident
            best_lag = lag
            best_matching_pairs = pairs
            
    print(f"\nBest beat-matching lag found: {best_lag:.2f} seconds")
    print(f"Number of aligned beats: {max_coincident} out of {len(aud_peaks_t)}")
    
    print("\nAligned Beat Pairs (Steth Time -> Shifted Time -> RF Time):")
    for ap, rp, d in best_matching_pairs:
        shifted_t = ap - best_lag
        print(f"  Steth: {ap:.3f}s  -->  Shifted: {shifted_t:.3f}s  -->  RF: {rp:.3f}s  (diff: {d:.3f}s)")
        
if __name__ == '__main__':
    main()
