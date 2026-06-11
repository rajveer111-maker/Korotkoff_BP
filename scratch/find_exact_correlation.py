import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.io import wavfile

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
FS_RF = 10000

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w, center=True).mean().fillna(0).values)

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
    
    # Filter RF Korotkoff (10-200 Hz)
    sos_k = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
    phase_k = sosfiltfilt(sos_k, phase_clean)
    vel_koro = np.append(np.diff(phase_k) * FS_RF, 0) * SCALE
    
    # 2. Load Audio
    print("Loading Stethoscope Audio...")
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # Filter Audio in 80-200 Hz band
    sos_k_aud = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k_aud, audio)
    
    # 3. Compute envelopes at 1000 Hz
    fs_down = 1000
    
    # Downsample RF velocity energy
    vel_env_full = vel_koro**2
    down_rf = max(1, int(FS_RF / fs_down))
    vel_env_down = vel_env_full[::down_rf]
    t_rf_down = t_rf[::down_rf]
    
    # Downsample Audio energy
    aud_env_full = koro_aud**2
    down_aud = max(1, int(fs_aud / fs_down))
    aud_env_down = aud_env_full[::down_aud]
    t_aud_down = t_aud[::down_aud]
    
    # Interpolate Audio energy to RF timeline to compare
    aud_env_rf = np.interp(t_rf_down, t_aud_down, aud_env_down)
    
    # Smooth envelopes with a 1-second window
    rf_env_sm = smooth(vel_env_down, int(fs_down * 1.5))
    aud_env_sm = smooth(aud_env_rf, int(fs_down * 1.5))
    
    # Normalize
    rf_env_sm /= (np.max(rf_env_sm) + 1e-20)
    aud_env_sm /= (np.max(aud_env_sm) + 1e-20)
    
    # Let's perform cross-correlation of envelopes between 20s and 50s
    s_idx = int(20 * fs_down)
    e_idx = min(int(50 * fs_down), len(rf_env_sm))
    
    rf_seg = rf_env_sm[s_idx:e_idx]
    aud_seg = aud_env_sm[s_idx:e_idx]
    
    cc = np.correlate(rf_seg, aud_seg, mode='full')
    lags = np.arange(-len(rf_seg) + 1, len(rf_seg))
    best_lag_samples = lags[np.argmax(cc)]
    best_lag_sec = best_lag_samples / fs_down
    
    print(f"\nEnvelope cross-correlation lag between 20s and 50s: {best_lag_sec:.2f} seconds")
    
    # Now let's find the heartbeat rate in the RF signal and Audio signal
    # RF HR: Filter 0.5 - 3.0 Hz
    sos_hr = butter(4, [0.5, 3.0], btype='band', fs=FS_RF, output='sos')
    rf_hr = sosfiltfilt(sos_hr, phase_clean)
    
    # Let's count heartbeat peaks in RF between 20s and 30s
    rf_hr_focus = rf_hr[int(20*FS_RF):int(30*FS_RF)]
    peaks_rf, _ = signal.find_peaks(rf_hr_focus, distance=int(FS_RF*0.5), prominence=np.std(rf_hr_focus)*0.5)
    rf_hr_t = 20.0 + peaks_rf / FS_RF
    
    print("\nRF Heartbeat peaks between 20s and 30s:")
    for pt in rf_hr_t:
        print(f"  t={pt:.3f}s")
    if len(rf_hr_t) >= 2:
        print(f"  Average RF HR interval: {np.mean(np.diff(rf_hr_t)):.3f}s (HR: {60.0/np.mean(np.diff(rf_hr_t)):.1f} bpm)")
        
    # Audio clicks between 24s and 27s (where we saw the periodic clicks)
    aud_koro_peaks = np.array([24.410, 25.517, 26.558])
    print(f"\nStethoscope Clicks in 24s-27s: {aud_koro_peaks}")
    print(f"  Average Audio click interval: {np.mean(np.diff(aud_koro_peaks)):.3f}s (HR: {60.0/np.mean(np.diff(aud_koro_peaks)):.1f} bpm)")

if __name__ == '__main__':
    main()
