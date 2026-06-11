import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert
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
    
    ic = ir - ir.mean()
    qc = qr - qr.mean()
    iq = ic + 1j * qc
    
    sos_lp = signal.butter(4, 50.0, btype='low', fs=fs, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    
    idx_deflation = int(20.0 * fs)
    phase_unwrap_def = np.unwrap(np.angle(iq_clean[idx_deflation:]))
    dphi_def = np.diff(phase_unwrap_def)
    carrier_offset = np.median(dphi_def)
    dphi_clean_def = dphi_def - carrier_offset
    dphi_clean_def = np.clip(dphi_clean_def, -0.5, 0.5)
    phase_clean_def = np.insert(np.cumsum(dphi_clean_def), 0, 0)
    
    t_idx = np.arange(len(phase_clean_def))
    t_norm = (t_idx - np.mean(t_idx)) / (np.max(t_idx) - np.min(t_idx) + 1e-9)
    poly_def = np.polyfit(t_norm, phase_clean_def, 2)
    phase_clean_def = phase_clean_def - np.polyval(poly_def, t_norm)
    
    phase_clean_inf = np.angle(iq_clean[:idx_deflation])
    phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(fs*1.0), center=True).mean().bfill().ffill().values
    shift = phase_clean_def[0] - phase_clean_inf[-1]
    phase_clean_inf = phase_clean_inf + shift
    
    phase_clean = np.zeros(len(iq))
    phase_clean[:idx_deflation] = phase_clean_inf
    phase_clean[idx_deflation:] = phase_clean_def
    
    LAMBDA_MM = (299792458 / 0.9e9) * 1000
    SCALE = LAMBDA_MM / (4 * np.pi)
    
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=fs, output='sos')
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)
    vel_koro = np.append(np.diff(phase_koro * SCALE * 1000) * fs / 1000, 0)
    
    return t, vel_koro, fs

def main():
    t_rf, vel, fs_rf = load_rf(RF_PATH)
    
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    rf_env_full = np.convolve(vel**2, np.ones(int(fs_rf * 1.0))/(fs_rf * 1.0), mode='same')
    rf_env_n = rf_env_full / (np.max(rf_env_full) + 1e-20)
    
    aud_env_full = np.convolve(koro_aud**2, np.ones(int(fs_aud * 1.0))/(fs_aud * 1.0), mode='same')
    aud_env_n = aud_env_full / (np.max(aud_env_full) + 1e-20)
    aud_env_rf = np.interp(t_rf, t_aud, aud_env_n)
    
    # Method 1: Wide correlation (5s to 50s)
    s1, e1 = int(5 * fs_rf), min(int(50 * fs_rf), len(rf_env_n))
    cc1 = np.correlate(rf_env_n[s1:e1], aud_env_rf[s1:e1], mode='full')
    lag_samples1 = np.argmax(cc1) - len(rf_env_n[s1:e1]) + 1
    lag_sec1 = lag_samples1 / fs_rf
    
    # Method 2: Deflation startup correlation (18s to 25s)
    s2, e2 = int(18 * fs_rf), min(int(25 * fs_rf), len(rf_env_n))
    cc2 = np.correlate(rf_env_n[s2:e2], aud_env_rf[s2:e2], mode='full')
    lag_samples2 = np.argmax(cc2) - len(rf_env_n[s2:e2]) + 1
    lag_sec2 = lag_samples2 / fs_rf
    
    print("=" * 60)
    print("LAG ANALYSIS COMPARISON")
    print("=" * 60)
    print(f"Method 1: Wide Correlation (5s-50s) Lag = {lag_sec1:.4f}s")
    print(f"Method 2: Startup Correlation (18s-25s) Lag = {lag_sec2:.4f}s")
    
    # Let's inspect where the peak energy is located in both envelopes in [18s, 25s]
    rf_peak_idx = s2 + np.argmax(rf_env_n[s2:e2])
    aud_peak_idx = int((s2 / fs_rf) * fs_aud) + np.argmax(aud_env_n[int((s2 / fs_rf) * fs_aud):int((e2 / fs_rf) * fs_aud)])
    rf_peak_t = t_rf[rf_peak_idx]
    aud_peak_t = t_aud[aud_peak_idx]
    
    print(f"\nDeflation startup peak times:")
    print(f"  RF envelope peak: {rf_peak_t:.4f}s")
    print(f"  Stethoscope envelope peak: {aud_peak_t:.4f}s")
    print(f"  Direct peak difference (RF - Steth): {rf_peak_t - aud_peak_t:.4f}s")

if __name__ == '__main__':
    main()
