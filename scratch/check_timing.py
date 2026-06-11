import os, sys
import numpy as np
import pandas as pd
import h5py
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.io import wavfile

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
FS = 10000

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w, center=True).mean().fillna(0).values)

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def main():
    print("Loading RF signal...")
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0, :], data[1, :]
    t_rf = np.arange(len(i_raw)) / FS
    
    print("Applying Decoupled Phase Reconstruction...")
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    idx_deflation = int(20.0 * FS)
    
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
    phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(FS*1.0), center=True).mean().bfill().ffill().values
    shift = phase_clean_def[0] - phase_clean_inf[-1]
    phase_clean_inf = phase_clean_inf + shift
    
    phase_clean = np.zeros(len(iq))
    phase_clean[:idx_deflation] = phase_clean_inf
    phase_clean[idx_deflation:] = phase_clean_def
    
    LAMBDA_MM = (299792458 / 0.9e9) * 1000
    SCALE = LAMBDA_MM / (4 * np.pi)
    
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)
    disp_koro = phase_koro * SCALE * 1000
    vel_koro = np.append(np.diff(disp_koro) * FS / 1000, 0)
    
    ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
    T_SKIP = 5
    vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS))
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    rf_on = t_rf[max(si, int(T_SKIP*FS))]
    rf_off = t_rf[min(ei, int((t_rf[-1]-T_SKIP)*FS))]
    
    print(f"Dynamically detected RF Window: {rf_on:.2f}s - {rf_off:.2f}s (dur: {rf_off - rf_on:.2f}s)")
    
    print("\nLoading Stethoscope audio...")
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # Let's check different filter bands for audio
    for band in [[20, 200], [80, 200], [100, 250]]:
        sos_k = butter(4, band, btype='band', fs=fs_aud, output='sos')
        koro_aud = sosfiltfilt(sos_k, audio)
        
        aud_env = np.abs(hilbert(koro_aud))
        aud_env_sm = smooth(aud_env, int(fs_aud * 0.5))
        
        # Find index of max in audio deflation window (after 20s)
        idx_defl = t_aud >= 20.0
        t_defl = t_aud[idx_defl]
        env_defl = aud_env_sm[idx_defl]
        
        max_idx = np.argmax(env_defl)
        max_time = t_defl[max_idx]
        max_val = env_defl[max_idx]
        
        # Check other peaks
        peaks, properties = signal.find_peaks(env_defl, distance=int(fs_aud*0.5), prominence=max_val*0.05)
        print(f"\nFilter Band: {band[0]}-{band[1]} Hz")
        print(f"  Absolute maximum after 20s is at {max_time:.2f}s with value {max_val:.2e}")
        print("  Prominent envelope peaks after 20s:")
        for p in peaks:
            print(f"    Peak at {t_defl[p]:.2f}s, height: {env_defl[p]:.2e}")

if __name__ == '__main__':
    main()
