"""
Analyze all 10 stethoscope wav files for Sub_1_Prof_kan to identify
which ones have a clear, valid Korotkoff energy hump in the active region.
"""
import os, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan'

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

for i in range(1, 11):
    wp = os.path.join(BASE, f'sthethoscope_rec{i:02d}.wav')
    if not os.path.exists(wp):
        wp = os.path.join(BASE, f'sthethoscope_rec{i}.wav')
    if not os.path.exists(wp):
        print(f"Session {i}: wav file not found")
        continue
    
    fs_a, aud = wavfile.read(wp)
    aud = aud.astype(np.float64) / 32768.0
    if aud.ndim > 1: aud = aud.mean(1)
    
    # Process
    st_bp = bpf(aud, 30, 1000, fs_a)
    st_hilb = np.abs(hilbert(st_bp))
    
    # Calculate energy in active window (25s - 40s) vs quiet baseline (18s - 22s)
    t = np.arange(len(st_hilb)) / fs_a
    mask_active = (t >= 28.0) & (t <= 38.0)
    mask_noise = (t >= 18.0) & (t <= 22.0)
    
    if not np.any(mask_active) or not np.any(mask_noise):
        print(f"Session {i}: timeline issues")
        continue
        
    act_mean = np.mean(st_hilb[mask_active])
    noise_mean = np.mean(st_hilb[mask_noise])
    snr = act_mean / (noise_mean + 1e-12)
    
    print(f"Session {i}: Active Mean={act_mean:.5f}, Noise Mean={noise_mean:.5f}, Ratio (SNR)={snr:.2f}")
