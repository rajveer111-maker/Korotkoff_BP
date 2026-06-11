"""
Analyze all 10 stethoscope wav files for Sub_1_Prof_kan using the quiet baseline window (22s to 27s).
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
        continue
    
    fs_a, aud = wavfile.read(wp)
    aud = aud.astype(np.float64) / 32768.0
    if aud.ndim > 1: aud = aud.mean(1)
    
    # Process
    st_bp = bpf(aud, 30, 1000, fs_a)
    st_hilb = np.abs(hilbert(st_bp))
    
    t = np.arange(len(st_hilb)) / fs_a
    mask_active = (t >= 31.0) & (t <= 37.0)
    mask_quiet  = (t >= 22.0) & (t <= 27.0)
    
    act_mean = np.mean(st_hilb[mask_active])
    quiet_mean = np.mean(st_hilb[mask_quiet])
    ratio = act_mean / (quiet_mean + 1e-12)
    
    print(f"Session {i}: Active Mean={act_mean:.5f}, Quiet Mean={quiet_mean:.5f}, Ratio={ratio:.2f}")
