"""
Chest Vital Signs Analysis — UPDATED 0.9 GHz VERSION
=====================================================
Fixes applied:
1. Corrected Cardiac Displacement calculation (shows microns instead of mm drift).
2. Separated "Total Chest Motion" from "Heartbeat Pulse".
3. Added 60 Hz Notch and IQ Centering for high stability.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (butter, filtfilt, find_peaks, welch, iirnotch, sosfiltfilt)
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────
H5_FILE     = r'd:/Bioview/My_RF_work_v1/data_new/rec_koro11_1.h5'
FS          = 10000.0    
FC_HZ       = 0.9e9      
C_LIGHT     = 299792458

def bpf(x, lo, hi, fs=FS, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def hpf(x, cutoff, fs=FS, order=3):
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch_filter(x, freq, fs=FS):
    b, a = iirnotch(freq, 30, fs)
    return filtfilt(b, a, x)

def load_data(path):
    with h5py.File(path, 'r') as f:
        raw = f['data'][:]
    if raw.shape[0] > raw.shape[1]: raw = raw.T
    I, Q = raw[0], raw[1]
    # Notch and Center
    I = notch_filter(I, 60, FS)
    Q = notch_filter(Q, 60, FS)
    I_c, Q_c = I - np.mean(I), Q - np.mean(Q)
    return I_c + 1j * Q_c

def process():
    S = load_data(H5_FILE)
    time = np.arange(len(S)) / FS
    
    # 1. Total Displacement (Physical Units)
    phase = np.unwrap(np.angle(S))
    # λ = c/f, d = (phi * λ) / (4*pi)
    total_disp_mm = (phase * (C_LIGHT / FC_HZ) * 1000) / (4 * np.pi)
    total_disp_mm = detrend(total_disp_mm)
    
    # 2. Respiration & Cardiac Bands
    resp_sig_mm = bpf(total_disp_mm, 0.1, 0.5, FS)
    card_sig_mm = bpf(total_disp_mm, 0.8, 3.0, FS)
    
    # 3. Conversion to Microns (µm) for the Heart
    card_sig_um = card_sig_mm * 1000
    
    # 4. Rate Estimation
    # Heart Rate
    min_dist = int(FS * 0.4)
    pks, _ = find_peaks(card_sig_mm, distance=min_dist, prominence=np.std(card_sig_mm)*0.4)
    hr_bpm = (len(pks) / time[-1]) * 60
    
    # 5. Output
    print(f"\nRESULTS FOR {Path(H5_FILE).name}")
    print(f"----------------------------------------")
    print(f"Total Chest Motion (Drift) : {np.std(total_disp_mm):.2f} mm RMS")
    print(f"Heartbeat Pulse Amplitude  : {np.std(card_sig_um):.2f} µm RMS  <-- THIS IS THE REAL VITAL")
    print(f"Heart Rate                 : {hr_bpm:.1f} BPM")
    print(f"----------------------------------------")
    
    # Plotting
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(time, total_disp_mm, color='black', alpha=0.5)
    plt.title('Total Chest Motion (Includes Breathing/Drift)'); plt.ylabel('mm')
    
    plt.subplot(3, 1, 2)
    plt.plot(time, card_sig_um, color='firebrick')
    plt.title('Filtered Heartbeat Signal (Corrected Units)'); plt.ylabel('µm')
    
    plt.subplot(3, 1, 3)
    f, p = welch(card_sig_um, FS, nperseg=int(FS*10))
    plt.semilogy(f, p, color='brown'); plt.xlim(0, 5)
    plt.title('Heart Rate Spectral Peak'); plt.xlabel('Hz'); plt.ylabel('µm²/Hz')
    
    plt.tight_layout()
    plt.savefig('chest_vitals_corrected.png')
    print("Corrected plot saved to: chest_vitals_corrected.png")

if __name__ == '__main__':
    from scipy.signal import detrend
    process()
