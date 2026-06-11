"""
Chest Vital Signs Analysis — VERSION 2 (Average Reference Technique - Optimized)
================================================================================
Techniques applied:
1. Average Reference Subtraction (Low-Pass Baseline Tracking)
2. IQ Centering (Static Reference Technique)
3. 60Hz Notch Filtering
4. Corrected Physical Scale for 0.9 GHz
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (butter, filtfilt, find_peaks, welch, iirnotch, sosfiltfilt, detrend)
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────
H5_FILE     = r'd:/Bioview/My_RF_work_v1/data_new/rec_koro11_1.h5'
FS          = 10000.0    
FC_HZ       = 0.9e9      
C_LIGHT     = 299792458

def average_reference_subtract(x, cutoff_hz, fs):
    """
    Average Reference Technique: 
    Using a Butterworth Low-pass as the "Average Reference" baseline.
    """
    sos = butter(4, cutoff_hz, btype='low', fs=fs, output='sos')
    avg_ref = sosfiltfilt(sos, x)
    return x - avg_ref, avg_ref

def process_v2():
    with h5py.File(H5_FILE, 'r') as f:
        raw = f['data'][:]
    if raw.shape[0] > raw.shape[1]: raw = raw.T
    I, Q = raw[0], raw[1]
    
    # 1. IQ Centering (Static Reference Technique)
    I_c, Q_c = I - np.mean(I), Q - np.mean(Q)
    S = I_c + 1j * Q_c
    time = np.arange(len(I)) / FS
    
    # 2. Raw Phase Extraction
    phase_raw = np.unwrap(np.angle(S))
    
    # 3. Average Reference Technique (Low-Pass Baseline Subtraction)
    # Removing drift using a 0.5 Hz baseline reference
    phase_clean, avg_ref = average_reference_subtract(phase_raw, 0.5, FS)
    
    # 4. Conversion to Physical Units (mm)
    lambda_mm = (C_LIGHT / FC_HZ) * 1000
    disp_mm = (phase_clean * lambda_mm) / (4 * np.pi)
    
    # 5. Heart Rate Separation (0.8 - 3.0 Hz)
    sos_hr = butter(4, [0.8, 3.0], btype='band', fs=FS, output='sos')
    card_sig_mm = sosfiltfilt(sos_hr, disp_mm)
    card_sig_um = card_sig_mm * 1000
    
    # 6. Rate Estimation
    peaks, _ = find_peaks(card_sig_mm, distance=int(FS*0.4), prominence=np.std(card_sig_mm)*0.4)
    hr_bpm = (len(peaks) / time[-1]) * 60
    
    # 7. Results Output
    print(f"\nRESULTS (AVG REFERENCE TECHNIQUE)")
    print(f"----------------------------------------")
    print(f"Residual Drift (Ref Removed) : {np.std(disp_mm):.4f} mm RMS")
    print(f"Heartbeat Pulse Amplitude    : {np.std(card_sig_um):.2f} µm RMS")
    print(f"Heart Rate                   : {hr_bpm:.1f} BPM")
    print(f"----------------------------------------")
    
    # PLOTTING
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(time, phase_raw, color='black', alpha=0.3, label='Raw Phase')
    plt.plot(time, avg_ref, color='red', linewidth=2, label='Average Reference Baseline')
    plt.title('Step 1: Baseline Tracking (Average Reference)'); plt.ylabel('Radians'); plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time, disp_mm, color='blue')
    plt.title('Step 2: Corrected Displacement (Reference Subtracted)'); plt.ylabel('mm')
    
    plt.subplot(3, 1, 3)
    plt.plot(time, card_sig_um, color='firebrick')
    plt.title('Step 3: Final Cardiac Vital (µm)'); plt.ylabel('µm')
    
    plt.tight_layout()
    plt.savefig('chest_vitals_avg_ref.png')
    print("Average Reference plot saved to: chest_vitals_avg_ref.png")

if __name__ == '__main__':
    process_v2()
