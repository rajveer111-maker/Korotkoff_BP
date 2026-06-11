import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000  # 1MHz / 100

def analyze():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_signal = data[0, :]
    q_signal = data[1, :]
    
    # 1. Check for saturation
    i_max = np.max(np.abs(i_signal))
    q_max = np.max(np.abs(q_signal))
    print(f"I max: {i_max:.4f}, Q max: {q_max:.4f}")
    
    # 2. Magnitude and Phase
    magnitude = np.sqrt(i_signal**2 + q_signal**2)
    phase = np.unwrap(np.angle(i_signal + 1j * q_signal))
    
    # 3. Filter for Korotkoff sounds (10-50 Hz)
    # Use a bandpass filter
    b, a = signal.butter(4, [10, 50], btype='bandpass', fs=fs)
    koro_mag = signal.filtfilt(b, a, magnitude)
    koro_ph = signal.filtfilt(b, a, phase)
    
    # 4. Energy analysis
    rms_koro = np.sqrt(np.mean(koro_mag**2))
    rms_total = np.sqrt(np.mean(magnitude**2))
    snr_approx = 20 * np.log10(rms_koro / (np.std(magnitude - np.mean(magnitude)) + 1e-6))
    
    print(f"RMS Korotkoff Band: {rms_koro:.6f}")
    print(f"Total Magnitude RMS: {rms_total:.6f}")
    
    # 5. Look for peaks in Korotkoff band
    # Simple peak detection on envelope
    peaks, _ = signal.find_peaks(np.abs(koro_mag), height=np.std(koro_mag)*3, distance=fs*0.5)
    print(f"Detected {len(peaks)} potential Korotkoff pulses.")
    
    # 6. Conclusion logic
    valid = True
    reasons = []
    
    if i_max > 0.95 or q_max > 0.95:
        valid = False
        reasons.append("Signal is SATURATED (clipping). Decrease RX gain.")
    if i_max < 0.001:
        valid = False
        reasons.append("Signal is TOO WEAK. Increase RX gain.")
    if len(peaks) < 5:
        reasons.append("Few Korotkoff pulses detected. Ensure cuff is deflating or antenna is positioned correctly.")
    
    if valid:
        print("RESULT: DATA IS VALID for Korotkoff detection.")
    else:
        print("RESULT: DATA IS INVALID.")
    
    for r in reasons:
        print(f"  - {r}")

if __name__ == '__main__':
    analyze()
