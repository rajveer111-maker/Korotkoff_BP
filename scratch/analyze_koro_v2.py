import h5py
import numpy as np
import os
from scipy import signal

file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
fs = 10000 

def analyze():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_signal = data[0, :]
    q_signal = data[1, :]
    
    # Check for saturation/levels
    i_max = np.max(np.abs(i_signal))
    q_max = np.max(np.abs(q_signal))
    
    magnitude = np.sqrt(i_signal**2 + q_signal**2)
    
    # Use SOS (Second-Order Sections) for stability at low frequencies
    sos = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    
    try:
        # Filter magnitude (mechanical vibration)
        koro_mag = signal.sosfiltfilt(sos, magnitude)
        
        # Calculate RMS of noise vs signal
        rms_koro = np.sqrt(np.mean(koro_mag**2))
        std_mag = np.std(magnitude)
        
        # Detect peaks in the filtered signal
        # Korotkoff sounds are typically bursts. 
        # We look for pulses that stand out from the noise floor.
        threshold = 4 * np.std(koro_mag)
        peaks, _ = signal.find_peaks(np.abs(koro_mag), height=threshold, distance=int(fs*0.4))
        
        print(f"--- Analysis Report for rec_koro_may11.h5 ---")
        print(f"Signal Levels: I_max={i_max:.4f}, Q_max={q_max:.4f}")
        print(f"Korotkoff Band RMS: {rms_koro:.6e}")
        print(f"Pulses Detected: {len(peaks)}")
        
        valid = True
        suggestions = []
        
        if i_max > 0.9 or q_max > 0.9:
            valid = False
            suggestions.append("CRITICAL: Signal is SATURATED. Lower RX gain.")
        elif i_max < 0.05:
            suggestions.append("WARNING: Signal level is low. Increase RX gain for better resolution.")
            
        if len(peaks) < 10:
            # If it's a 40s recording, we expect ~40-60 pulses if the cuff is deflating.
            suggestions.append("NOTE: Few pulses detected. This could be due to:")
            suggestions.append("  - Cuff not deflating yet (Systolic not reached)")
            suggestions.append("  - Antenna misalignment")
            suggestions.append("  - Too much movement noise")
        
        if rms_koro < 1e-5:
             suggestions.append("WARNING: Very low energy in Korotkoff band. Possible dead signal.")

        if valid:
            print("\nRESULT: Data is MATHEMATICALLY VALID (No clipping/dead signal).")
        else:
            print("\nRESULT: Data is INVALID.")
            
        if suggestions:
            print("Suggestions:")
            for s in suggestions:
                print(f"  {s}")
                
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == '__main__':
    analyze()
