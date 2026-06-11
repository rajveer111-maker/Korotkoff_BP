import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from koro_parallel_features import load_stethoscope

audio_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.mp4'

def test_band(low, high):
    t_aud, aud_raw, _, fs_aud = load_stethoscope(audio_path)
    sos = butter(4, [low, high], btype='band', fs=fs_aud, output='sos')
    filtered = sosfiltfilt(sos, aud_raw)
    env = np.abs(hilbert(filtered))
    env_smoothed = np.convolve(env, np.ones(int(fs_aud * 0.5))/(fs_aud * 0.5), mode='same')
    
    # Find peaks after 20s (since deflation starts at 20s)
    peaks, _ = find_peaks(env_smoothed, distance=int(fs_aud * 0.8), prominence=np.max(env_smoothed)*0.03)
    
    print(f"\n--- Testing Band {low} - {high} Hz ---")
    print(f"Max envelope value overall: {np.max(env_smoothed):.6f}")
    print("Peaks detected:")
    for p in peaks:
        print(f"  Peak at {t_aud[p]:.2f}s, value: {env_smoothed[p]:.5f} (Rel to max: {env_smoothed[p]/np.max(env_smoothed)*100:.1f}%)")

def main():
    test_band(20, 200)
    test_band(80, 200)
    test_band(100, 250)

if __name__ == '__main__':
    main()
