import os
import sys
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

def test_band(low, high):
    fs_aud, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # Restrict to deflation period (t >= 20s)
    idx_def = t_aud >= 20.0
    t_aud = t_aud[idx_def]
    audio = audio[idx_def]
    
    sos = butter(4, [low, high], btype='band', fs=fs_aud, output='sos')
    filtered = sosfiltfilt(sos, audio)
    env = np.abs(hilbert(filtered))
    env_smoothed = np.convolve(env, np.ones(int(fs_aud * 0.5))/(fs_aud * 0.5), mode='same')
    
    # Find peaks with a very low prominence since we centered on deflation
    peaks, _ = find_peaks(env_smoothed, distance=int(fs_aud * 0.8), prominence=np.max(env_smoothed)*0.1)
    
    print(f"\n--- Testing Band {low} - {high} Hz (Deflation period t >= 20s) ---")
    print(f"Max envelope value in deflation: {np.max(env_smoothed):.6f}")
    print("Peaks detected during deflation:")
    for p in peaks:
        print(f"  Peak at {t_aud[p]:.2f}s, value: {env_smoothed[p]:.5f} (Rel to max: {env_smoothed[p]/np.max(env_smoothed)*100:.1f}%)")

def main():
    test_band(20, 200)
    test_band(80, 200)
    test_band(100, 250)

if __name__ == '__main__':
    main()
