import os
import sys
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

def main():
    fs_aud, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # 80-200 Hz bandpass filter
    sos = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    filtered = sosfiltfilt(sos, audio)
    env = np.abs(hilbert(filtered))
    env_smoothed = np.convolve(env, np.ones(int(fs_aud * 0.5))/(fs_aud * 0.5), mode='same')
    
    # Find peaks in the deflation period (t >= 20s) with a low prominence
    # to capture all heartbeat pulses
    peaks, _ = find_peaks(env_smoothed, distance=int(fs_aud * 0.4), prominence=0.005)
    
    print("Detected acoustic pulses in 80-200 Hz band (t >= 20s):")
    count = 0
    for p in peaks:
        t = t_aud[p]
        if t >= 20.0 and t <= 30.0:
            count += 1
            print(f"  Pulse {count}: Peak at {t:.2f}s, value: {env_smoothed[p]:.5f}")

if __name__ == '__main__':
    main()
