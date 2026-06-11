import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

def main():
    print("Loading Stethoscope audio...")
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # Filter 100-250 Hz
    sos_k = butter(4, [100, 250], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    aud_env = np.abs(hilbert(koro_aud))
    
    # Focus on 20s to 30s
    idx = (t_aud >= 20.0) & (t_aud <= 30.0)
    t_focus = t_aud[idx]
    env_focus = aud_env[idx]
    
    # Find all peaks in envelope
    peaks, props = find_peaks(env_focus, distance=int(fs_aud * 0.4), prominence=0.001)
    
    print("Envelope peaks between 20.0s and 30.0s in 100-250 Hz band:")
    for p in peaks:
        t_peak = t_focus[p]
        val_peak = env_focus[p]
        print(f"  Peak at {t_peak:.3f}s with envelope height {val_peak:.4f}")

if __name__ == '__main__':
    main()
