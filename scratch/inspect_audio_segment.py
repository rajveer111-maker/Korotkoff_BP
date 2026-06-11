import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

audio_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

fs_aud, audio = wavfile.read(audio_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1:
    audio = audio.mean(axis=1)
t_aud = np.arange(len(audio)) / fs_aud

# Filter in 80-200 Hz
sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
koro_aud = sosfiltfilt(sos_k, audio)

# Compute envelope
aud_env = np.abs(hilbert(koro_aud))
aud_env_smoothed = np.convolve(aud_env, np.ones(int(fs_aud * 0.1))/(fs_aud * 0.1), mode='same')

# Zoom into 19s to 26s
idx = (t_aud >= 19.0) & (t_aud <= 26.0)
t_zoom = t_aud[idx]
env_zoom = aud_env_smoothed[idx]
raw_zoom = koro_aud[idx]

print("=== INSPECTING AUDIO IN 19s - 26s ===")
print(f"Max envelope in [19, 26]: {np.max(env_zoom):.6f}")
print(f"Mean envelope in [19, 26]: {np.mean(env_zoom):.6f}")

# Find peaks in [19, 26] with a very low prominence
peaks, properties = find_peaks(env_zoom, distance=int(fs_aud * 0.5), prominence=0.001)
print("\nDetected peaks in [19, 26]:")
for p in peaks:
    print(f"  Time: {t_zoom[p]:.4f}s, Envelope: {env_zoom[p]:.6f}")

# Print physical characteristics
diffs = np.diff(t_zoom[peaks]) if len(peaks) > 1 else []
print(f"\nPeak-to-Peak Intervals: {diffs}")
