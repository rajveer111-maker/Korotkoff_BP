import numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io import wavfile

AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope1.mp4'

print("Loading Stethoscope...")
try:
    from moviepy import AudioFileClip
    clip = AudioFileClip(AUDIO_PATH)
    audio = clip.to_soundarray()
    fs_aud = clip.fps
    clip.close()
except:
    fs_aud, audio = wavfile.read(AUDIO_PATH.replace('.mp4','.wav'))
    audio = audio.astype(np.float64) / 32768.0

if audio.ndim > 1:
    audio = audio.mean(axis=1)

t_aud = np.arange(len(audio))/fs_aud

# Filter to heart sound band 0.5-5 Hz
sos_h = butter(4, [0.5, 5.0], btype='band', fs=fs_aud, output='sos')
hr_aud = sosfiltfilt(sos_h, audio)

# HR detection on stethoscope (time domain)
peaks_aud, _ = signal.find_peaks(np.abs(hr_aud), distance=int(fs_aud*0.4), prominence=np.std(hr_aud)*0.5)
if len(peaks_aud)>2:
    iv = np.diff(t_aud[peaks_aud]); viv = iv[(iv>0.3)&(iv<2.0)]
    hr_aud_bpm = 60.0/np.median(viv) if len(viv)>0 else 0
else: hr_aud_bpm = 0

print(f"Time-Domain Steth Peaks HR: {hr_aud_bpm:.1f} BPM, peaks found: {len(peaks_aud)}")

# Welch PSD on Stethoscope
f_aud, p_aud = welch(hr_aud, fs=fs_aud, nperseg=min(len(hr_aud), int(fs_aud * 20)))

# Test different search bands for Steth
for low_f in [0.5, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2]:
    mask = (f_aud >= low_f) & (f_aud <= 3.0)
    hz = f_aud[mask][np.argmax(p_aud[mask])]
    print(f"Steth PSD (low_f={low_f} Hz): {hz * 60.0:.1f} BPM")
