import sys
import numpy as np
from scipy.signal import hilbert, find_peaks

sys.path.append(r'd:\Bioview\My_RF_work_v1')
from koro_parallel_features import load_stethoscope

AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\new_recoding.mp4'

t_aud, aud_raw, koro_aud, fs_aud, _ = load_stethoscope(AUDIO_PATH)
if t_aud is not None:
    env = np.abs(hilbert(koro_aud))
    env_smooth = np.convolve(env, np.ones(int(fs_aud*0.1))/int(fs_aud*0.1), mode='same')
    
    # Find peaks to see if there are beat-like structures
    peaks, properties = find_peaks(env_smooth, distance=fs_aud*0.5, prominence=np.max(env_smooth)*0.1)
    
    with open(r'C:\Users\rajve\.gemini\antigravity\brain\b11c4ec4-c7a3-4eaf-86b7-1efc0188caab\scratch\stats.txt', 'w') as f:
        f.write(f"Audio Duration: {t_aud[-1]:.2f}s\n")
        f.write(f"Max Envelope Value: {np.max(env_smooth):.4f}\n")
        f.write(f"Mean Envelope Value: {np.mean(env_smooth):.4f}\n")
        f.write(f"Number of distinct peaks found (prominence > 10% max): {len(peaks)}\n")
        
        if len(peaks) > 0:
            peak_times = t_aud[peaks]
            f.write(f"Peak times: {peak_times}\n")
            f.write(f"Average Heart Rate (if peaks are beats): {60 / np.mean(np.diff(peak_times)):.1f} BPM\n")
