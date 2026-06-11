import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

DATA_DIR = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
SUBJECTS = ['Sub_1_Prof_kan', 'Sub_2_Rajveer']

def find_robust_stethoscope_window(koro_aud, t_aud, fs_aud):
    aud_env = np.abs(hilbert(koro_aud))
    min_onset_s = 20.0
    max_offset_s = 45.0
    
    idx = (t_aud >= min_onset_s) & (t_aud <= max_offset_s)
    t_defl = t_aud[idx]
    env_defl = aud_env[idx]
    
    # Detect peaks with low prominence to capture all heartbeat clicks
    peaks_idx, _ = find_peaks(env_defl, distance=int(fs_aud * 0.4), prominence=0.005)
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    n_peaks = len(peaks_t)
    if n_peaks < 2:
        return min_onset_s, min_onset_s + 5.0
        
    best_seq = []
    for i in range(n_peaks):
        seq = [i]
        curr = i
        for j in range(i + 1, n_peaks):
            diff = peaks_t[j] - peaks_t[curr]
            if 0.65 <= diff <= 1.35:
                seq.append(j)
                curr = j
        
        if len(seq) > len(best_seq):
            heights = peaks_h[seq]
            median_h = np.median(heights)
            clean_seq = [idx_ for idx_ in seq if peaks_h[idx_] <= 4.0 * median_h]
            if len(clean_seq) > len(best_seq):
                best_seq = clean_seq
                
    if len(best_seq) >= 2:
        st_on = peaks_t[best_seq[0]]
        st_off = peaks_t[best_seq[-1]]
        st_on = max(min_onset_s, st_on - 0.3)
        st_off = min(t_aud[-1], st_off + 0.3)
    else:
        st_on, st_off = 23.0, 27.0
        
    return st_on, st_off

def check_all():
    print("Checking detected stethoscope windows across all sessions...")
    for sub in SUBJECTS:
        sub_dir = os.path.join(DATA_DIR, sub)
        if not os.path.exists(sub_dir):
            continue
        print(f"\n--- Subject: {sub} ---")
        for i in range(1, 11):
            wav_path = os.path.join(sub_dir, f'sthethoscope_rec{i:02d}.wav')
            if not os.path.exists(wav_path) and i == 9 and sub == 'Sub_1_Prof_kan':
                wav_path = os.path.join(sub_dir, f'sthethoscope_rec9.wav')
                
            if not os.path.exists(wav_path):
                print(f"  Session {i}: WAV not found")
                continue
                
            try:
                fs_aud, audio = wavfile.read(wav_path)
                audio = audio.astype(np.float64) / 32768.0
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                t_aud = np.arange(len(audio)) / fs_aud
                
                sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
                koro_aud = sosfiltfilt(sos_k, audio)
                
                st_on, st_off = find_robust_stethoscope_window(koro_aud, t_aud, fs_aud)
                print(f"  Session {i:02d}: window = {st_on:.2f}s - {st_off:.2f}s | dur = {st_off - st_on:.2f}s")
            except Exception as e:
                print(f"  Session {i:02d}: [ERROR] {e}")

if __name__ == '__main__':
    check_all()
