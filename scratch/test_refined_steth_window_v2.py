import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

def find_robust_stethoscope_window_v2(koro_aud, t_aud, fs_aud, max_search_s=34.0):
    aud_env = np.abs(hilbert(koro_aud))
    min_onset_s = 20.0
    
    idx = (t_aud >= min_onset_s) & (t_aud <= max_search_s)
    t_defl = t_aud[idx]
    env_defl = aud_env[idx]
    
    # Detect peaks with low prominence to capture all clicks
    peaks_idx, _ = find_peaks(env_defl, distance=int(fs_aud * 0.4), prominence=0.005)
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    # Filter peaks to only keep those within a biophysically plausible height range for Korotkoff sounds
    # The true clicks are robustly between 0.10 and 0.50
    valid_idx = (peaks_h >= 0.10) & (peaks_h <= 0.50)
    peaks_t = peaks_t[valid_idx]
    peaks_h = peaks_h[valid_idx]
    
    n_peaks = len(peaks_t)
    print(f"Filtered peaks in [20.0s, {max_search_s:.1f}s]: {n_peaks}")
    for i, (t_p, h_p) in enumerate(zip(peaks_t, peaks_h)):
        print(f"  Peak {i:02d}: time = {t_p:.2f}s, height = {h_p:.4f}")
        
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
            best_seq = seq
            
    print(f"Best sequence: {best_seq}")
    if len(best_seq) >= 2:
        for idx_ in best_seq:
            print(f"  Seq Peak: index = {idx_}, time = {peaks_t[idx_]:.2f}s, height = {peaks_h[idx_]:.4f}")
        st_on = peaks_t[best_seq[0]]
        st_off = peaks_t[best_seq[-1]]
        st_on = max(min_onset_s, st_on - 0.3)
        st_off = min(t_aud[-1], st_off + 0.3)
    else:
        st_on, st_off = 23.0, 27.0
        
    return st_on, st_off

def main():
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    st_on, st_off = find_robust_stethoscope_window_v2(koro_aud, t_aud, fs_aud)
    print(f"\nFinal Detected Acoustic Window: {st_on:.2f}s - {st_off:.2f}s (dur = {st_off - st_on:.2f}s)")

if __name__ == '__main__':
    main()
