import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\sthethoscope_rec01.wav'
# If wav doesn't exist, we can convert mp4 once in the test
mp4_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\sthethoscope_rec01.mp4'

def find_true_korotkoff_click_train(audio, fs, min_onset_s=20.0, max_offset_s=45.0):
    t = np.arange(len(audio)) / fs
    
    # Filter 80-200 Hz
    sos_k = butter(4, [80, 200], btype='band', fs=fs, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    aud_env = np.abs(hilbert(koro_aud))
    
    # Deflation search space
    idx = (t >= min_onset_s) & (t <= max_offset_s)
    t_defl = t[idx]
    env_defl = aud_env[idx]
    
    # Detect peaks with low prominence to capture all heartbeat clicks
    peaks_idx, _ = find_peaks(env_defl, distance=int(fs * 0.4), prominence=0.005)
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    n_peaks = len(peaks_t)
    if n_peaks < 2:
        return min_onset_s, min_onset_s + 5.0
        
    # We want to find a sequence of peaks that are:
    # 1. Spaced by 0.65s to 1.35s (human heartbeat interval)
    # 2. Homogeneous in height (e.g. no peak in the sequence is > 4x the median height of the sequence, 
    #    which filters out the massive valve opening click)
    best_seq = []
    
    for i in range(n_peaks):
        seq = [i]
        curr = i
        for j in range(i + 1, n_peaks):
            diff = peaks_t[j] - peaks_t[curr]
            if 0.65 <= diff <= 1.35:
                # Add to sequence candidate
                seq.append(j)
                curr = j
        
        if len(seq) > len(best_seq):
            # Check homogeneity constraint:
            # We want to filter out the massive valve clicks which are > 1.0, 
            # while the true Korotkoff clicks are around 0.15 - 0.35.
            # Let's check if the standard deviation of heights in the sequence is small,
            # or if the ratio of max to min height is reasonable (e.g. <= 4.0)
            heights = peaks_h[seq]
            median_h = np.median(heights)
            
            # Filter out any peaks in the sequence that are outliers (e.g. > 4x the median)
            clean_seq = [idx for idx in seq if peaks_h[idx] <= 4.0 * median_h]
            
            if len(clean_seq) > len(best_seq):
                best_seq = clean_seq
                
    if len(best_seq) >= 2:
        st_on = peaks_t[best_seq[0]]
        st_off = peaks_t[best_seq[-1]]
        # Pad slightly by 0.3s to capture the click envelopes
        st_on = max(min_onset_s, st_on - 0.3)
        st_off = min(t[-1], st_off + 0.3)
    else:
        st_on, st_off = 23.0, 27.0
        
    return st_on, st_off

def main():
    # Make sure WAV file is created for testing
    if not os.path.exists(wav_path) and os.path.exists(mp4_path):
        print("Extracting mp4 to wav...")
        from moviepy import AudioFileClip
        clip = AudioFileClip(mp4_path)
        clip.write_audiofile(wav_path, fps=44100, verbose=False, logger=None)
        clip.close()
        
    print("Loading wav file...")
    fs, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        
    st_on, st_off = find_true_korotkoff_click_train(audio, fs)
    print(f"Detected True Acoustic Korotkoff Click-Train Window: {st_on:.2f}s - {st_off:.2f}s (dur: {st_off - st_on:.2f}s)")

if __name__ == '__main__':
    main()
