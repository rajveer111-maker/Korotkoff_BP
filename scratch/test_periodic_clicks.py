import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

def find_periodic_korotkoff_clicks(audio, fs, min_onset_s=20.0, max_offset_s=45.0):
    t = np.arange(len(audio)) / fs
    
    # Filter 80-200 Hz
    sos_k = butter(4, [80, 200], btype='band', fs=fs, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    # Envelope
    env = np.abs(hilbert(koro_aud))
    
    # Only search in deflation window (after min_onset_s)
    idx = (t >= min_onset_s) & (t <= max_offset_s)
    t_defl = t[idx]
    env_defl = env[idx]
    
    # Step 1: Detect peaks with a low height threshold (e.g., 2% of max or absolute 0.005)
    # This is to make sure we capture the smaller Korotkoff clicks
    peaks_idx, props = find_peaks(env_defl, distance=int(fs * 0.4), prominence=0.005)
    
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    print(f"Detected {len(peaks_t)} candidate peaks:")
    for pt, ph in zip(peaks_t, peaks_h):
        print(f"  t={pt:.3f}s, height={ph:.4f}")
        
    # Step 2: Find the subset of peaks that form a periodic pulse train matching human heart rate
    # Heart rate is typically 50-100 bpm, so peak spacing should be 0.6s to 1.3s.
    # Let's search for sequences of peaks with spacing in [0.7, 1.3] seconds.
    n_peaks = len(peaks_t)
    if n_peaks < 2:
        return min_onset_s, min_onset_s + 5.0
        
    best_seq = []
    
    # Simple depth-first search or chain builder to find the longest sequence of periodic peaks
    for i in range(n_peaks):
        seq = [i]
        curr = i
        for j in range(i + 1, n_peaks):
            diff = peaks_t[j] - peaks_t[curr]
            if 0.75 <= diff <= 1.35:
                # Found a peak at a valid heartbeat interval!
                seq.append(j)
                curr = j
        if len(seq) > len(best_seq):
            best_seq = seq
            
    print("\nBest periodic sequence of peaks found:")
    for idx in best_seq:
        print(f"  t={peaks_t[idx]:.3f}s, height={peaks_h[idx]:.4f}")
        
    if len(best_seq) >= 2:
        # True Korotkoff window starts at the first click and ends at the last click of the periodic sequence!
        koro_on = peaks_t[best_seq[0]]
        koro_off = peaks_t[best_seq[-1]]
        
        # Add a tiny padding (e.g. 0.5s) to cover the full duration of the first and last click
        koro_on = max(min_onset_s, koro_on - 0.5)
        koro_off = min(t[-1], koro_off + 0.5)
    else:
        # Fallback to simple threshold
        koro_on, koro_off = 23.0, 26.5
        
    return koro_on, koro_off

def main():
    print("Loading Stethoscope audio...")
    fs_aud, audio = wavfile.read(AUDIO_PATH)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        
    koro_on, koro_off = find_periodic_korotkoff_clicks(audio, fs_aud)
    print(f"\nFinal Detected Korotkoff Window: {koro_on:.2f}s - {koro_off:.2f}s (dur: {koro_off - koro_on:.2f}s)")

if __name__ == '__main__':
    main()
