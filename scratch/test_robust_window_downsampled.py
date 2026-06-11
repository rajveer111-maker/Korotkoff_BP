import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

MIN_ONSET_S = 20.0
MIN_TAIL_S = 5.0
MIN_DUR_S = 3.0
MAX_DUR_S = 25.0

def smooth_fast(x, w):
    # O(N) fast running mean using uniform_filter1d
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=int(w), mode='nearest')

def sliding_rms_fast(x, w):
    return np.sqrt(smooth_fast(x**2, w))

def find_sustained_legacy_fast(curve, time, fs, rec_dur, min_dur=3.0, max_dur=25.0):
    ss = int(MIN_ONSET_S * fs)
    se = int((rec_dur - MIN_TAIL_S) * fs)
    if se <= ss + int(min_dur * fs):
        return None
    
    # Fast O(N) smoothing
    cc = smooth_fast(curve, int(fs * 1.0))
    
    # Precompute cumulative sum for O(1) interval sum calculation
    cumsum = np.insert(np.cumsum(cc), 0, 0.0)
    
    best_score, best_on, best_off = -1, 0, 0
    # Grid search at 1000 Hz fs is extremely fast
    for dt in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        ws = int(dt * fs)
        dw = np.exp(-0.5 * ((dt - 10.0) / 3.0)**2)  # Prior centered around 10.0s Korotkoff duration
        for s in range(ss, se - ws, int(fs * 0.25)):
            e = s + ws
            if e > se:
                break
            sc = (cumsum[e] - cumsum[s]) * dw
            if sc > best_score:
                best_score = sc
                best_on  = time[s]
                best_off = time[min(e, len(time) - 1)]
    d = best_off - best_on
    return {'onset': best_on, 'offset': best_off, 'duration': d} if d > 2 else None

def main():
    print("Loading stethoscope audio...")
    fs_aud, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    print("Applying 80-200 Hz bandpass filter...")
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    # Calculate Hilbert envelope at full fs
    print("Computing Hilbert envelope...")
    aud_env = np.abs(hilbert(koro_aud))
    
    # Downsample envelope and filtered signal to 1000 Hz
    fs_down = 1000
    down_factor = int(fs_aud / fs_down)
    print(f"Downsampling from {fs_aud} Hz to {fs_down} Hz (factor: {down_factor})...")
    
    # Downsample using decimate or simple slicing (slicing is instant and perfectly fine for envelopes)
    t_down = np.arange(len(audio))[::down_factor] / fs_aud
    env_down = aud_env[::down_factor]
    koro_down = koro_aud[::down_factor]
    
    # Run the 3 methods at 1000 Hz!
    rec_dur = t_down[-1]
    
    print("Running Method A (Envelope RMS)...")
    aud_curve_a = sliding_rms_fast(env_down, int(fs_down * 0.5))**2
    aud_win_a = find_sustained_legacy_fast(aud_curve_a, t_down, fs_down, rec_dur)

    print("Running Method B (Direct Bandpassed RMS)...")
    aud_curve_b = sliding_rms_fast(koro_down, int(fs_down * 0.3))**2
    aud_win_b = find_sustained_legacy_fast(aud_curve_b, t_down, fs_down, rec_dur)

    print("Running Method C (STFT Sub-band Power)...")
    nps = min(256, len(koro_down)//2)
    f_s, t_s, Zs = signal.stft(koro_down, fs=fs_down, nperseg=nps, noverlap=nps * 3 // 4)
    Ps = np.abs(Zs)**2
    km_aud = (f_s >= 80) & (f_s <= 200)
    se_aud = np.mean(Ps[km_aud, :], axis=0) if np.any(km_aud) else np.zeros(len(t_s))
    aud_curve_c = np.interp(t_down, t_s, se_aud)
    aud_win_c = find_sustained_legacy_fast(aud_curve_c, t_down, fs_down, rec_dur)

    steth_wins = [w for w in [aud_win_a, aud_win_b, aud_win_c] if w is not None]
    for name, w in zip(['A', 'B', 'C'], [aud_win_a, aud_win_b, aud_win_c]):
        if w:
            print(f"  Method {name}: {w['onset']:.2f}s - {w['offset']:.2f}s (dur: {w['duration']:.2f}s)")
        else:
            print(f"  Method {name} failed")
            
    if steth_wins:
        st_on  = float(np.median([w['onset']  for w in steth_wins]))
        st_off = float(np.median([w['offset'] for w in steth_wins]))
        print(f"\n[Consensus Stethoscope Window]: {st_on:.2f}s - {st_off:.2f}s (dur: {st_off-st_on:.2f}s)")
    else:
        print("Consensus failed")

if __name__ == '__main__':
    main()
