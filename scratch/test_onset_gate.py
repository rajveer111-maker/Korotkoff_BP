import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

# Set MIN_ONSET_S to 22.0 to bypass the valve opening click (20.0s - 21.5s)
MIN_ONSET_S = 22.0
MIN_TAIL_S = 5.0

def smooth(x, w):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=int(w), mode='nearest')

def sliding_rms(x, w):
    return np.sqrt(smooth(x**2, w))

def find_sustained_legacy(curve, time, fs, rec_dur, min_dur=2.0, max_dur=15.0):
    ss = int(MIN_ONSET_S * fs)
    se = int((rec_dur - MIN_TAIL_S) * fs)
    if se <= ss + int(min_dur * fs):
        return None
    
    cc = smooth(curve, int(fs * 0.5))
    cumsum = np.insert(np.cumsum(cc), 0, 0.0)
    
    best_score, best_on, best_off = -1, 0, 0
    # Search over dt with a prior centered around 4.0s (since the true duration is around 4.0s!)
    for dt in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.2, 0.2):
        ws = int(dt * fs)
        dw = np.exp(-0.5 * ((dt - 4.0) / 1.5)**2)  # Prior centered on 4.0s (matches RF Korotkoff duration!)
        for s in range(ss, se - ws, int(fs * 0.1)):
            e = s + ws
            if e > se:
                break
            sc = (cumsum[e] - cumsum[s]) * dw
            if sc > best_score:
                best_score = sc
                best_on  = time[s]
                best_off = time[min(e, len(time) - 1)]
    d = best_off - best_on
    return {'onset': best_on, 'offset': best_off, 'duration': d} if d > 1.5 else None

def main():
    fs_aud, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # 80-200 Hz Bandpass
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    aud_env = np.abs(hilbert(koro_aud))
    
    # Downsample to 1000 Hz
    fs_down = 1000
    down_factor = int(fs_aud / fs_down)
    t_down = t_aud[::down_factor]
    env_down = aud_env[::down_factor]
    koro_down = koro_aud[::down_factor]
    rec_dur = t_down[-1]
    
    # Method A: Envelope RMS
    aud_curve_a = sliding_rms(env_down, int(fs_down * 0.5))**2
    aud_win_a = find_sustained_legacy(aud_curve_a, t_down, fs_down, rec_dur)
    
    # Method B: Direct Bandpassed RMS
    aud_curve_b = sliding_rms(koro_down, int(fs_down * 0.3))**2
    aud_win_b = find_sustained_legacy(aud_curve_b, t_down, fs_down, rec_dur)
    
    print(f"Results with MIN_ONSET_S = {MIN_ONSET_S}s and 4.0s Prior:")
    if aud_win_a:
        print(f"  Method A (Env RMS): {aud_win_a['onset']:.2f}s - {aud_win_a['offset']:.2f}s (dur: {aud_win_a['duration']:.2f}s)")
    if aud_win_b:
        print(f"  Method B (BP RMS): {aud_win_b['onset']:.2f}s - {aud_win_b['offset']:.2f}s (dur: {aud_win_b['duration']:.2f}s)")

if __name__ == '__main__':
    main()
