import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, medfilt
from scipy.io import wavfile

wav_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'

MIN_ONSET_S = 20.0
MIN_TAIL_S = 5.0
MIN_DUR_S = 3.0
MAX_DUR_S = 25.0

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True).mean().fillna(0).values)

def find_sustained_legacy(curve, time, fs, rec_dur, min_dur=3.0, max_dur=25.0):
    ss = int(MIN_ONSET_S * fs)
    se = int((rec_dur - MIN_TAIL_S) * fs)
    if se <= ss + int(min_dur * fs):
        return None
    sw = max(3, int(fs * 0.5)) | 1
    cc = medfilt(curve, min(sw, len(curve) if len(curve) % 2 == 1 else len(curve) - 1))
    cc = smooth(cc, int(fs * 1.0))
    best_score, best_on, best_off = -1, 0, 0
    for dt in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        ws = int(dt * fs)
        dw = np.exp(-0.5 * ((dt - 10.0) / 3.0)**2)  # Prior centered around 10.0s Korotkoff duration
        for s in range(ss, se - ws, int(fs * 0.25)):
            e = s + ws
            if e > se:
                break
            sc = np.sum(cc[s:e]) * dw
            if sc > best_score:
                best_score = sc
                best_on  = time[s]
                best_off = time[min(e, len(time) - 1)]
    d = best_off - best_on
    return {'onset': best_on, 'offset': best_off, 'duration': d} if d > 2 else None

def main():
    fs_aud, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_aud = np.arange(len(audio)) / fs_aud
    
    # 80-200 Hz filter
    sos_k = butter(4, [80, 200], btype='band', fs=fs_aud, output='sos')
    koro_aud = sosfiltfilt(sos_k, audio)
    
    # Run the 3 methods
    rec_dur_aud = t_aud[-1]
    
    # Method A: Envelope RMS
    aud_env = np.abs(hilbert(koro_aud))
    aud_curve_a = sliding_rms(aud_env, int(fs_aud * 0.5))**2
    aud_win_a = find_sustained_legacy(aud_curve_a, t_aud, fs_aud, rec_dur_aud)

    # Method B: Direct Bandpassed RMS
    aud_curve_b = sliding_rms(koro_aud, int(fs_aud * 0.3))**2
    aud_curve_b = smooth(aud_curve_b, int(fs_aud * 1.0))
    aud_win_b = find_sustained_legacy(aud_curve_b, t_aud, fs_aud, rec_dur_aud)

    # Method C: STFT Sub-band Power
    nps = min(4096, len(koro_aud)//2)
    f_s, t_s, Zs = signal.stft(koro_aud, fs=fs_aud, nperseg=nps, noverlap=nps * 3 // 4)
    Ps = np.abs(Zs)**2
    km_aud = (f_s >= 80) & (f_s <= 200)
    se_aud = np.mean(Ps[km_aud, :], axis=0) if np.any(km_aud) else np.zeros(len(t_s))
    aud_curve_c = np.interp(t_aud, t_s, se_aud)
    aud_win_c = find_sustained_legacy(aud_curve_c, t_aud, fs_aud, rec_dur_aud)

    steth_wins = [w for w in [aud_win_a, aud_win_b, aud_win_c] if w is not None]
    for name, w in zip(['A', 'B', 'C'], [aud_win_a, aud_win_b, aud_win_c]):
        if w:
            print(f"Method {name}: {w['onset']:.2f}s - {w['offset']:.2f}s (dur: {w['duration']:.2f}s)")
        else:
            print(f"Method {name} failed")
            
    if steth_wins:
        st_on  = float(np.median([w['onset']  for w in steth_wins]))
        st_off = float(np.median([w['offset'] for w in steth_wins]))
        print(f"\nConsensus Stethoscope Window: {st_on:.2f}s - {st_off:.2f}s (dur: {st_off-st_on:.2f}s)")
    else:
        print("Consensus failed")

if __name__ == '__main__':
    main()
