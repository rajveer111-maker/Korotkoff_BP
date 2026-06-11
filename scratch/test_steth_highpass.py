import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000

# DSP Helpers
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth_box(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def normalise(env, t, k_on, k_off, pct=5):
    base_mask  = (t >= 20.0) & (t <= k_on - 2.0)
    koro_mask  = (t >= k_on) & (t <= k_off)
    baseline   = np.percentile(env[base_mask], pct) if base_mask.any() else 0.0
    e          = np.maximum(env - baseline, 0)
    peak       = np.max(e[koro_mask]) if koro_mask.any() else 1.0
    return e / (peak + 1e-12)

def cusum_detect(env, t, search_on=20.0, search_off=52.0,
                 lower=0.08, upper=0.92, w_smooth=1.5):
    mask = (t >= search_on) & (t <= search_off)
    ev   = smooth_box(env[mask], w_smooth, FS)
    ts   = t[mask]
    if len(ev) == 0 or np.max(ev) == 0: return search_on, search_off
    cs = np.cumsum(ev)
    cs = cs / cs[-1]
    i_on  = np.where(cs >= lower)[0]
    i_off = np.where(cs >= upper)[0]
    on  = float(ts[i_on[0]])  if len(i_on)  else search_on
    off = float(ts[i_off[0]]) if len(i_off) else search_off
    return on, off

SESSIONS = [
    dict(sub=1, sub_dir='Sub_1_Prof_kan', label='Subject 1',
         rec=6,  k_on=27.53, k_off=43.33, lag=1.7083),
    dict(sub=2, sub_dir='Sub_2_Rajveer',  label='Subject 2',
         rec=4,  k_on=27.38, k_off=42.00, lag=2.6042),
]

for st_hp in [50, 80, 100, 120]:
    print(f"\n--- Steth High-pass: {st_hp} Hz ---")
    for s in SESSIONS:
        wav_path = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")
        fs_a, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float64) / 32768.0
        if audio.ndim > 1: audio = audio.mean(1)
        
        audio_bp    = bpf(audio, st_hp, 1000, fs_a)
        steth_env_a = bpf(np.abs(hilbert(audio_bp)), 20, min(200, fs_a/2-1), fs_a)
        steth_tkeo_a = smooth_box(tkeo(steth_env_a), 1.5, fs_a)
        
        t_a = np.arange(len(steth_tkeo_a)) / fs_a
        t = np.arange(int(55 * FS)) / FS
        steth_env = np.interp(t, t_a + s['lag'], steth_tkeo_a)
        steth_n = normalise(steth_env, t, s['k_on'], s['k_off'], pct=5)
        
        # Grid search CUSUM thresholds for this high-pass setting
        best_err = 999.0
        best_pair = None
        best_window = None
        
        thresholds = [
            (l, u) 
            for l in np.arange(0.02, 0.16, 0.01)
            for u in np.arange(0.80, 0.98, 0.01)
        ]
        
        for l, u in thresholds:
            on, off = cusum_detect(steth_n, t, search_on=20.0, search_off=52.0, lower=l, upper=u)
            dur = off - on
            err = abs(dur - (s['k_off'] - s['k_on']))
            # Also require onset to be reasonably close to k_on (e.g. within 3 seconds) to prevent false-positives
            if err < best_err and abs(on - s['k_on']) < 3.0:
                best_err = err
                best_pair = (l, u)
                best_window = (on, off)
                
        if best_pair is not None:
            print(f"  {s['label']}: Best L={best_pair[0]:.2f}, U={best_pair[1]:.2f} | Det: {best_window[0]:.2f} to {best_window[1]:.2f} ({best_window[1]-best_window[0]:.2f}s vs GT {s['k_off']-s['k_on']:.2f}s) |Err|={best_err:.3f}s")
        else:
            print(f"  {s['label']}: No valid window found within 3s of onset!")
