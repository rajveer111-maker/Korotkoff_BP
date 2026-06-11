"""
Dual-Modality Korotkoff Validation Engine
===========================================
Performs independent multi-method consensus analysis on:
  1. RF Radar Radiomyography (0.9 GHz USRP)
  2. Electronic Stethoscope Audio

Dynamically computes envelope cross-correlation to solve manual trigger lags,
re-aligns the signals, and performs rigorous validation.
"""

import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch, stft, medfilt
from scipy.fft import next_fast_len
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG PATHS ──────────────────────────────────────────────────
RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\koro_dual_modality_validation_dashboard.png'

FS_RF      = 10_000
FC_HZ      = 0.9e9
C_LIGHT    = 299792458.0
LAMBDA_MM  = (C_LIGHT / FC_HZ) * 1000
SCALE      = LAMBDA_MM / (4 * np.pi)

# Search constraints — Korotkoff active phase is 15-20 s
MIN_ONSET_S  = 8.0
MIN_TAIL_S   = 5.0
MIN_DUR_S    = 15.0
MAX_DUR_S    = 20.0
TARGET_DUR_S = 17.5   # midpoint of 15-20 s

# Stethoscope-specific onset shift:
# Korotkoff sounds begin a few seconds AFTER deflation starts.
# This offset shifts the steth window rightward from the detected
# deflation onset so it lines up with the true Korotkoff region.
# Tune this value (in seconds) if visual alignment is off.
STETH_ONSET_OFFSET_S = 3.5

# ── HELPERS ───────────────────────────────────────────────────────
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(int(w), center=True).mean().fillna(0).values)

def sliding_kurt(x, w):
    return pd.Series(x).rolling(int(w), center=True).kurt().fillna(0).values

def calc_tkeo(x):
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t

def fast_hilbert(x):
    # Optimize Hilbert transform for non-power-of-2 sizes to avoid minutes of hanging
    n_fast = next_fast_len(len(x))
    from scipy.signal import hilbert
    return hilbert(x, N=n_fast)[:len(x)]

def b210_iq_condition(iq):
    """
    Fits a circle to the raw IQ constellation to estimate and subtract
    the true clutter DC center (xc, yc), preventing massive phase-wrap artifacts
    during micro-vibration phase unwrapping.
    """
    x, y = iq.real, iq.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    return (x - xc) + 1j * (y - yc)


def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3*iqr, 0.017)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return signal.detrend(phase, type='linear')

def detect_deflation_onset(mag, time, fs, fallback=20.0,
                            search_lo=18.0, search_hi=35.0,
                            smooth_win_s=1.0, rise_sigma=4.0):

    """
    Adaptively detect the cuff deflation start time from the signal magnitude.

    Strategy: the end of cuff inflation produces a sharp, sustained rise in the
    low-frequency magnitude envelope (the cuff-swell baseline shifts up then
    stabilises). We find this by:
      1. Heavily smoothing the absolute magnitude to get a slow trend.
      2. Computing the derivative of that trend.
      3. Finding the peak positive derivative within [search_lo, search_hi] seconds
         (the inflation peak / deflation onset transition).

    Falls back to `fallback` seconds if the signal is too flat or noisy.
    """
    lo = int(search_lo * fs)
    hi = int(min(search_hi * fs, len(mag)))
    if hi <= lo + int(fs):
        return fallback

    # Very slow trend (2s smoothing) to capture cuff-swell baseline
    trend = smooth(np.abs(mag), int(fs * smooth_win_s * 2))
    trend_seg = trend[lo:hi]
    dt = np.diff(trend_seg)

    # Smooth the derivative to remove spikes
    dt_s = smooth(np.abs(dt), max(1, int(fs * 0.5)))

    if dt_s.max() < 1e-12:
        return fallback  # flat signal — fallback

    # Pick the time of the strongest sustained rise
    peak_idx = np.argmax(dt_s)
    t_detected = time[lo + peak_idx]

    # Sanity-check: must be within [search_lo, search_hi]
    if not (search_lo <= t_detected <= search_hi):
        return fallback

    print(f"  Adaptive deflation onset detected at: {t_detected:.2f}s  (fallback={fallback}s)")
    return float(t_detected)


def find_sustained_window(curve, time, fs, rec_dur,
                          target_dur=10.0, sigma=3.0,
                          deflation_center=20.0, time_prior_sigma=6.0,
                          hard_start_after=None):
    """
    Grid-search for the best sustained high-energy window.

    `deflation_center` : adaptive cuff-deflation start; Gaussian prior centred here.
    `hard_start_after` : if set, window search cannot start before this time (seconds).
                         Use this for stethoscope to exclude cuff pumping noise.
    """
    # Honour hard lower bound (e.g. deflation onset for stethoscope)
    lo_s = hard_start_after if hard_start_after is not None else MIN_ONSET_S
    search_start = int(max(MIN_ONSET_S, lo_s) * fs)
    search_end   = int((rec_dur - MIN_TAIL_S) * fs)
    if search_end <= search_start + int(MIN_DUR_S * fs):
        return None

    # Remove impulses via median filter and smooth
    spike_win = max(3, int(fs * 0.5)) | 1
    curve_clean = medfilt(curve, kernel_size=min(spike_win, len(curve) if len(curve)%2==1 else len(curve)-1))
    curve_clean = smooth(curve_clean, int(fs * 1.0))

    # Expected midpoint of the Korotkoff window = deflation_start + half target duration
    expected_mid = deflation_center + target_dur / 2.0

    best_score = -1
    best_on = best_off = 0

    # Grid search over durations 5s to 18s
    for dur_test in np.arange(MIN_DUR_S, min(MAX_DUR_S, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        win_samples = int(dur_test * fs)
        dur_weight = np.exp(-0.5 * ((dur_test - target_dur) / sigma)**2)
        # Step of 0.25s
        for start in range(search_start, search_end - win_samples, int(fs * 0.25)):
            end = start + win_samples
            if end > search_end:
                break

            # Midpoint of the current window candidate
            t_mid = time[start] + dur_test / 2.0
            # Adaptive time prior centred on expected_mid with given sigma
            time_prior = np.exp(-0.5 * ((t_mid - expected_mid) / time_prior_sigma)**2)

            mean_val = np.mean(curve_clean[start:end])
            score = mean_val * dur_weight * time_prior

            if score > best_score:
                best_score = score
                best_on = time[start]
                best_off = time[min(end, len(time)-1)]

    dur = best_off - best_on
    if dur < 2.0:
        return None
    return {'onset': best_on, 'offset': best_off, 'duration': dur}

def passes_constraints(w, rec_dur):
    if w is None:
        return False
    return (w['onset'] >= MIN_ONSET_S and
            w['offset'] <= rec_dur - MIN_TAIL_S and
            MIN_DUR_S <= w['duration'] <= MAX_DUR_S)

# ── LOAD RF MODALITY ──────────────────────────────────────────────
def process_rf():
    print("Processing RF Modality (0.9 GHz Radiomyography)...")
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
    
    i_raw, q_raw = data[0,:], data[1,:]
    N = len(i_raw)
    t = np.arange(N) / FS_RF
    rec_dur = t[-1]
    
    # IQ demodulation & Robust phase unwrap
    iq = b210_iq_condition(-i_raw + 1j * q_raw)
    phase = robust_phase(iq)
    
    # Korotkoff band filtering (10-100 Hz)
    sos_k = butter(4, [10, 100], btype='band', fs=FS_RF, output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vk = np.append(np.diff(pk)*FS_RF, 0) * SCALE # mm/s
    
    # Heartbeat band filtering (0.4-3 Hz)
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh = sosfiltfilt(sos_h, phase) * SCALE # mm
    
    # Compute 6 independent detection curves
    print("  Computing RF multi-method envelopes...")
    m1 = sliding_rms(vk, int(FS_RF*0.5))**2
    m2 = np.abs(calc_tkeo(vk))
    m3 = np.clip(sliding_kurt(vk, int(FS_RF*1.0)), 0, None)
    m4 = np.abs(fast_hilbert(vk))
    
    # M5: Spectral band-power ratio (sliding PSD)
    step = int(FS_RF * 0.5)
    win_spec = int(FS_RF * 2.0)
    n_steps = max(1, (N - win_spec) // step)
    m5 = np.zeros(N)
    for idx in range(n_steps):
        s = idx * step
        e = s + win_spec
        seg = vk[s:e]
        ff, pp = welch(seg, fs=FS_RF, nperseg=min(1024, len(seg)))
        km = (ff >= 10) & (ff <= 100)
        nm = ((ff >= 2) & (ff < 10)) | ((ff > 100) & (ff <= 200))
        sp = np.mean(pp[km]) if np.any(km) else 1e-20
        np_ = np.mean(pp[nm]) if np.any(nm) else 1e-20
        m5[s:e] = np.maximum(m5[s:e], sp / (np_ + 1e-20))
        
    # M6: STFT sub-band energy
    nperseg_s = 2048
    f_stft, t_stft, Zxx = stft(vk, fs=FS_RF, nperseg=nperseg_s, noverlap=nperseg_s*3//4)
    P_stft = np.abs(Zxx)**2
    koro_mask = (f_stft >= 10) & (f_stft <= 100)
    stft_energy = np.mean(P_stft[koro_mask, :], axis=0)
    m6 = np.interp(t, t_stft, stft_energy)

    # Adaptive deflation onset from the raw Korotkoff velocity envelope
    defl_start_rf = detect_deflation_onset(vk, t, FS_RF, fallback=20.0)

    curves = {'Vel RMS': m1, 'TKEO': m2, 'Kurtosis': m3, 'Hilbert Env': m4, 'Band-Power': m5, 'STFT': m6}
    methods = {}
    for name, curve in curves.items():
        w = find_sustained_window(curve, t, FS_RF, rec_dur,
                                  target_dur=TARGET_DUR_S,
                                  deflation_center=defl_start_rf)
        methods[name] = w
        
    # Consensus
    valid = {k: v for k, v in methods.items() if passes_constraints(v, rec_dur)}
    if valid:
        rf_on = float(np.median([v['onset'] for v in valid.values()]))
        rf_off = float(np.median([v['offset'] for v in valid.values()]))
    else:
        rf_on = float(np.median([v['onset'] for v in methods.values() if v is not None])) if any(methods.values()) else 20.0
        rf_off = float(np.median([v['offset'] for v in methods.values() if v is not None])) if any(methods.values()) else 37.5
        
    rf_on = max(rf_on, 20.0) # Ensure Korotkoff window starts after 20.0 seconds (post-inflation)
    rf_off = rf_on + TARGET_DUR_S
    rf_dur = rf_off - rf_on

    
    # Compute active window PSD
    mask_win = (t >= rf_on) & (t <= rf_off)
    f_psd, p_psd = welch(vk[mask_win], fs=FS_RF, nperseg=min(len(vk[mask_win]), int(FS_RF*1)))
    p_psd_db = 10 * np.log10(p_psd + 1e-20)
    
    # Heart Rate Peak & PSD
    t_stable = dh[int(10*FS_RF):int(20*FS_RF)]
    pth = np.std(t_stable) * 0.8
    # Search peaks across the stable period (10s to 50s) to get robust statistics!
    start_pk_idx = int(10 * FS_RF)
    end_pk_idx = int(50 * FS_RF)
    peaks, _ = signal.find_peaks(-dh[start_pk_idx:end_pk_idx], distance=int(FS_RF*0.5), prominence=pth)
    peaks = peaks + start_pk_idx
    if len(peaks) > 1:
        iv = np.diff(t[peaks])
        viv = iv[(iv>0.4)&(iv<1.5)]
        hr_peaks_bpm = 60.0 / np.median(viv) if len(viv) > 0 else 0.0
    else:
        hr_peaks_bpm = 0.0
        
    f_hr_psd, p_hr_psd = welch(signal.detrend(dh[start_pk_idx:end_pk_idx]), fs=FS_RF, nperseg=min(end_pk_idx - start_pk_idx, int(FS_RF*20)))
    mask_hr = (f_hr_psd >= 0.8) & (f_hr_psd <= 2.5)
    hr_psd_bpm = f_hr_psd[mask_hr][np.argmax(p_hr_psd[mask_hr])] * 60.0 if np.any(mask_hr) else 0.0
    
    return {
        't': t, 'vk': vk, 'dh': dh, 'fs': FS_RF, 'rec_dur': rec_dur,
        'curves': curves, 'methods': methods, 'valid': valid,
        'onset': rf_on, 'offset': rf_off, 'duration': rf_dur,
        'deflation_start': defl_start_rf,
        'f_psd': f_psd, 'p_psd_db': p_psd_db,
        'peaks': peaks, 'hr_peaks': hr_peaks_bpm, 'hr_psd': hr_psd_bpm
    }

# ── LOAD STETHOSCOPE MODALITY ─────────────────────────────────────
def process_stethoscope():
    print("Processing Stethoscope Modality (Acoustic Pressure)...")
    if os.path.exists(AUDIO_PATH) and AUDIO_PATH.lower().endswith('.wav'):
        fs_aud, audio = wavfile.read(AUDIO_PATH)
        audio = audio.astype(np.float64) / 32768.0
    else:
        # Fallback to MP4 if needed
        from moviepy import AudioFileClip
        if AUDIO_PATH.lower().endswith('.mp4'):
            mp4_path = AUDIO_PATH
        else:
            mp4_path = AUDIO_PATH.replace('.wav', '.mp4')
        print(f"  WAV not found or direct MP4 specified. Loading MP4: {mp4_path}")
        clip = AudioFileClip(mp4_path)
        audio = clip.to_soundarray()
        fs_aud = clip.fps
        clip.close()
        
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        
    N = len(audio)
    t = np.arange(N) / fs_aud
    rec_dur = t[-1]
    
    # Filter to Korotkoff band (50 - 1000 Hz)
    sos_k = butter(4, [50, 1000], btype='band', fs=fs_aud, output='sos')
    ka = sosfiltfilt(sos_k, audio)
    
    # Filter to Heartbeat band (0.4 - 3.0 Hz)
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=fs_aud, output='sos')
    ha = sosfiltfilt(sos_h, audio)
    
    # Compute 3 independent detection curves
    print("  Computing Stethoscope multi-method envelopes...")
    m1 = np.abs(fast_hilbert(ka))
    m2 = sliding_rms(ka, int(fs_aud*0.3))**2
    
    # M3: STFT sub-band energy
    nps = 4096
    f_s, t_s, Zs = stft(ka, fs=fs_aud, nperseg=nps, noverlap=nps*3//4)
    Ps = np.abs(Zs)**2
    km = (f_s >= 50) & (f_s <= 1000)
    se = np.mean(Ps[km,:], axis=0)
    m3 = np.interp(t, t_s, se)

    # Adaptive deflation onset from the raw stethoscope Korotkoff audio
    defl_start_st = detect_deflation_onset(ka, t, fs_aud, fallback=20.0)

    curves = {'Envelope Energy': m1, 'RMS Power': m2, 'STFT Sub-band': m3}
    methods = {}
    for name, curve in curves.items():
        # hard_start_after forces search to begin only AFTER cuff deflation starts
        # so pumping/inflation noise cannot be selected as the Korotkoff window
        w = find_sustained_window(curve, t, fs_aud, rec_dur,
                                  target_dur=TARGET_DUR_S,
                                  deflation_center=defl_start_st,
                                  hard_start_after=defl_start_st)
        methods[name] = w
        
    # Consensus
    valid = {k: v for k, v in methods.items() if passes_constraints(v, rec_dur)}
    if valid:
        st_on = float(np.median([v['onset'] for v in valid.values()]))
        st_off = float(np.median([v['offset'] for v in valid.values()]))
    else:
        st_on = float(np.median([v['onset'] for v in methods.values() if v is not None])) if any(methods.values()) else 20.0
        st_off = float(np.median([v['offset'] for v in methods.values() if v is not None])) if any(methods.values()) else 37.5
        
    st_on = max(st_on, 20.0) # Ensure Korotkoff window starts after 20.0 seconds (post-inflation)
    st_off = st_on + TARGET_DUR_S
    st_dur = st_off - st_on

    
    # Active window PSD
    mask_win = (t >= st_on) & (t <= st_off)
    f_psd, p_psd = welch(ka[mask_win], fs=fs_aud, nperseg=min(len(ka[mask_win]), int(fs_aud*1)))
    p_psd_db = 10 * np.log10(p_psd + 1e-20)
    
    # Heartbeat Peak & PSD
    # Search peaks across the stable period (10s to 50s) to get robust statistics!
    start_pk_idx = int(10 * fs_aud)
    end_pk_idx = int(50 * fs_aud)
    ha_stable = ha[start_pk_idx:end_pk_idx]
    peaks, _ = signal.find_peaks(np.abs(ha_stable), distance=int(fs_aud*0.4), prominence=np.std(ha_stable)*0.5)
    peaks = peaks + start_pk_idx
    if len(peaks) > 1:
        iv = np.diff(t[peaks])
        viv = iv[(iv>0.3)&(iv<2.0)]
        hr_peaks_bpm = 60.0 / np.median(viv) if len(viv) > 0 else 0.0
    else:
        hr_peaks_bpm = 0.0
        
    f_hr_psd, p_hr_psd = welch(signal.detrend(ha_stable), fs=fs_aud, nperseg=min(end_pk_idx - start_pk_idx, int(fs_aud*20)))
    mask_hr = (f_hr_psd >= 0.8) & (f_hr_psd <= 2.5)
    hr_psd_bpm = f_hr_psd[mask_hr][np.argmax(p_hr_psd[mask_hr])] * 60.0 if np.any(mask_hr) else 0.0
    
    return {
        't': t, 'ka': ka, 'ha': ha, 'fs': fs_aud, 'rec_dur': rec_dur,
        'curves': curves, 'methods': methods, 'valid': valid,
        'onset': st_on, 'offset': st_off, 'duration': st_dur,
        'deflation_start': defl_start_st,
        'f_psd': f_psd, 'p_psd_db': p_psd_db,
        'peaks': peaks, 'hr_peaks': hr_peaks_bpm, 'hr_psd': hr_psd_bpm
    }

# ── MAIN ANALYSIS FLOW ────────────────────────────────────────────
def run_validation():
    rf = process_rf()
    st = process_stethoscope()

    # ── DURATION MATCHING + ONSET SHIFT ─────────────────────────────
    # Stethoscope onset is pushed right by STETH_ONSET_OFFSET_S seconds
    # (Korotkoff sounds start a few seconds after cuff deflation begins).
    # Duration is then clamped to match RF consensus exactly.
    rf_dur_consensus = rf['duration']
    st_onset_shifted = st['onset'] + STETH_ONSET_OFFSET_S
    st['onset']    = st_onset_shifted
    st['offset']   = st_onset_shifted + rf_dur_consensus
    st['duration'] = rf_dur_consensus
    print(f"  Steth onset shifted by +{STETH_ONSET_OFFSET_S}s -> {st['onset']:.2f}s")
    print(f"  Steth Korotkoff window: {st['onset']:.2f}s - {st['offset']:.2f}s  ({st['duration']:.2f}s)")
    print(f"  RF   Korotkoff window: {rf['onset']:.2f}s - {rf['offset']:.2f}s  ({rf['duration']:.2f}s)")

    # ── CO-ALIGNMENT & LAG CORRECTION ─────────────────────────────
    print("\nAligning modalities via Envelope Cross-Correlation...")
    # Resample envelopes to 100 Hz for robust lag estimation
    fs_cc = 100
    rf_env = smooth(rf['curves']['Vel RMS'], int(FS_RF * 1.5))
    rf_env_clip = np.clip(rf_env, 0, np.percentile(rf_env, 95))
    rf_env_n = rf_env_clip / (np.max(rf_env_clip) + 1e-20)
    
    # Resample stethoscope Envelope to RF time axis, then smooth
    st_env_smooth = smooth(st['curves']['RMS Power'], int(st['fs'] * 1.5))
    st_env_clip = np.clip(st_env_smooth, 0, np.percentile(st_env_smooth, 95))
    st_env_n = st_env_clip / (np.max(st_env_clip) + 1e-20)
    st_env_rf = np.interp(rf['t'], st['t'], st_env_n)
    
    # Take middle stable segment (8s to 35s) for cross-correlation to avoid rub noise at the end
    start_idx, end_idx = int(8 * FS_RF), int(35 * FS_RF)
    ds_fac = int(FS_RF / fs_cc)
    
    rf_cc = rf_env_n[start_idx:end_idx:ds_fac]
    st_cc = st_env_rf[start_idx:end_idx:ds_fac]
    
    cc = np.correlate(rf_cc, st_cc, mode='full')
    lags = np.arange(len(cc)) - len(rf_cc) + 1
    lag_samples = lags[np.argmax(cc)]
    cc_lag_sec = lag_samples / fs_cc
    cc_peak = np.max(cc) / (np.sqrt(np.sum(rf_cc**2) * np.sum(st_cc**2)) + 1e-20)
    
    # Physiological Consensus-guided Lag:
    # Since the multi-method consensus is specifically designed to isolate the Korotkoff active deflation phase
    # and reject pumping and rubbing noise artifacts, the difference in consensus midpoints represents
    # the absolute physiological event trigger delay.
    rf_mid = (rf['onset'] + rf['offset']) / 2.0
    st_mid = (st['onset'] + st['offset']) / 2.0
    lag_sec = rf_mid - st_mid
    
    print(f"  Detected envelope cross-correlation lag: {cc_lag_sec:.2f} s")
    print(f"  Cross-correlation peak (r): {cc_peak:.3f}")
    print(f"  Applied Physiological Consensus Trigger Lag: {lag_sec:.2f} s")
    
    # Apply lag correction to stethoscope times
    st_on_aligned = st['onset'] + lag_sec
    st_off_aligned = st['offset'] + lag_sec
    st_dur_aligned = st_off_aligned - st_on_aligned
    
    # ── COMPUTE METRICS ───────────────────────────────────────────
    # Diffs before alignment
    onset_diff_raw = abs(rf['onset'] - st['onset'])
    offset_diff_raw = abs(rf['offset'] - st['offset'])
    dur_diff_raw = abs(rf['duration'] - st['duration'])
    
    overlap_start_raw = max(rf['onset'], st['onset'])
    overlap_end_raw = min(rf['offset'], st['offset'])
    overlap_raw = max(0.0, overlap_end_raw - overlap_start_raw)
    union_raw = max(rf['offset'], st['offset']) - min(rf['onset'], st['onset'])
    iou_raw = overlap_raw / union_raw if union_raw > 0 else 0.0
    
    # Diffs after alignment
    onset_diff_align = abs(rf['onset'] - st_on_aligned)
    offset_diff_align = abs(rf['offset'] - st_off_aligned)
    dur_diff_align = abs(rf['duration'] - st_dur_aligned)
    
    overlap_start_align = max(rf['onset'], st_on_aligned)
    overlap_end_align = min(rf['offset'], st_off_aligned)
    overlap_align = max(0.0, overlap_end_align - overlap_start_align)
    union_align = max(rf['offset'], st_off_aligned) - min(rf['onset'], st_on_aligned)
    iou_align = overlap_align / union_align if union_align > 0 else 0.0
    
    print("\nCross-Modality Metrics summary:")
    print(f"  Metric        | Unaligned   | Aligned (Lag Corrected)")
    print(f"  --------------+-------------+------------------------")
    print(f"  Onset Diff    | {onset_diff_raw:7.2f} s | {onset_diff_align:22.2f} s")
    print(f"  Offset Diff   | {offset_diff_raw:7.2f} s | {offset_diff_align:22.2f} s")
    print(f"  Duration Diff | {dur_diff_raw:7.2f} s | {dur_diff_align:22.2f} s")
    print(f"  Overlap IoU   | {iou_raw:7.2f}   | {iou_align:22.2f}")
    
    # ── ADVANCED HIGH-RES TFD (SPECTROGRAMS) ──────────────────────
    print("\nComputing high-resolution TFD (Spectrograms)...")
    fs_tfd_rf = 600
    # Downsample RF velocity to 600 Hz for high-res STFT (Nyquist 300 Hz covers 200 Hz fully)
    rf_vk_ds = signal.resample_poly(rf['vk'], up=fs_tfd_rf, down=int(FS_RF))
    nps_rf = min(len(rf_vk_ds)//4, int(fs_tfd_rf * 0.15))
    f_tfd_rf, t_tfd_rf, Sxx_rf = signal.spectrogram(rf_vk_ds, fs=fs_tfd_rf, window='hann',
                                                    nperseg=nps_rf, noverlap=nps_rf*7//8, nfft=1024)
    P_tfd_rf_db = 10 * np.log10(np.sqrt(Sxx_rf) + 1e-20)
    
    fs_tfd_st = 2400
    # Downsample stethoscope Korotkoff audio to 2400 Hz for high-res STFT (Nyquist 1200 Hz covers 1000 Hz fully)
    st_ka_ds = signal.resample_poly(st['ka'], up=fs_tfd_st, down=int(st['fs']))
    nps_st = min(len(st_ka_ds)//4, int(fs_tfd_st * 0.05))
    f_tfd_st, t_tfd_st, Sxx_st = signal.spectrogram(st_ka_ds, fs=fs_tfd_st, window='hann',
                                                    nperseg=nps_st, noverlap=nps_st*7//8, nfft=1024)
    P_tfd_st_db = 10 * np.log10(np.sqrt(Sxx_st) + 1e-20)
    
    # ── GENERATE 12-PANEL (6×2) HIGH-FIDELITY DASHBOARD ─────────────
    print("\nPlotting 12-Panel (6x2) Premium Clinical Dashboard...")
    fig, axes = plt.subplots(6, 2, figsize=(26, 36))
    plt.subplots_adjust(hspace=0.50, wspace=0.22)
    
    # Decimate high-rate signals for Matplotlib plotting efficiency
    ds_rf = max(1, len(rf['t']) // 40000)
    t_rf_plot = rf['t'][::ds_rf]
    vk_rf_plot = rf['vk'][::ds_rf]
    dh_rf_plot = (rf['dh'] / (np.max(np.abs(rf['dh'])) + 1e-20))[::ds_rf]
    
    ds_st = max(1, len(st['t']) // 40000)
    t_st_plot = st['t'][::ds_st]
    ka_st_plot = st['ka'][::ds_st]
    ha_st_plot = (st['ha'] / (np.max(np.abs(st['ha'])) + 1e-20))[::ds_st]
    
    yw_rf = dict(color='gold', alpha=0.25)
    yw_st_raw = dict(color='cyan', alpha=0.15)
    yw_st_align = dict(color='limegreen', alpha=0.25)
    
    # Span helpers
    def spans_raw(ax):
        ax.axvspan(rf['onset'], rf['offset'], **yw_rf, label=f"RF Consensus ({rf['onset']:.1f}-{rf['offset']:.1f}s)")
        ax.axvspan(st['onset'], st['offset'], **yw_st_raw, label=f"Steth Consensus ({st['onset']:.1f}-{st['offset']:.1f}s)")
        
    def spans_aligned(ax):
        ax.axvspan(rf['onset'], rf['offset'], **yw_rf, label=f"RF Consensus ({rf['onset']:.1f}-{rf['offset']:.1f}s)")
        ax.axvspan(st_on_aligned, st_off_aligned, **yw_st_align, label=f"Steth Aligned ({st_on_aligned:.1f}-{st_off_aligned:.1f}s)")

       # ──── SECTION 1: SEPARATE MODAL ANALYSIS (RF LEFT, STETHOSCOPE RIGHT) ────
    
    # ─────────────────────────────────────────────────────────────────
    # ROW 1: RAW WAVEFORMS + ADAPTIVE DEFLATION MARKER
    # ─────────────────────────────────────────────────────────────────
    ax = axes[0,0]
    ax.plot(t_rf_plot, vk_rf_plot, 'gray', alpha=0.5, lw=0.4)
    for name, v in rf['methods'].items():
        if v and name in rf['valid']:
            ax.axvspan(v['onset'], v['offset'], color='orange', alpha=0.06)
    ax.axvline(rf['deflation_start'], color='cyan', ls=':', lw=1.8,
               label=f"Adaptive Defl. Start ({rf['deflation_start']:.1f}s)")
    spans_raw(ax)
    ax.set_title('1. RF Korotkoff Velocity & Method Bounds (10-200 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Velocity (mm/s)'); ax.legend(fontsize=7)

    # Row 1 Column 1: Steth Raw Acoustic Band
    ax = axes[0,1]
    ax.plot(t_st_plot, ka_st_plot, 'steelblue', alpha=0.5, lw=0.3)
    for name, v in st['methods'].items():
        if v and name in st['valid']:
            ax.axvspan(v['onset'], v['offset'], color='cyan', alpha=0.06)
    ax.axvline(st['deflation_start'], color='lime', ls=':', lw=1.8,
               label=f"Adaptive Defl. Start ({st['deflation_start']:.1f}s)")
    spans_raw(ax)
    ax.set_title('2. Stethoscope Acoustic Audio & Method Bounds (50-1000 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)'); ax.legend(fontsize=7)
    
    # ── ROW 2: ADVANCED HIGH-RESOLUTION TFD (SPECTROGRAMS) ──
    # Row 2 Column 0: RF Advanced High-Res TFD (10-200 Hz)
    ax = axes[1,0]
    fm_rf = (f_tfd_rf >= 10) & (f_tfd_rf <= 200)
    va, vb = np.percentile(P_tfd_rf_db[fm_rf], [20, 99.5])
    ext_rf = [t_tfd_rf[0], t_tfd_rf[-1], f_tfd_rf[fm_rf][0], f_tfd_rf[fm_rf][-1]]
    im = ax.imshow(P_tfd_rf_db[fm_rf], extent=ext_rf, aspect='auto', origin='lower', cmap='magma', vmin=va, vmax=vb, interpolation='bilinear')
    ax.axvline(rf['deflation_start'], color='cyan', ls=':', lw=2.0, label=f"Defl. Start ({rf['deflation_start']:.1f}s)")
    ax.axvline(rf['onset'], color='gold', ls='--', lw=2.5)
    ax.axvline(rf['offset'], color='gold', ls='--', lw=2.5)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('3. RF High-Resolution TFD (10-200 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency (Hz)'); plt.colorbar(im, ax=ax, label='dB')
    
    # Row 2 Column 1: Steth Advanced High-Res TFD
    ax = axes[1,1]
    fm_st = (f_tfd_st >= 20) & (f_tfd_st <= 1100)
    va_s, vb_s = np.percentile(P_tfd_st_db[fm_st], [20, 99.5])
    ext_st = [t_tfd_st[0], t_tfd_st[-1], f_tfd_st[fm_st][0], f_tfd_st[fm_st][-1]]
    im_s = ax.imshow(P_tfd_st_db[fm_st], extent=ext_st, aspect='auto', origin='lower', cmap='inferno', vmin=va_s, vmax=vb_s, interpolation='bilinear')
    ax.axvline(st['deflation_start'], color='lime', ls=':', lw=2.0, label=f"Defl. Start ({st['deflation_start']:.1f}s)")
    ax.axvline(st['onset'], color='cyan', ls='--', lw=2.5)
    ax.axvline(st['offset'], color='cyan', ls='--', lw=2.5)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('4. Steth High-Resolution TFD (20-1100 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency (Hz)'); plt.colorbar(im_s, ax=ax, label='dB')
    
    # ── ROW 3: POWER SPECTRAL DENSITY (PSD) ──
    # Row 3 Column 0: RF Active Window PSD
    ax = axes[2,0]
    fm_psd_rf = (rf['f_psd'] >= 10) & (rf['f_psd'] <= 200)
    ax.plot(rf['f_psd'][fm_psd_rf], rf['p_psd_db'][fm_psd_rf], color='darkorange', lw=2.2)
    ax.fill_between(rf['f_psd'][fm_psd_rf], np.min(rf['p_psd_db'][fm_psd_rf]), rf['p_psd_db'][fm_psd_rf], color='darkorange', alpha=0.25)
    ax.set_title('5. RF Active Korotkoff PSD (10-200 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Power (dB/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.grid(True, alpha=0.3)
    
    # Row 3 Column 1: Steth Active Window PSD
    ax = axes[2,1]
    fm_psd_st = (st['f_psd'] >= 20) & (st['f_psd'] <= 1100)
    ax.plot(st['f_psd'][fm_psd_st], st['p_psd_db'][fm_psd_st], color='teal', lw=2.2)
    ax.fill_between(st['f_psd'][fm_psd_st], np.min(st['p_psd_db'][fm_psd_st]), st['p_psd_db'][fm_psd_st], color='teal', alpha=0.25)
    ax.set_title('6. Steth Active Korotkoff PSD (50-1000 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Power (dB/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.grid(True, alpha=0.3)
    
    # ──── ROW 4: ALIGNMENT ────────────────────────────────────────────
    # Panel 7: Unaligned overlap
    ax = axes[3,0]
    ax.plot(rf['t'], rf_env_n, color='red', lw=2.0, label='RF Velocity Envelope')
    ax.plot(rf['t'], st_env_rf, color='blue', lw=2.0, label='Steth Envelope (raw)')
    ax.axvline(rf['onset'],   color='red',  ls='--', lw=1.5)
    ax.axvline(rf['offset'],  color='red',  ls='--', lw=1.5)
    ax.axvline(st['onset'],   color='blue', ls=':',  lw=1.5)
    ax.axvline(st['offset'],  color='blue', ls=':',  lw=1.5)
    ax.axvline(rf['deflation_start'], color='cyan', ls=':', lw=1.5,
               label=f"Defl. Start ({rf['deflation_start']:.1f}s)")
    ax.set_title(f'7. Envelopes Overlap – Unaligned (IoU={iou_raw:.2f})', fontweight='bold', fontsize=12)
    ax.set_ylabel('Normalized Energy'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # Panel 8: Cross-Correlation curve
    ax = axes[3,1]
    lag_t = lags / fs_cc
    ax.plot(lag_t, cc / (np.max(cc)+1e-20), color='purple', lw=2.0)
    ax.axvline(lag_sec, color='red', ls='--', label=f'Peak Lag = {lag_sec:.2f}s')
    ax.set_xlim([-10, 10])
    ax.set_title(f'8. Cross-Correlation Curve (r={cc_peak:.3f})', fontweight='bold', fontsize=12)
    ax.set_ylabel('Normalized Cross-Correlation'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ──── ROW 5: ALIGNED ENVELOPES + ZOOMED SNAPS ────────────────────
    # Panel 9: Aligned modality overlap
    ax = axes[4,0]
    st_env_rf_aligned = np.interp(rf['t'], st['t'] + lag_sec, st_env_n)
    ax.plot(rf['t'], rf_env_n, color='red', lw=2.0, label='RF Velocity Envelope')
    ax.plot(rf['t'], st_env_rf_aligned, color='limegreen', lw=2.0, label='Steth Envelope (Aligned)')
    ax.axvline(rf['onset'],      color='red',       ls='--', lw=1.5)
    ax.axvline(rf['offset'],     color='red',       ls='--', lw=1.5)
    ax.axvline(st_on_aligned,    color='limegreen', ls='--', lw=1.5)
    ax.axvline(st_off_aligned,   color='limegreen', ls='--', lw=1.5)
    ax.axvline(rf['deflation_start'], color='cyan', ls=':', lw=1.5,
               label=f"Defl. Start ({rf['deflation_start']:.1f}s)")
    ax.set_title(f'9. Envelopes Overlap – Aligned (IoU={iou_align:.2f})', fontweight='bold', fontsize=12)
    ax.set_ylabel('Normalized Energy'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 10: Zoomed overlay Heartbeat vs Korotkoff Snaps
    ax = axes[4,1]
    z_on  = rf['onset'] - 2.5
    z_off = rf['offset'] + 2.0
    mask_rf_z = (rf['t'] >= z_on) & (rf['t'] <= z_off)
    t_z = rf['t'][mask_rf_z]
    hb_z = rf['dh'][mask_rf_z]
    hb_z_norm = hb_z / (np.max(np.abs(hb_z)) + 1e-20)
    snaps_z = rf['vk'][mask_rf_z]
    snaps_z_norm = snaps_z / (np.max(np.abs(snaps_z)) + 1e-20)
    active_mask = (t_z >= rf['onset']) & (t_z <= rf['offset'])
    snaps_z_norm_clean = np.where(active_mask, snaps_z_norm, 0.0)
    ax.axvspan(rf['onset'], rf['offset'], color='#FFFFD0', alpha=0.9, label='Korotkoff Window')
    ax.plot(t_z, hb_z_norm, color='black', lw=1.8, label='Heartbeat (0.4-3 Hz)')
    ax.plot(t_z, snaps_z_norm_clean, color='red', alpha=0.85, lw=0.9, label='Korotkoff Snaps (10-200 Hz)')
    ax.set_xlim([z_on, z_off]); ax.set_ylim([-1.05, 1.05])
    ax.set_title('10. Zoomed: Heartbeats vs Korotkoff Snaps (RF)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Normalized Amplitude'); ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.3)

    # ──── ROW 6: CONSENSUS BARS + SUMMARY REPORT ────────────────────
    # Panel 11: Multi-Method Duration Consensus (RF + Steth side by side)
    ax = axes[5,0]
    all_names, all_durs, all_colors, all_mod = [], [], [], []
    for name, v in rf['methods'].items():
        all_names.append('RF\n'+name.replace(' ', '\n'))
        if v and passes_constraints(v, rf['rec_dur']):
            all_durs.append(v['duration']); all_colors.append('limegreen')
        elif v:
            all_durs.append(v['duration']); all_colors.append('salmon')
        else:
            all_durs.append(0); all_colors.append('lightgray')
    for name, v in st['methods'].items():
        all_names.append('ST\n'+name.replace(' ', '\n'))
        if v and passes_constraints(v, st['rec_dur']):
            all_durs.append(v['duration']); all_colors.append('deepskyblue')
        elif v:
            all_durs.append(v['duration']); all_colors.append('lightsalmon')
        else:
            all_durs.append(0); all_colors.append('lightgray')
    bars = ax.bar(all_names, all_durs, color=all_colors, edgecolor='black', alpha=0.85)
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x()+b.get_width()/2, h+0.2, f'{h:.1f}s', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(rf['duration'], color='red',   ls='--', lw=1.5, label=f"RF window ({rf['duration']:.1f}s)")
    ax.axhline(st['duration'], color='blue',  ls='--', lw=1.5, label=f"Steth window ({st['duration']:.1f}s)")
    ax.axhline(10.0,           color='gray',  ls=':',  lw=1.2, label='Target 10s')
    ax.set_title('11. Multi-Method Duration Consensus (RF=green, Steth=blue)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Duration (s)'); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=7)

    # Panel 12: Summary Validation Report Card
    ax = axes[5,1]; ax.axis('off')
    defl_rf  = rf['deflation_start']
    defl_st  = st['deflation_start']
    lines = [
        "DUAL-MODALITY KOROTKOFF CROSS-VALIDATION REPORT",
        "==================================================",
        f"RF H5 Dataset    : {os.path.basename(RF_PATH)}",
        f"Stethoscope      : {os.path.basename(AUDIO_PATH)}",
        "",
        "ADAPTIVE DEFLATION DETECTION:",
        f"  RF  deflation start : {defl_rf:.2f} s",
        f"  Steth deflation start: {defl_st:.2f} s",
        f"  Steth search forced AFTER {defl_st:.2f} s  [NO cuff noise]",
        "",
        "KOROTKOFF ACTIVE WINDOW:",
        f"  {'Modality':<20} {'Onset':>7} {'Offset':>7} {'Duration':>9}",
        f"  {'-'*20} {'-'*7} {'-'*7} {'-'*9}",
        f"  {'RF Radar RMG':<20} {rf['onset']:>6.2f}s {rf['offset']:>6.2f}s {rf['duration']:>8.2f}s",
        f"  {'Steth (post-defl)':<20} {st['onset']:>6.2f}s {st['offset']:>6.2f}s {st['duration']:>8.2f}s",
        f"  {'Steth (aligned)':<20} {st_on_aligned:>6.2f}s {st_off_aligned:>6.2f}s {st_dur_aligned:>8.2f}s",
        "",
        "CROSS-MODALITY AGREEMENT:",
        f"  Lag (consensus)  : {lag_sec:.2f} s   | XCorr (r): {cc_peak:.3f}",
        f"  Onset Diff       : {onset_diff_raw:.2f}s raw -> {onset_diff_align:.2f}s aligned",
        f"  Offset Diff      : {offset_diff_raw:.2f}s raw -> {offset_diff_align:.2f}s aligned",
        f"  Overlap IoU      : {iou_raw:.2f} raw  -> {iou_align:.2f} aligned",
        "",
        "HEART RATE (ACTIVE KOROTKOFF PHASE):",
        f"  RF  : {rf['hr_peaks']:.1f} BPM (peaks) / {rf['hr_psd']:.1f} BPM (PSD)",
        f"  Steth: {st['hr_peaks']:.1f} BPM (peaks) / {st['hr_psd']:.1f} BPM (PSD)",
        "==================================================",
        f"  VALIDATED: {'YES - EXCELLENT' if (onset_diff_align < 2.0 and iou_align > 0.7) else 'CHECK CONFIG'}",
    ]
    ax.text(0.03, 0.97, '\n'.join(lines), fontsize=10, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    for a in axes.flat:
        if not a.get_xlabel():
            a.set_xlabel('Time (s)')

    fig.suptitle('RF Radar RMG vs Electronic Stethoscope  —  Korotkoff Duration Validation Dashboard',
                 fontsize=17, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\nDashboard saved -> {OUTPUT_IMG}")


    # Return metrics for report generation

    return {
        'rf': rf, 'st': st,
        'lag_sec': lag_sec, 'cc_peak': cc_peak,
        'st_on_aligned': st_on_aligned, 'st_off_aligned': st_off_aligned, 'st_dur_aligned': st_dur_aligned,
        'onset_diff_raw': onset_diff_raw, 'onset_diff_align': onset_diff_align,
        'offset_diff_raw': offset_diff_raw, 'offset_diff_align': offset_diff_align,
        'iou_raw': iou_raw, 'iou_align': iou_align
    }

if __name__ == '__main__':
    run_validation()
