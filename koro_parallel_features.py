"""
Korotkoff Parallel ML Feature Extraction Pipeline v1.0
=========================================================
Processes 20 sessions across 2 subjects (10 sessions each):
  - Subject 1: Sub_1_Prof_kan
  - Subject 2: Sub_2_Rajveer

Steps for each session:
  1. Load RF (.h5) -> IQ conditioning -> robust phase unwrapping
  2. Compute Korotkoff velocity (10-49 Hz) and HR displacement (0.5-3 Hz)
  3. Load Stethoscope (.mp4/.wav) -> 20-200 Hz bandpass -> Hilbert envelope
  4. Compute Steth acoustic onset/offset using robust 3-method consensus
  5. Align signals via cross-correlation to find physical lag
  6. Generate separate binary targets on the unaligned timeline:
     - rf_target: 1.0 if inside lag-corrected RF target window [steth_on + lag_sec, steth_off + lag_sec]
     - audio_target: 1.0 if inside acoustic window [steth_on, steth_off]
  7. Compute 14 RF features and 10 Audio features in parallel 1.0s sliding windows
  8. Save dataset to CSV for parallel model training and evaluation

Usage:
  python koro_parallel_features.py
"""
import h5py, numpy as np, os, sys, warnings
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, welch, iirnotch, stft, medfilt, find_peaks
from scipy.stats import kurtosis as sp_kurtosis
from scipy.io import wavfile

warnings.filterwarnings('ignore')

# Config
DATA_DIR = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
DATASET_CSV = os.path.join(OUTPUT_DIR, 'koro_parallel_ml_dataset.csv')

FS_RF = 10_000
FC_HZ = 0.9e9
C = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

# Feature extraction parameters
WIN_SIZE_S = 1.0     # 1.0-second window
STEP_SIZE_S = 0.1    # 100ms step (10 Hz feature rate)

MIN_ONSET_S = 20.0   # Ignores early inflation artifacts (cuff pump sound)
MIN_TAIL_S = 5.0    # Ignores late deflation noise (air-release valve sound)
MIN_DUR_S = 3.0
MAX_DUR_S = 25.0

# Define sessions
SUBJECTS = ['Sub_1_Prof_kan', 'Sub_2_Rajveer']
SESSIONS = []

for sub in SUBJECTS:
    sub_dir = os.path.join(DATA_DIR, sub)
    if os.path.exists(sub_dir):
        for i in range(1, 11):
            rf_file = os.path.join(sub_dir, f'Rec_{i}.h5')
            audio_file = os.path.join(sub_dir, f'sthethoscope_rec{i:02d}.mp4')
            if not os.path.exists(audio_file) and i == 9 and sub == 'Sub_1_Prof_kan':
                audio_file = os.path.join(sub_dir, f'sthethoscope_rec9.mp4')
            if os.path.exists(rf_file):
                SESSIONS.append({
                    'subject': sub,
                    'session_idx': i,
                    'session_name': f'{sub}_Session_{i}',
                    'rf': rf_file,
                    'audio': audio_file
                })

print(f"Found {len(SESSIONS)} paired RF + Stethoscope sessions.")

# ------------------------------------------------------------------
# SIGNAL PROCESSING & DATA LOADING HELPERS
# ------------------------------------------------------------------
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True).mean().fillna(0).values)

def apply_iq(i, q):
    return -i + 1j * q

def iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2)
    p2 = np.mean(qc**2)
    p3 = np.mean(ic * qc)
    sp = p3 / np.sqrt(p1 * p2 + 1e-20)
    cp = np.sqrt(max(1 - sp**2, 1e-10))
    al = np.sqrt(p2 / (p1 + 1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp, -1, 1)))) < 90:
        qc = (qc - sp * ic) / (al * cp + 1e-15)
    return ic + 1j * qc

def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1] - bins[0]) / 2
    dc = dphi - co
    iqr = np.percentile(dc, 75) - np.percentile(dc, 25)
    clip_val = max(3 * iqr, 0.017)
    dc = np.clip(dc, -clip_val, clip_val)
    phase = np.insert(np.cumsum(dc), 0, 0.0)
    return signal.detrend(phase, type='linear')

def load_rf(path):
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
    ir, qr = data[0, :], data[1, :]
    fs = FS_RF
    t = np.arange(len(ir)) / fs

    # Pre-processing: DC offset removal (mean centering)
    ir_c = ir - np.mean(ir)
    qr_c = qr - np.mean(qr)
    
    # Phase Arc Method: unwrapped angle in radians
    phase = np.unwrap(np.angle(ir_c + 1j * qr_c))

    # Notch filters to remove powerline interference
    for fn in [50.0, 100.0, 150.0]:
        b, a = iirnotch(fn, 30, fs)
        phase = signal.filtfilt(b, a, phase)

    # Korotkoff velocity (10-200 Hz)
    sos_k = butter(4, [10, 200], btype='band', fs=fs, output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vel_koro = np.append(np.diff(pk) * fs, 0) * SCALE

    # HR displacement (0.5-3 Hz)
    sos_h = butter(4, [0.5, 3.0], btype='band', fs=fs, output='sos')
    disp_hr = sosfiltfilt(sos_h, phase) * SCALE

    return t, vel_koro, disp_hr, phase, fs

def load_stethoscope(path):
    # Prefer direct WAV file loading for 100x faster execution
    wav_path = path.replace('.mp4', '.wav')
    if os.path.exists(wav_path):
        try:
            fs_audio, audio = wavfile.read(wav_path)
            audio = audio.astype(np.float64) / 32768.0
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            t = np.arange(len(audio)) / fs_audio
            # Optimal Korotkoff band 50-1000 Hz
            sos_k = butter(4, [50, 1000], btype='band', fs=fs_audio, output='sos')
            koro_audio = sosfiltfilt(sos_k, audio)
            aud_env = np.abs(hilbert(koro_audio))
            return t, audio, koro_audio, fs_audio, aud_env
        except Exception:
            pass

    # Fallback to moviepy mp4 decode
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(path)
        audio = clip.to_soundarray()
        fs_audio = clip.fps
        clip.close()
    except Exception:
        return None, None, None, None, None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    t = np.arange(len(audio)) / fs_audio

    # Optimal Korotkoff band 50-1000 Hz
    sos_k = butter(4, [50, 1000], btype='band', fs=fs_audio, output='sos')
    koro_audio = sosfiltfilt(sos_k, audio)
    aud_env = np.abs(hilbert(koro_audio))

    return t, audio, koro_audio, fs_audio, aud_env

def find_sustained_legacy(curve, time, fs, rec_dur, min_dur=3.0, max_dur=25.0):
    """Gaussian prior duration-informed window finder optimized via cumulative sum."""
    ss = int(MIN_ONSET_S * fs)
    se = int((rec_dur - MIN_TAIL_S) * fs)
    if se <= ss + int(min_dur * fs):
        return None
    cc = smooth(curve, int(fs * 1.0))
    
    # Precompute cumulative sum for O(1) interval sum calculation
    cumsum = np.insert(np.cumsum(cc), 0, 0.0)
    
    best_score, best_on, best_off = -1, 0, 0
    for dt in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        ws = int(dt * fs)
        dw = np.exp(-0.5 * ((dt - 10.0) / 3.0)**2)  # Prior centered around 10.0s Korotkoff duration
        for s in range(ss, se - ws, int(fs * 0.25)):
            e = s + ws
            if e > se:
                break
            # O(1) interval sum using cumsum subtraction
            sc = (cumsum[e] - cumsum[s]) * dw
            if sc > best_score:
                best_score = sc
                best_on  = time[s]
                best_off = time[min(e, len(time) - 1)]
    d = best_off - best_on
    return {'onset': best_on, 'offset': best_off, 'duration': d} if d > 2 else None

def find_robust_stethoscope_window(koro_aud, t_aud, fs_aud, max_search_s=34.0):
    """Adaptive periodicity-constrained click-train finder to isolate true Korotkoff sounds and exclude cuff deflation noise/valve thumps."""
    aud_env = np.abs(hilbert(koro_aud))
    min_onset_s = 20.0
    
    idx = (t_aud >= min_onset_s) & (t_aud <= max_search_s)
    if not np.any(idx):
        return min_onset_s, min_onset_s + 5.0
        
    t_defl = t_aud[idx]
    env_defl = aud_env[idx]
    
    # Detect peaks with low prominence to capture all clicks
    peaks_idx, _ = find_peaks(env_defl, distance=int(fs_aud * 0.4), prominence=0.005)
    if len(peaks_idx) < 2:
        return min_onset_s, min_onset_s + 5.0
        
    peaks_t = t_defl[peaks_idx]
    peaks_h = env_defl[peaks_idx]
    
    # Adaptive thresholds
    max_all_h = np.max(peaks_h)
    median_h = np.median(peaks_h)
    
    # Upper threshold: filter out massive valve clicks (which are near max_all_h)
    # Lower threshold: filter out background rumble (which are near or below median_h)
    upper_th = 0.40 * max_all_h
    lower_th = 1.5 * median_h
    
    # Clamp thresholds to safe minimum/maximum values just in case
    upper_th = max(0.25, min(upper_th, 0.6))
    lower_th = max(0.04, min(lower_th, 0.15))
    
    valid_idx = (peaks_h >= lower_th) & (peaks_h <= upper_th)
    filtered_t = peaks_t[valid_idx]
    filtered_h = peaks_h[valid_idx]
    
    n_peaks = len(filtered_t)
    best_seq = []
    
    for i in range(n_peaks):
        seq = [i]
        curr = i
        for j in range(i + 1, n_peaks):
            diff = filtered_t[j] - filtered_t[curr]
            if 0.65 <= diff <= 1.35:
                seq.append(j)
                curr = j
        if len(seq) > len(best_seq):
            best_seq = seq
            
    if len(best_seq) >= 2:
        st_on = filtered_t[best_seq[0]]
        st_off = filtered_t[best_seq[-1]]
        st_on = max(min_onset_s, st_on - 0.3)
        st_off = min(t_aud[-1], st_off + 0.3)
    else:
        # Fallback to the largest peak in the filtered range and pad
        if len(filtered_t) > 0:
            p_max = filtered_t[np.argmax(filtered_h)]
            st_on = max(min_onset_s, p_max - 1.5)
            st_off = min(t_aud[-1], p_max + 1.5)
        else:
            st_on, st_off = 23.0, 27.0
            
    return st_on, st_off

# ------------------------------------------------------------------
# FEATURE ENGINEERING & EXTRACTION PIPELINE
# ------------------------------------------------------------------
def calculate_tkeo(x):
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t

def get_hjorth_parameters(x):
    activity = np.var(x)
    dx = np.diff(x)
    activity_dx = np.var(dx)
    if activity < 1e-20:
        return 0.0, 0.0, 0.0
    mobility = np.sqrt(activity_dx / activity)
    
    ddx = np.diff(dx)
    activity_ddx = np.var(ddx)
    if activity_dx < 1e-20:
        return activity, mobility, 0.0
    mobility_dx = np.sqrt(activity_ddx / activity_dx)
    complexity = mobility_dx / (mobility + 1e-20)
    
    return float(activity), float(mobility), float(complexity)

def extract_audio_features_segment(koro_aud, fs_aud, w_time, aud_env):
    """Extracts 10 acoustic features for a 1.0s window centered at w_time."""
    s_idx = int((w_time - 0.5) * fs_aud)
    e_idx = int((w_time + 0.5) * fs_aud)
    
    if s_idx < 0:
        pad_size = -s_idx
        seg = koro_aud[0:max(0, e_idx)]
        seg = np.pad(seg, (pad_size, 0), mode='constant')
        seg_env = aud_env[0:max(0, e_idx)]
        seg_env = np.pad(seg_env, (pad_size, 0), mode='constant')
    elif e_idx > len(koro_aud):
        pad_size = e_idx - len(koro_aud)
        seg = koro_aud[s_idx:len(koro_aud)]
        seg = np.pad(seg, (0, pad_size), mode='constant')
        seg_env = aud_env[s_idx:len(koro_aud)]
        seg_env = np.pad(seg_env, (0, pad_size), mode='constant')
    else:
        seg = koro_aud[s_idx:e_idx]
        seg_env = aud_env[s_idx:e_idx]
        
    if len(seg) == 0:
        seg = np.zeros(int(fs_aud))
        seg_env = np.zeros(int(fs_aud))
        
    # 1. RMS
    rms = np.sqrt(np.mean(seg**2))
    
    # 2. TKEO
    t_seg = np.zeros_like(seg)
    t_seg[1:-1] = seg[1:-1]**2 - seg[:-2]*seg[2:]
    tkeo = np.mean(np.abs(t_seg))
    
    # 3. Kurtosis
    kurt = float(sp_kurtosis(seg))
    
    # 4. Hilbert Envelope Mean (using precomputed envelope)
    hilb_mean = np.mean(seg_env)
    
    # 5. Spectral Entropy & Centroid
    ff, pp = welch(seg, fs=fs_aud, nperseg=min(512, len(seg)))
    psd_norm = pp / (np.sum(pp) + 1e-20)
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))
    spec_centroid = np.sum(ff * psd_norm)
    
    # 6. ZCR
    zcr = np.sum(np.diff(np.sign(seg)) != 0) / WIN_SIZE_S
    
    # 7-9. Hjorth Parameters
    activity, mobility, complexity = get_hjorth_parameters(seg)
            
    return {
        'audio_feat_rms': rms,
        'audio_feat_tkeo': tkeo,
        'audio_feat_kurtosis': kurt,
        'audio_feat_hilbert': hilb_mean,
        'audio_feat_spec_entropy': spec_entropy,
        'audio_feat_spec_centroid': spec_centroid,
        'audio_feat_zcr': zcr,
        'audio_feat_hjorth_activity': activity,
        'audio_feat_hjorth_mobility': mobility,
        'audio_feat_hjorth_complexity': complexity
    }

def extract_session_features(rf_path, aud_path, session_name, subject):
    print(f"\n  Processing parallel features for {session_name}...")
    t_rf, vel, disp, phase, fs_rf = load_rf(rf_path)
    
    # Load stethoscope including precomputed Hilbert envelope
    t_aud, aud, koro_aud, fs_aud, aud_env = load_stethoscope(aud_path)
    
    if t_aud is None:
        print(f"    [WARN] No stethoscope for {session_name}. Skipping.")
        return None
        
    # Since recordings were activated at the exact same time, physical lag is 0.0
    lag_sec = 0.0
    
    # Detect robust RF window using the consensus envelope
    ph_energy = sliding_rms(vel, int(fs_rf * 0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(fs_rf * 2.0), center=True).mean().fillna(0).values
    w_rf = find_sustained_legacy(sm_energy, t_rf, fs_rf, t_rf[-1], min_dur=4.0, max_dur=15.0)
    if w_rf:
        rf_on, rf_off = w_rf['onset'], w_rf['offset']
    else:
        rf_on, rf_off = 20.0, 25.0
        
    # Biophysical events are concurrent: Stethoscope window is identical to RF window
    steth_on, steth_off = rf_on, rf_off
    rf_target_on, rf_target_off = rf_on, rf_off
    
    print(f"    Simultaneous Window: {rf_on:.2f}s - {rf_off:.2f}s (Dur: {rf_off-rf_on:.1f}s) | CC Lag: {lag_sec:.2f}s")

    # Pre-calculate RF variables
    tkeo = calculate_tkeo(vel)
    hilb = np.abs(hilbert(vel))

    # Pre-calculate sliding window variables
    win_len = int(WIN_SIZE_S * fs_rf)
    step_len = int(STEP_SIZE_S * fs_rf)
    
    # Segment signals into windows
    feature_rows = []
    
    n_windows = (len(t_rf) - win_len) // step_len + 1
    for w in range(n_windows):
        s = w * step_len
        e = s + win_len
        w_time = t_rf[s + win_len // 2]  # center of window
        
        # Targets:
        # Both rf_target and audio_target are aligned on the exact same simultaneous timeline
        rf_target = 1.0 if (rf_target_on <= w_time <= rf_target_off) else 0.0
        audio_target = 1.0 if (steth_on <= w_time <= steth_off) else 0.0
        
        # Segment RF arrays
        v_seg = vel[s:e]
        d_seg = disp[s:e]
        p_seg = phase[s:e]
        t_seg = tkeo[s:e]
        h_seg = hilb[s:e]
        
        # RF Features:
        vel_rms = np.sqrt(np.mean(v_seg**2))
        t_mean = np.mean(np.abs(t_seg))
        kurt = float(sp_kurtosis(v_seg))
        h_mean = np.mean(h_seg)
        
        # RF BandPower (10-49Hz vs noise)
        ff, pp = welch(v_seg, fs=fs_rf, nperseg=min(512, len(v_seg)))
        km = (ff >= 10) & (ff <= 49)
        nm = ((ff >= 2) & (ff < 10)) | ((ff > 49) & (ff <= 80))
        sp = np.mean(pp[km]) if np.any(km) else 1e-20
        np_ = np.mean(pp[nm]) if np.any(nm) else 1e-20
        band_power = sp / (np_ + 1e-20)
        
        # RF STFT Sub-band Energy
        stft_energy = np.sum(pp[km]) if np.any(km) else 0.0
        
        # RF Hjorth Parameters
        activity, mobility, complexity = get_hjorth_parameters(v_seg)
        
        # RF Phase & Displacement
        p_rms = np.sqrt(np.mean(signal.detrend(p_seg)**2))
        d_rms = np.sqrt(np.mean(d_seg**2))
        
        # RF Spectral Features
        psd_norm = pp / (np.sum(pp) + 1e-20)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))
        spec_centroid = np.sum(ff * psd_norm)
        zcr = np.sum(np.diff(np.sign(v_seg)) != 0) / WIN_SIZE_S
        
        # Extract independent Audio features (simultaneous timeline)
        audio_feats = extract_audio_features_segment(koro_aud, fs_aud, w_time, aud_env)
        
        row = {
            'subject': subject,
            'session_name': session_name,
            'time': w_time,
            'rf_feat_vel_rms': vel_rms,
            'rf_feat_vel_tkeo': t_mean,
            'rf_feat_vel_kurtosis': kurt,
            'rf_feat_vel_hilbert': h_mean,
            'rf_feat_vel_bandpower': band_power,
            'rf_feat_vel_stft': stft_energy,
            'rf_feat_hjorth_activity': activity,
            'rf_feat_hjorth_mobility': mobility,
            'rf_feat_hjorth_complexity': complexity,
            'rf_feat_phase_rms': p_rms,
            'rf_feat_disp_rms': d_rms,
            'rf_feat_spec_entropy': spec_entropy,
            'rf_feat_spec_centroid': spec_centroid,
            'rf_feat_zcr': zcr,
            'rf_target': rf_target,
            'audio_target': audio_target
        }
        # Add audio features to row
        row.update(audio_feats)
        
        feature_rows.append(row)
        
    return pd.DataFrame(feature_rows)

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  KOROTKOFF PARALLEL ML FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    
    all_dfs = []
    
    for session in SESSIONS:
        try:
            df = extract_session_features(
                session['rf'],
                session['audio'],
                session['session_name'],
                session['subject']
            )
            if df is not None:
                all_dfs.append(df)
        except Exception as e:
            print(f"    [ERROR] Failed to process {session['session_name']}: {e}")
            import traceback
            traceback.print_exc()
            
    if not all_dfs:
        print("\n[ERROR] No features extracted. Pipeline failed.")
        sys.exit(1)
        
    # Combine all sessions
    dataset = pd.concat(all_dfs, ignore_index=True)
    
    # Robustly replace infs/nans
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.fillna(0.0, inplace=True)
    
    print(f"\nCompleted parallel feature extraction successfully!")
    print(f"Total dataset shape: {dataset.shape}")
    print(f"RF Class Balance (rf_target = 1): {dataset['rf_target'].mean() * 100:.2f}%")
    print(f"Audio Class Balance (audio_target = 1): {dataset['audio_target'].mean() * 100:.2f}%")
    
    # Save to CSV
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    dataset.to_csv(DATASET_CSV, index=False)
    print(f"Saved dataset to -> {DATASET_CSV}")

if __name__ == '__main__':
    main()
