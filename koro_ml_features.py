"""
Korotkoff ML Feature Extraction Pipeline v1.0
===============================================
Processes 20 sessions across 2 subjects (10 sessions each):
  - Subject 1: Sub_1_Prof_kan
  - Subject 2: Sub_2_Rajveer

Steps for each session:
  1. Load RF (.h5) -> IQ conditioning -> robust phase unwrapping
  2. Compute Korotkoff velocity (10-49 Hz) and HR displacement (0.5-3 Hz)
  3. Load Stethoscope (.mp4/.wav) -> 20-200 Hz bandpass -> Hilbert envelope
  4. Compute Steth acoustic onset/offset (consensus ground truth)
  5. Align signals via cross-correlation to find physical lag
  6. Generate lag-corrected binary targets on the RF timeline
  7. Compute 14 engineered features in 1.0s sliding windows (100ms step)
  8. Save combined dataset to CSV for model training and evaluation

Usage:
  python koro_ml_features.py
"""
import h5py, numpy as np, os, sys, warnings
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, welch, iirnotch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.io import wavfile

warnings.filterwarnings('ignore')

# Config
DATA_DIR = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
DATASET_CSV = os.path.join(OUTPUT_DIR, 'koro_ml_dataset.csv')

FS_RF = 10_000
FC_HZ = 0.9e9
C = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

# Feature extraction parameters
WIN_SIZE_S = 1.0     # 1.0-second window
STEP_SIZE_S = 0.1    # 100ms step (10 Hz feature rate)

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

    iq = iq_condition(apply_iq(ir, qr))
    phase = robust_phase(iq)

    for fn in [50.0, 100.0, 150.0]:
        b, a = iirnotch(fn, 30, fs)
        phase = signal.filtfilt(b, a, phase)

    # Korotkoff velocity (10-49 Hz)
    sos_k = butter(4, [10, 49], btype='band', fs=fs, output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vel_koro = np.append(np.diff(pk) * fs, 0) * SCALE

    # HR displacement (0.5-3 Hz)
    sos_h = butter(4, [0.5, 3.0], btype='band', fs=fs, output='sos')
    disp_hr = sosfiltfilt(sos_h, phase) * SCALE

    return t, vel_koro, disp_hr, phase, fs

def load_stethoscope(path):
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(path)
        audio = clip.to_soundarray()
        fs_audio = clip.fps
        clip.close()
    except Exception:
        wav_path = path.replace('.mp4', '.wav')
        if os.path.exists(wav_path):
            fs_audio, audio = wavfile.read(wav_path)
            audio = audio.astype(np.float64) / 32768.0
        else:
            return None, None, None, None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    t = np.arange(len(audio)) / fs_audio

    # Korotkoff band 20-200 Hz
    sos_k = butter(4, [20, 200], btype='band', fs=fs_audio, output='sos')
    koro_audio = sosfiltfilt(sos_k, audio)

    return t, audio, koro_audio, fs_audio

def find_acoustic_window(koro_aud, t_aud, fs_aud):
    """Detects stethoscope ground truth window using double-envelope RMS thresholding."""
    aud_env = np.abs(hilbert(koro_aud))
    
    # 0.5s RMS smoothed with 1.0s moving average
    w500 = int(fs_aud * 0.5)
    rms_500 = np.sqrt(pd.Series(aud_env).pow(2).rolling(w500, center=True).mean().fillna(0).values)
    smoothed = np.convolve(rms_500, np.ones(int(fs_aud))/fs_aud, mode='same')
    
    # Adaptive threshold on smoothed acoustic envelope
    noise_head = smoothed[:int(8 * fs_aud)]
    noise_tail = smoothed[-int(5 * fs_aud):] if len(smoothed) > int(15 * fs_aud) else noise_head
    noise = np.concatenate([noise_head, noise_tail])
    med = np.median(noise)
    mad = np.median(np.abs(noise - med)) * 1.4826
    thresh = med + 2.5 * mad
    
    active = smoothed > thresh
    if not np.any(active):
        return 10.0, 20.0  # Safe fallback
        
    # Find longest sustained active segment
    diff = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return 10.0, 20.0
        
    run_lengths = ends - starts
    best_idx = np.argmax(run_lengths)
    
    onset = t_aud[starts[best_idx]]
    offset = t_aud[min(ends[best_idx], len(t_aud)-1)]
    
    # Verify limits
    if offset - onset < 3.0:
        return 10.0, 20.0
        
    return onset, offset

# ------------------------------------------------------------------
# FEATURE ENGINEERING & EXTRACTION PIPELINE
# ------------------------------------------------------------------
def calculate_tkeo(x):
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t

def get_hjorth_parameters(x):
    """Computes Hjorth Activity, Mobility, and Complexity."""
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

def extract_session_features(rf_path, aud_path, session_name, subject):
    print(f"\n  Processing features for {session_name}...")
    t_rf, vel, disp, phase, fs_rf = load_rf(rf_path)
    t_aud, aud, koro_aud, fs_aud = load_stethoscope(aud_path)
    
    if t_aud is None:
        print(f"    [WARN] No stethoscope for {session_name}. Skipping.")
        return None
        
    # Get true stethoscope acoustic window
    steth_on, steth_off = find_acoustic_window(koro_aud, t_aud, fs_aud)
    
    # Compute cross-correlation lag to align stethoscope with RF
    rf_env_full = np.convolve(vel**2, np.ones(int(fs_rf * 2.0))/(fs_rf * 2.0), mode='same')
    rf_env_n = rf_env_full / (np.max(rf_env_full) + 1e-20)
    
    aud_env_full = np.convolve(koro_aud**2, np.ones(int(fs_aud * 2.0))/(fs_aud * 2.0), mode='same')
    aud_env_n = aud_env_full / (np.max(aud_env_full) + 1e-20)
    aud_env_rf = np.interp(t_rf, t_aud, aud_env_n)
    
    # Correlate between 5s and 50s
    s_idx, e_idx = int(5 * fs_rf), min(int(50 * fs_rf), len(rf_env_n))
    cc = np.correlate(rf_env_n[s_idx:e_idx], aud_env_rf[s_idx:e_idx], mode='full')
    lag_samples = np.argmax(cc) - len(rf_env_n[s_idx:e_idx]) + 1
    lag_sec = lag_samples / fs_rf
    
    # Lag-corrected target bounds on RF timeline
    rf_target_on = steth_on + lag_sec
    rf_target_off = steth_off + lag_sec
    print(f"    Steth Window: {steth_on:.2f}s - {steth_off:.2f}s | CC Lag: {lag_sec:.2f}s | RF Target: {rf_target_on:.2f}s - {rf_target_off:.2f}s")

    # Compute TKEO
    tkeo = calculate_tkeo(vel)
    
    # Compute Hilbert envelope
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
        
        # Binary target: 1 if window center is in the lag-corrected target window
        target = 1.0 if (rf_target_on <= w_time <= rf_target_off) else 0.0
        
        # Segment arrays
        v_seg = vel[s:e]
        d_seg = disp[s:e]
        p_seg = phase[s:e]
        t_seg = tkeo[s:e]
        h_seg = hilb[s:e]
        
        # 1. Vel_RMS
        vel_rms = np.sqrt(np.mean(v_seg**2))
        
        # 2. Vel_TKEO
        t_mean = np.mean(np.abs(t_seg))
        
        # 3. Vel_Kurtosis
        kurt = float(sp_kurtosis(v_seg))
        
        # 4. Vel_Hilbert
        h_mean = np.mean(h_seg)
        
        # 5. Vel_BandPower (10-49Hz vs. 2-10Hz & 49-80Hz)
        ff, pp = welch(v_seg, fs=fs_rf, nperseg=min(512, len(v_seg)))
        km = (ff >= 10) & (ff <= 49)
        nm = ((ff >= 2) & (ff < 10)) | ((ff > 49) & (ff <= 80))
        sp = np.mean(pp[km]) if np.any(km) else 1e-20
        np_ = np.mean(pp[nm]) if np.any(nm) else 1e-20
        band_power = sp / (np_ + 1e-20)
        
        # 6. Vel_STFT (using PSD sum in 10-49Hz band)
        stft_energy = np.sum(pp[km]) if np.any(km) else 0.0
        
        # 7-9. Hjorth Parameters
        activity, mobility, complexity = get_hjorth_parameters(v_seg)
        
        # 10. Phase_Fluctuation_RMS (low frequency detrended phase)
        p_rms = np.sqrt(np.mean(signal.detrend(p_seg)**2))
        
        # 11. Disp_HR_RMS
        d_rms = np.sqrt(np.mean(d_seg**2))
        
        # 12. Spectral Entropy
        psd_norm = pp / (np.sum(pp) + 1e-20)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))
        
        # 13. Spectral Centroid
        spec_centroid = np.sum(ff * psd_norm)
        
        # 14. Zero Crossing Rate (ZCR)
        zcr = np.sum(np.diff(np.sign(v_seg)) != 0) / WIN_SIZE_S
        
        feature_rows.append({
            'subject': subject,
            'session_name': session_name,
            'time': w_time,
            'feat_vel_rms': vel_rms,
            'feat_vel_tkeo': t_mean,
            'feat_vel_kurtosis': kurt,
            'feat_vel_hilbert': h_mean,
            'feat_vel_bandpower': band_power,
            'feat_vel_stft': stft_energy,
            'feat_hjorth_activity': activity,
            'feat_hjorth_mobility': mobility,
            'feat_hjorth_complexity': complexity,
            'feat_phase_rms': p_rms,
            'feat_disp_rms': d_rms,
            'feat_spec_entropy': spec_entropy,
            'feat_spec_centroid': spec_centroid,
            'feat_zcr': zcr,
            'target': target
        })
        
    return pd.DataFrame(feature_rows)

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    # Use standard characters to avoid Windows terminal crashes
    print("  KOROTKOFF ML FEATURE EXTRACTION PIPELINE")
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
    print(f"\nCompleted feature extraction successfully!")
    print(f"Total dataset shape: {dataset.shape}")
    print(f"Class Balance (Target = 1): {dataset['target'].mean() * 100:.2f}%")
    
    # Save to CSV
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    dataset.to_csv(DATASET_CSV, index=False)
    print(f"Saved dataset to -> {DATASET_CSV}")

if __name__ == '__main__':
    main()
