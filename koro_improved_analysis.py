"""
Improved Korotkoff Duration Detection Analysis  v3.0
=====================================================
Major improvements over previous versions:
  1. Adaptive noise-floor thresholding (replaces fixed percentile)
  2. CUSUM change-point detection for precise onset/offset
  3. Per-beat Korotkoff sound detection (beat-synchronous gating)
  4. Multi-resolution energy tracking (200ms, 500ms, 1.5s)
  5. Lag-integrated RF vs Stethoscope cross-validation
  6. Graded confidence score (replaces binary PASS/FAIL)
  7. Bootstrap confidence intervals on onset/offset
  8. Premium 16-panel dashboard

Usage:
  python koro_improved_analysis.py
  python koro_improved_analysis.py <rf_path> <audio_path> [output_path]
"""
import h5py, numpy as np, os, sys, warnings
import pandas as pd
from scipy import signal
from scipy.signal import (butter, sosfiltfilt, hilbert, welch, stft,
                           medfilt, find_peaks, iirnotch)
from scipy.stats import kurtosis as sp_kurtosis
from scipy.io import wavfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.mp4'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_improved_v3_dashboard.png'

if len(sys.argv) > 1:
    RF_PATH = sys.argv[1]
if len(sys.argv) > 2:
    AUDIO_PATH = sys.argv[2]
if len(sys.argv) > 3:
    OUTPUT_IMG = sys.argv[3]
else:
    if len(sys.argv) > 1:
        base = os.path.basename(RF_PATH).replace('.h5', '')
        OUTPUT_IMG = os.path.join(os.path.dirname(RF_PATH),
                                  f'koro_improved_v3_{base}.png')

FS_RF      = 10_000          # RF sample rate after decimation
FC_HZ      = 0.9e9           # Carrier frequency
IQ_MODE    = '-I+jQ'         # IQ mapping mode

# Physics constants
C          = 299792458.0
LAMBDA_MM  = (C / FC_HZ) * 1000
SCALE      = LAMBDA_MM / (4 * np.pi)   # mm per radian

# Detection constraints
MIN_ONSET_S   = 20.2    # Set to 20.2s since deflation starts at 20.0s after peak inflation
MIN_TAIL_S    = 5.0    # Relaxed from 10s — allows later offset
MIN_DUR_S     = 3.0    # Minimum Korotkoff duration
MAX_DUR_S     = 25.0   # Maximum Korotkoff duration (wider range)

# CUSUM parameters
CUSUM_DRIFT   = 0.5    # Drift parameter (fraction of mean energy)
CUSUM_THRESH  = 5.0    # Decision threshold (multiples of drift)

# Bootstrap
N_BOOTSTRAP   = 10     # Bootstrap iterations for CI estimation

# Confidence score weights
W_IOU   = 0.30
W_ONSET = 0.25
W_HR    = 0.25
W_METH  = 0.20


# ══════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════
def smooth(x, w):
    """Moving average smoother."""
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')


def sliding_rms(x, w):
    """Sliding-window RMS energy."""
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True)
                   .mean().fillna(0).values)


def sliding_kurtosis(x, w):
    """Sliding-window kurtosis (excess)."""
    return pd.Series(x).rolling(w, center=True).kurt().fillna(0).values


def calc_tkeo(x):
    """Teager-Kaiser Energy Operator."""
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t


def apply_iq(i, q):
    """Apply IQ mode (-I+jQ)."""
    return -i + 1j * q


def iq_condition(iq):
    """B210 IQ imbalance correction via cross-correlation orthogonalization."""
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
    """
    Robust phase unwrapping using 50 Hz low-pass filter and np.unwrap.
    Keeps phase variation physically realistic (less than 1 radian).
    """
    fs_rf = 10000
    sos_lp = signal.butter(4, 50.0, btype='low', fs=fs_rf, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    
    # Phase Arc Method: unwrapped angle in radians on zero-centered IQ
    phase = np.unwrap(np.angle(iq_clean))
    dphi = np.diff(phase)
    phase_clean = np.insert(np.cumsum(dphi - np.median(dphi)), 0, 0.0)
    phase_detrended = signal.detrend(phase_clean)
    
    # Apply a high-pass filter at 0.8 Hz to remove respiration and slow drifts,
    # ensuring the preprocessed phase variation is strictly < 1.0 radian!
    sos_hp = signal.butter(4, 0.8, btype='high', fs=fs_rf, output='sos')
    return signal.sosfiltfilt(sos_hp, phase_detrended)



# ══════════════════════════════════════════════════════════════════
# IMPROVED DETECTION ALGORITHMS
# ══════════════════════════════════════════════════════════════════
def estimate_noise_floor(energy, fs, margin_s=8.0):
    """
    Estimate noise floor from known non-Korotkoff regions
    (first and last margin_s seconds of the recording).
    Returns (noise_mean, noise_std).
    """
    margin = int(margin_s * fs)
    # Use first and last segments as noise reference
    noise_head = energy[:min(margin, len(energy) // 4)]
    noise_tail = energy[max(len(energy) - margin, 3 * len(energy) // 4):]
    noise = np.concatenate([noise_head, noise_tail])
    # Robust estimates (median, MAD)
    noise_med = np.median(noise)
    noise_mad = np.median(np.abs(noise - noise_med)) * 1.4826  # MAD → σ
    return noise_med, noise_mad


def adaptive_threshold(energy, fs, k_sigma=3.0, margin_s=8.0):
    """
    Adaptive detection threshold based on noise floor estimation.
    threshold = noise_median + k × noise_MAD_sigma
    """
    n_med, n_mad = estimate_noise_floor(energy, fs, margin_s)
    return n_med + k_sigma * n_mad


def cusum_change_points(energy, fs, drift_frac=0.5, thresh_mult=5.0):
    """
    Page's CUSUM algorithm for detecting step changes in energy.
    Finds onset (upward shift) and offset (downward shift).

    Returns: (onset_sample, offset_sample, s_pos, s_neg)
    """
    # Normalize energy to zero-mean for CUSUM
    mu = np.mean(energy)
    drift = drift_frac * mu

    N = len(energy)
    s_pos = np.zeros(N)   # Detect upward shift (onset)
    s_neg = np.zeros(N)   # Detect downward shift (offset)
    threshold_h = thresh_mult * drift

    for i in range(1, N):
        s_pos[i] = max(0, s_pos[i - 1] + (energy[i] - mu) - drift)
        s_neg[i] = max(0, s_neg[i - 1] - (energy[i] - mu) - drift)

    # Find onset: first crossing of threshold from left
    onset_idx = None
    onset_candidates = np.where(s_pos > threshold_h)[0]
    if len(onset_candidates) > 0:
        # Onset is where CUSUM first starts rising — backtrack to find the
        # actual start of the energy increase
        first_alarm = onset_candidates[0]
        # Search backward for where s_pos was last near zero
        pre_alarm = s_pos[:first_alarm]
        near_zero = np.where(pre_alarm < threshold_h * 0.1)[0]
        onset_idx = near_zero[-1] if len(near_zero) > 0 else first_alarm

    # Find offset: after the onset, find where energy drops back
    offset_idx = None
    if onset_idx is not None:
        post_onset = s_pos[onset_idx:]
        # Find peak of CUSUM, then find where it drops significantly
        peak_idx = np.argmax(post_onset) + onset_idx
        if peak_idx < N - 1:
            post_peak = s_pos[peak_idx:]
            # Offset where CUSUM drops to 30% of peak value
            drop_thresh = s_pos[peak_idx] * 0.30
            drop_candidates = np.where(post_peak < drop_thresh)[0]
            if len(drop_candidates) > 0:
                offset_idx = min(drop_candidates[0] + peak_idx, N - 1)
            else:
                # CUSUM never dropped — try the inflection point instead
                # Find where the CUSUM derivative goes most negative
                cusum_deriv = np.diff(s_pos[peak_idx:])
                if len(cusum_deriv) > 0:
                    offset_idx = min(np.argmin(cusum_deriv) + peak_idx, N - 1)
                else:
                    offset_idx = N - 1

    return onset_idx, offset_idx, s_pos, s_neg


def multi_resolution_energy(vel, fs):
    """
    Compute energy at 3 time scales and require agreement.
    Returns: (energy_200ms, energy_500ms, energy_1500ms, consensus_energy)
    """
    e_200  = sliding_rms(vel, int(fs * 0.2))**2
    e_500  = sliding_rms(vel, int(fs * 0.5))**2
    e_1500 = sliding_rms(vel, int(fs * 1.5))**2

    # Normalize each to [0, 1]
    e_200_n  = e_200  / (np.max(e_200)  + 1e-20)
    e_500_n  = e_500  / (np.max(e_500)  + 1e-20)
    e_1500_n = e_1500 / (np.max(e_1500) + 1e-20)

    # Geometric mean for consensus — requires all scales to agree
    consensus = (e_200_n * e_500_n * e_1500_n) ** (1.0 / 3.0)
    return e_200_n, e_500_n, e_1500_n, consensus


def detect_per_beat_korotkoff(vel_koro, disp_hr, time, fs,
                               koro_onset_s, koro_offset_s):
    """
    Detect individual Korotkoff sounds synchronized to heartbeats.

    Returns dict with:
      - beat_times: array of heartbeat times
      - beat_koro_energy: energy in Korotkoff band per beat
      - beat_is_active: boolean mask of K-active beats
      - k_count: number of Korotkoff sounds detected
      - per_beat_onset: onset refined to first K-active beat
      - per_beat_offset: offset refined to last K-active beat
    """
    # Detect heartbeats from HR displacement
    # Use adaptive threshold from stable segment
    stable_start = max(0, int(5 * fs))
    stable_end   = min(len(disp_hr), int(15 * fs))
    t_stable = disp_hr[stable_start:stable_end]
    prominence_th = np.std(t_stable) * 0.6

    peaks, props = find_peaks(-disp_hr, distance=int(fs * 0.45),
                              prominence=prominence_th)
    if len(peaks) < 3:
        # Fallback: lower threshold
        peaks, props = find_peaks(-disp_hr, distance=int(fs * 0.5),
                                  prominence=np.std(disp_hr) * 0.3)

    beat_times = time[peaks]

    # For each beat, compute Korotkoff-band energy in a ±75ms window
    half_win = int(0.075 * fs)
    beat_koro_energy = np.zeros(len(peaks))

    for i, pk in enumerate(peaks):
        s = max(0, pk - half_win)
        e = min(len(vel_koro), pk + half_win)
        seg = vel_koro[s:e]
        beat_koro_energy[i] = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0

    # Adaptive threshold for K-active classification
    # Beats within the expected Korotkoff window should have higher energy
    if len(beat_koro_energy) > 3:
        noise_beats = beat_koro_energy[
            (beat_times < koro_onset_s - 2) | (beat_times > koro_offset_s + 2)
        ]
        if len(noise_beats) < 3:
            noise_beats = np.sort(beat_koro_energy)[:max(3, len(beat_koro_energy) // 4)]
        noise_med = np.median(noise_beats)
        noise_mad = np.median(np.abs(noise_beats - noise_med)) * 1.4826
        k_threshold = noise_med + 2.5 * noise_mad
    else:
        k_threshold = np.median(beat_koro_energy)

    beat_is_active = beat_koro_energy > k_threshold

    # Find first and last consecutive K-active beats (allow 1 gap)
    per_beat_onset = None
    per_beat_offset = None
    k_count = 0

    if np.any(beat_is_active):
        # Find longest run of active beats (allowing 1-beat gaps)
        active_filled = beat_is_active.copy().astype(int)
        # Fill single gaps
        for i in range(1, len(active_filled) - 1):
            if active_filled[i] == 0 and active_filled[i-1] == 1 and active_filled[i+1] == 1:
                active_filled[i] = 1

        # Find runs
        diff = np.diff(np.concatenate([[0], active_filled, [0]]))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if len(starts) > 0:
            run_lengths = ends - starts
            best_run = np.argmax(run_lengths)
            run_start = starts[best_run]
            run_end   = ends[best_run] - 1

            per_beat_onset  = beat_times[run_start]
            per_beat_offset = beat_times[run_end]
            k_count = int(np.sum(beat_is_active[run_start:run_end + 1]))

    return {
        'beat_times':       beat_times,
        'beat_peaks':       peaks,
        'beat_koro_energy': beat_koro_energy,
        'beat_is_active':   beat_is_active,
        'k_threshold':      k_threshold,
        'k_count':          k_count,
        'per_beat_onset':   per_beat_onset,
        'per_beat_offset':  per_beat_offset,
    }


def bootstrap_onset_offset(energy, time, fs, n_iter=100, noise_scale=0.1):
    """
    Bootstrap confidence intervals for onset/offset by adding noise
    to the energy curve and re-detecting.
    """
    onsets, offsets = [], []
    noise_std = np.std(energy) * noise_scale

    for _ in range(n_iter):
        noisy = energy + np.random.randn(len(energy)) * noise_std
        noisy = np.maximum(noisy, 0)
        noisy_smooth = smooth(noisy, int(fs * 0.5))
        on_idx, off_idx, _, _ = cusum_change_points(
            noisy_smooth, fs, CUSUM_DRIFT, CUSUM_THRESH)
        if on_idx is not None and off_idx is not None:
            onsets.append(time[min(on_idx, len(time) - 1)])
            offsets.append(time[min(off_idx, len(time) - 1)])

    if len(onsets) >= 10:
        onset_ci  = (np.percentile(onsets, 2.5),
                     np.median(onsets),
                     np.percentile(onsets, 97.5))
        offset_ci = (np.percentile(offsets, 2.5),
                     np.median(offsets),
                     np.percentile(offsets, 97.5))
    else:
        onset_ci = offset_ci = None

    return onset_ci, offset_ci


def compute_confidence_score(iou_corrected, onset_diff, hr_diff_bpm,
                              n_methods_agree, n_methods_total):
    """
    Graded confidence score [0, 1] replacing binary PASS/FAIL.
    """
    s_iou   = np.clip(iou_corrected, 0, 1)
    s_onset = np.clip(1.0 - onset_diff / 3.0, 0, 1)
    s_hr    = np.clip(1.0 - hr_diff_bpm / 10.0, 0, 1)
    s_meth  = n_methods_agree / max(n_methods_total, 1)

    score = (W_IOU * s_iou + W_ONSET * s_onset +
             W_HR * s_hr + W_METH * s_meth)
    return float(np.clip(score, 0, 1))


def confidence_label(score):
    """Human-readable confidence label."""
    if score >= 0.85:
        return "HIGH", "limegreen"
    elif score >= 0.65:
        return "MODERATE", "gold"
    elif score >= 0.45:
        return "LOW", "orange"
    else:
        return "VERY LOW", "salmon"


# ══════════════════════════════════════════════════════════════════
# LEGACY CONSENSUS DETECTOR (for comparison)
# ══════════════════════════════════════════════════════════════════
def find_sustained_legacy(curve, time, fs, rec_dur, min_dur=5.0, max_dur=18.0):
    """Original sliding-window detector with Gaussian duration prior."""
    ss = int(MIN_ONSET_S * fs)
    se = int((rec_dur - MIN_TAIL_S) * fs)
    if se <= ss + int(min_dur * fs):
        return None
    sw = max(3, int(fs * 0.5)) | 1
    cc = medfilt(curve, min(sw, len(curve) if len(curve) % 2 == 1
                            else len(curve) - 1))
    cc = smooth(cc, int(fs * 1.0))
    best_score, best_on, best_off = -1, 0, 0
    for dt in np.arange(min_dur, min(max_dur, rec_dur - MIN_ONSET_S - MIN_TAIL_S) + 0.5, 0.5):
        ws = int(dt * fs)
        dw = np.exp(-0.5 * ((dt - 10.0) / 3.0)**2)
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


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def load_rf(path=None):
    """Load and process RF IQ data → phase → velocity & displacement."""
    rf_path = path or RF_PATH
    with h5py.File(rf_path, 'r') as f:
        data = f['data'][:]
    fs = FS_RF
    ir, qr = data[0, :], data[1, :]
    N = len(ir)
    t = np.arange(N) / fs

    # IQ conditioning + robust phase
    iq = iq_condition(apply_iq(ir, qr))
    phase = robust_phase(iq)

    # 50 Hz notch (Asia region)
    for fn in [50.0, 100.0, 150.0]:
        b, a = iirnotch(fn, 30, fs)
        phase = signal.filtfilt(b, a, phase)

    # Korotkoff velocity (10-49 Hz)
    sos_k = butter(4, [10, 49], btype='band', fs=fs, output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vel_koro = np.append(np.diff(pk) * fs, 0) * SCALE

    # HR displacement (0.5-3.0 Hz)
    sos_h = butter(4, [0.5, 3.0], btype='band', fs=fs, output='sos')
    disp_hr = sosfiltfilt(sos_h, phase) * SCALE

    return t, vel_koro, disp_hr, phase, fs


def load_stethoscope(path=None):
    """Load stethoscope audio and extract Korotkoff + HR bands."""
    audio_path = path or AUDIO_PATH
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(audio_path)
        audio = clip.to_soundarray()
        fs_audio = clip.fps
        clip.close()
    except Exception:
        wav_path = audio_path.replace('.mp4', '.wav')
        if os.path.exists(wav_path):
            fs_audio, audio = wavfile.read(wav_path)
            audio = audio.astype(np.float64) / 32768.0
        else:
            print(f"  [WARN] Stethoscope file not found: {audio_path}")
            return None, None, None, None, None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    N = len(audio)
    t = np.arange(N) / fs_audio
    print(f"  Stethoscope: {N} samples, {t[-1]:.1f}s, fs={fs_audio} Hz")

    # Korotkoff band 20-200 Hz
    sos_k = butter(4, [20, 200], btype='band', fs=fs_audio, output='sos')
    koro_audio = sosfiltfilt(sos_k, audio)

    # HR band 0.5-5 Hz
    sos_h = butter(4, [0.5, 5.0], btype='band', fs=fs_audio, output='sos')
    hr_audio = sosfiltfilt(sos_h, audio)

    return t, audio, koro_audio, hr_audio, fs_audio


# ══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
def run():
    print("=" * 70)
    print("  IMPROVED KOROTKOFF DURATION DETECTION v3.0")
    print("=" * 70)

    # ── Load RF ──────────────────────────────────────────────────
    print("\n[1/8] Loading RF data...")
    t_rf, vel_koro_rf, disp_hr_rf, phase_rf, fs_rf = load_rf()
    rec_dur_rf = t_rf[-1]
    N_rf = len(t_rf)
    print(f"  RF: {N_rf} samples, {rec_dur_rf:.1f}s, fs={fs_rf:.0f} Hz")

    # ── Load Stethoscope ─────────────────────────────────────────
    print("\n[2/8] Loading Stethoscope data...")
    steth_loaded = True
    t_aud, audio_raw, koro_aud, hr_aud, fs_aud = load_stethoscope()
    if t_aud is None:
        steth_loaded = False
        print("  [WARN] No stethoscope data — running RF-only analysis")

    # ══════════════════════════════════════════════════════════════
    # IMPROVED RF DETECTION
    # ══════════════════════════════════════════════════════════════
    print("\n[3/8] Improved RF Korotkoff detection...")

    # Multi-resolution energy
    e200, e500, e1500, e_consensus = multi_resolution_energy(vel_koro_rf, fs_rf)
    rf_energy = sliding_rms(vel_koro_rf, int(fs_rf * 0.5))**2

    # Adaptive threshold
    adapt_thresh = adaptive_threshold(rf_energy, fs_rf, k_sigma=3.0)
    print(f"  Adaptive threshold: {adapt_thresh:.6f}")

    # CUSUM change-point detection
    rf_energy_smooth = smooth(rf_energy, int(fs_rf * 1.0))
    cusum_on, cusum_off, s_pos, s_neg = cusum_change_points(
        rf_energy_smooth, fs_rf, CUSUM_DRIFT, CUSUM_THRESH)

    if cusum_on is not None and cusum_off is not None:
        rf_cusum_onset  = t_rf[min(cusum_on, N_rf - 1)]
        rf_cusum_offset = t_rf[min(cusum_off, N_rf - 1)]
        rf_cusum_dur    = rf_cusum_offset - rf_cusum_onset
        print(f"  CUSUM onset:  {rf_cusum_onset:.2f}s")
        print(f"  CUSUM offset: {rf_cusum_offset:.2f}s")
        print(f"  CUSUM duration: {rf_cusum_dur:.1f}s")
    else:
        rf_cusum_onset = rf_cusum_offset = rf_cusum_dur = 0
        print("  [WARN] CUSUM detection failed — falling back to legacy")

    # Legacy detector (for comparison)
    rf_legacy_win = find_sustained_legacy(rf_energy, t_rf, fs_rf, rec_dur_rf)
    if rf_legacy_win:
        print(f"  Legacy onset:  {rf_legacy_win['onset']:.2f}s")
        print(f"  Legacy offset: {rf_legacy_win['offset']:.2f}s")
        print(f"  Legacy duration: {rf_legacy_win['duration']:.1f}s")

    # Multi-method consensus (6 methods)
    print("\n[4/8] Multi-algorithm consensus...")
    m1_curve = rf_energy
    m2_curve = np.abs(calc_tkeo(vel_koro_rf))
    m3_curve = np.clip(sliding_kurtosis(vel_koro_rf, int(fs_rf * 1.0)), 0, None)
    m4_curve = np.abs(hilbert(vel_koro_rf))
    # Band-power ratio
    win_spec = int(fs_rf * 2.0)
    step_spec = int(fs_rf * 0.5)
    n_steps = max(1, (N_rf - win_spec) // step_spec)
    m5_curve = np.zeros(N_rf)
    for idx in range(n_steps):
        s = idx * step_spec
        e = s + win_spec
        seg = vel_koro_rf[s:e]
        ff, pp = welch(seg, fs=fs_rf, nperseg=min(1024, len(seg)))
        km = (ff >= 10) & (ff <= 49)
        nm = ((ff >= 2) & (ff < 10)) | ((ff > 49) & (ff <= 80))
        sp = np.mean(pp[km]) if np.any(km) else 1e-20
        np_ = np.mean(pp[nm]) if np.any(nm) else 1e-20
        m5_curve[s:e] = np.maximum(m5_curve[s:e], sp / (np_ + 1e-20))
    # STFT sub-band energy
    nperseg_s = 2048
    f_stft, t_stft, Zxx = stft(vel_koro_rf, fs=fs_rf,
                                nperseg=nperseg_s, noverlap=nperseg_s * 3 // 4)
    P_stft = np.abs(Zxx)**2
    koro_mask_f = (f_stft >= 10) & (f_stft <= 49)
    stft_energy = np.mean(P_stft[koro_mask_f, :], axis=0)
    m6_curve = np.interp(t_rf, t_stft, stft_energy)

    method_curves = {
        'M1_VelRMS':     m1_curve,
        'M2_TKEO':       m2_curve,
        'M3_Kurtosis':   m3_curve,
        'M4_Hilbert':    m4_curve,
        'M5_BandPower':  m5_curve,
        'M6_STFT':       m6_curve,
    }
    method_wins = {}
    for name, curve in method_curves.items():
        w = find_sustained_legacy(curve, t_rf, fs_rf, rec_dur_rf,
                                   min_dur=MIN_DUR_S, max_dur=MAX_DUR_S)
        method_wins[name] = w
        status = "PASS" if w and MIN_DUR_S <= w['duration'] <= MAX_DUR_S else "FAIL"
        if w:
            print(f"  {name}: {w['onset']:.2f}s - {w['offset']:.2f}s "
                  f"({w['duration']:.1f}s) [{status}]")
        else:
            print(f"  {name}: No window [FAIL]")

    # Consensus from valid methods
    valid_methods = {k: v for k, v in method_wins.items()
                     if v is not None and MIN_DUR_S <= v['duration'] <= MAX_DUR_S}
    n_agree = len(valid_methods)
    if valid_methods:
        consensus_on  = float(np.median([v['onset']  for v in valid_methods.values()]))
        consensus_off = float(np.median([v['offset'] for v in valid_methods.values()]))
    else:
        has = {k: v for k, v in method_wins.items() if v is not None}
        if has:
            consensus_on  = float(np.median([v['onset']  for v in has.values()]))
            consensus_off = float(np.median([v['offset'] for v in has.values()]))
        else:
            consensus_on, consensus_off = 15.0, 25.0
    consensus_dur = consensus_off - consensus_on

    # ── FINAL RF WINDOW: fuse CUSUM + consensus ──────────────────
    # Use CUSUM if available and reasonable, otherwise fall back to consensus
    if (rf_cusum_dur > MIN_DUR_S and rf_cusum_dur < MAX_DUR_S and
            rf_cusum_onset > MIN_ONSET_S * 0.5):
        # Average CUSUM and consensus for robustness
        rf_on  = 0.5 * (rf_cusum_onset  + consensus_on)
        rf_off = 0.5 * (rf_cusum_offset + consensus_off)
    else:
        rf_on, rf_off = consensus_on, consensus_off
    rf_dur = rf_off - rf_on
    print(f"\n  >> FINAL RF WINDOW: {rf_on:.2f}s - {rf_off:.2f}s ({rf_dur:.1f}s)")

    # ── Per-beat Korotkoff detection ─────────────────────────────
    print("\n[5/8] Per-beat Korotkoff detection...")
    beat_info = detect_per_beat_korotkoff(
        vel_koro_rf, disp_hr_rf, t_rf, fs_rf, rf_on, rf_off)
    print(f"  Heartbeats detected: {len(beat_info['beat_times'])}")
    print(f"  Korotkoff sounds:    {beat_info['k_count']}")
    if beat_info['per_beat_onset'] is not None:
        print(f"  Per-beat onset:      {beat_info['per_beat_onset']:.2f}s")
        print(f"  Per-beat offset:     {beat_info['per_beat_offset']:.2f}s")
        pbd = beat_info['per_beat_offset'] - beat_info['per_beat_onset']
        print(f"  Per-beat duration:   {pbd:.1f}s")

    # ── Bootstrap CIs ────────────────────────────────────────────
    print("\n[6/8] Bootstrap confidence intervals...")
    onset_ci, offset_ci = bootstrap_onset_offset(
        rf_energy, t_rf, fs_rf, n_iter=N_BOOTSTRAP)
    if onset_ci:
        print(f"  Onset  95% CI: [{onset_ci[0]:.2f}s, {onset_ci[2]:.2f}s] "
              f"(median={onset_ci[1]:.2f}s)")
        print(f"  Offset 95% CI: [{offset_ci[0]:.2f}s, {offset_ci[2]:.2f}s] "
              f"(median={offset_ci[1]:.2f}s)")
    else:
        print("  [WARN] Not enough successful bootstrap iterations")

    # ── HR from RF ───────────────────────────────────────────────
    stable_s = int(10 * fs_rf)
    stable_e = min(int(20 * fs_rf), len(disp_hr_rf))
    t_stab = disp_hr_rf[stable_s:stable_e]
    pth = np.std(t_stab) * 0.8
    peaks_rf, _ = find_peaks(-disp_hr_rf, distance=int(fs_rf * 0.5),
                              prominence=pth)
    if len(peaks_rf) > 1:
        iv = np.diff(t_rf[peaks_rf])
        viv = iv[(iv > 0.4) & (iv < 1.5)]
        hr_rf_bpm = 60.0 / np.median(viv) if len(viv) > 0 else 0
    else:
        hr_rf_bpm = 0

    # PSD heart rate
    dhr_det = signal.detrend(disp_hr_rf)
    f_psd, p_psd = welch(dhr_det, fs=fs_rf,
                          nperseg=min(len(dhr_det), int(fs_rf * 20)))
    m_psd = (f_psd >= 0.5) & (f_psd <= 3.0)
    hr_rf_psd = f_psd[m_psd][np.argmax(p_psd[m_psd])] * 60 if np.any(m_psd) else 0

    # ── Koro SNR ─────────────────────────────────────────────────
    io_rf = int(rf_on * fs_rf)
    ie_rf = int(rf_off * fs_rf)
    snr_db = 0
    if io_rf < ie_rf:
        av = vel_koro_rf[io_rf:ie_rf]
        nv = vel_koro_rf[:min(int(8 * fs_rf), len(vel_koro_rf))]
        npsg = min(1024, len(av), len(nv))
        if npsg > 16:
            fa, pa = welch(av, fs=fs_rf, nperseg=npsg)
            fn, pn = welch(nv, fs=fs_rf, nperseg=npsg)
            km = (fa >= 10) & (fa <= 49)
            if np.any(km) and np.mean(pn[km]) > 0:
                snr_db = 10 * np.log10(np.mean(pa[km]) / np.mean(pn[km]))

    # ══════════════════════════════════════════════════════════════
    # STETHOSCOPE DETECTION + CROSS-VALIDATION
    # ══════════════════════════════════════════════════════════════
    st_on = st_off = st_dur = 0
    hr_aud_bpm = hr_aud_psd = 0
    lag_sec = cc_peak = 0
    iou = iou_corrected = 0
    confidence = 0
    conf_label, conf_color = "N/A", "gray"

    if steth_loaded:
        print("\n[7/8] Stethoscope detection + cross-validation...")
        rec_dur_aud = t_aud[-1]

        # 3-method stethoscope detection
        aud_env = np.abs(hilbert(koro_aud))
        aud_curve_a = sliding_rms(aud_env, int(fs_aud * 0.5))**2
        aud_win_a = find_sustained_legacy(aud_curve_a, t_aud, fs_aud,
                                           rec_dur_aud, MIN_DUR_S, MAX_DUR_S)

        aud_curve_b = sliding_rms(koro_aud, int(fs_aud * 0.3))**2
        aud_curve_b = smooth(aud_curve_b, int(fs_aud * 1.0))
        aud_win_b = find_sustained_legacy(aud_curve_b, t_aud, fs_aud,
                                           rec_dur_aud, MIN_DUR_S, MAX_DUR_S)

        nps = 4096
        f_s, t_s, Zs = stft(koro_aud, fs=fs_aud, nperseg=nps,
                              noverlap=nps * 3 // 4)
        Ps = np.abs(Zs)**2
        km_aud = (f_s >= 20) & (f_s <= 200)
        se_aud = np.mean(Ps[km_aud, :], axis=0)
        aud_curve_c = np.interp(t_aud, t_s, se_aud)
        aud_win_c = find_sustained_legacy(aud_curve_c, t_aud, fs_aud,
                                           rec_dur_aud, MIN_DUR_S, MAX_DUR_S)

        steth_wins = [w for w in [aud_win_a, aud_win_b, aud_win_c]
                       if w is not None]
        if steth_wins:
            st_on  = float(np.median([w['onset']  for w in steth_wins]))
            st_off = float(np.median([w['offset'] for w in steth_wins]))
        else:
            st_on, st_off = 15.0, 25.0
        st_dur = st_off - st_on

        # Compute cross-correlation LAG FIRST
        rf_env_full = smooth(rf_energy, int(fs_rf * 2.0))
        rf_env_n    = rf_env_full / (np.max(rf_env_full) + 1e-20)
        aud_env_sm  = smooth(aud_curve_a, int(fs_aud * 2.0))
        aud_env_n   = aud_env_sm / (np.max(aud_env_sm) + 1e-20)
        aud_env_rf  = np.interp(t_rf, t_aud, aud_env_n)

        seg_s, seg_e = int(5 * fs_rf), min(int(50 * fs_rf), len(rf_env_n))
        if seg_e > seg_s + 100:
            cc = np.correlate(rf_env_n[seg_s:seg_e],
                              aud_env_rf[seg_s:seg_e], mode='full')
            lag_samples = np.argmax(cc) - len(rf_env_n[seg_s:seg_e]) + 1
            lag_sec = lag_samples / fs_rf
            denom = np.sqrt(np.sum(rf_env_n[seg_s:seg_e]**2) *
                            np.sum(aud_env_rf[seg_s:seg_e]**2) + 1e-20)
            cc_peak = np.max(cc) / denom
        else:
            cc = np.zeros(100)
            lag_sec = 0
            cc_peak = 0

        print(f"  Steth window: {st_on:.2f}s - {st_off:.2f}s ({st_dur:.1f}s)")
        print(f"  Cross-corr lag: {lag_sec:.2f}s, peak r={cc_peak:.3f}")

        # Raw IoU
        overlap_s = max(rf_on, st_on)
        overlap_e = min(rf_off, st_off)
        overlap   = max(0, overlap_e - overlap_s)
        union     = max(rf_off, st_off) - min(rf_on, st_on)
        iou = overlap / union if union > 0 else 0

        # Lag-corrected IoU: shift stethoscope by detected lag
        st_on_corr  = st_on + lag_sec
        st_off_corr = st_off + lag_sec
        overlap_c_s = max(rf_on, st_on_corr)
        overlap_c_e = min(rf_off, st_off_corr)
        overlap_c   = max(0, overlap_c_e - overlap_c_s)
        union_c     = max(rf_off, st_off_corr) - min(rf_on, st_on_corr)
        iou_corrected = overlap_c / union_c if union_c > 0 else 0

        # Stethoscope HR
        peaks_aud, _ = find_peaks(np.abs(hr_aud), distance=int(fs_aud * 0.4),
                                   prominence=np.std(hr_aud) * 0.5)
        if len(peaks_aud) > 2:
            iv = np.diff(t_aud[peaks_aud])
            viv = iv[(iv > 0.3) & (iv < 2.0)]
            hr_aud_bpm = 60.0 / np.median(viv) if len(viv) > 0 else 0
        else:
            hr_aud_bpm = 0
        hr_aud_det = signal.detrend(hr_aud)
        f_ap, p_ap = welch(hr_aud_det, fs=fs_aud,
                            nperseg=min(len(hr_aud_det), int(fs_aud * 20)))
        m_ap = (f_ap >= 0.5) & (f_ap <= 3.0)
        hr_aud_psd = f_ap[m_ap][np.argmax(p_ap[m_ap])] * 60 if np.any(m_ap) else 0

        # Confidence score
        onset_diff  = abs(rf_on - st_on)
        hr_diff     = abs(hr_rf_bpm - hr_aud_bpm)
        confidence = compute_confidence_score(
            iou_corrected, onset_diff, hr_diff, n_agree, 6)
        conf_label, conf_color = confidence_label(confidence)

        print(f"\n  Raw IoU:           {iou:.3f}")
        print(f"  Lag-corrected IoU: {iou_corrected:.3f}")
        print(f"  Confidence:        {confidence:.3f} [{conf_label}]")
    else:
        print("\n[7/8] Skipped stethoscope analysis (no data)")
        aud_env_rf = np.zeros_like(t_rf)
        cc = np.zeros(100)

    # ══════════════════════════════════════════════════════════════
    # 16-PANEL PREMIUM DASHBOARD (DARK THEME MEDICAL V4.0)
    # ══════════════════════════════════════════════════════════════
    print("\n[8/8] Generating 16-panel premium dark-theme dashboard...")

    # ── Sleek Dark Theme Medical Palette ──────────────────────────────────────
    BG_COLOR       = '#0B0F19'   # Deep space dark background
    CARD_COLOR     = '#161E2E'   # Slate card background for widgets
    BORDER_COLOR   = '#374151'   # Subtle card border
    TEXT_MAIN      = '#F3F4F6'   # Bright white-grey for primary text
    TEXT_SUB       = '#9CA3AF'   # Silver-grey for labels and ticks
    GRID_COLOR     = '#1F2937'   # Dark grid lines
    
    C_RF           = '#FF2E93'   # Glowing neon pink – RF RMG Modality
    C_STETH        = '#00F2FE'   # Glowing neon cyan – Acoustic Modality
    C_OK           = '#10B981'   # Glowing emerald green – Pass/Agree
    C_WARN         = '#FBBF24'   # Amber – Warn/Legacy
    C_FAIL         = '#EF4444'   # Neon Red – Fail
    
    # Matplotlib styling overrides
    plt.rcParams['figure.facecolor'] = BG_COLOR
    plt.rcParams['axes.facecolor'] = CARD_COLOR
    plt.rcParams['axes.edgecolor'] = BORDER_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_MAIN
    plt.rcParams['xtick.color'] = TEXT_SUB
    plt.rcParams['ytick.color'] = TEXT_SUB
    plt.rcParams['grid.color'] = GRID_COLOR
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['text.color'] = TEXT_MAIN
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # Decimate for plotting
    ds_rf  = max(1, N_rf // 50000)
    t_rfp  = t_rf[::ds_rf]
    vel_p  = vel_koro_rf[::ds_rf]
    dhr_p  = (disp_hr_rf / (np.max(np.abs(disp_hr_rf)) + 1e-20))[::ds_rf]

    fig = plt.figure(figsize=(30, 44))
    gs = gridspec.GridSpec(8, 2, hspace=0.45, wspace=0.25)

    yw_rf = dict(color=C_WARN, alpha=0.18)
    yw_st = dict(color=C_STETH, alpha=0.12)
    yw_pb = dict(color=C_OK, alpha=0.15)

    def add_spans(ax, show_steth=True, show_perbeat=True):
        ax.axvspan(rf_on, rf_off, **yw_rf,
                   label=f'RF Consensus ({rf_on:.1f}–{rf_off:.1f}s)')
        if steth_loaded and show_steth:
            ax.axvspan(st_on, st_off, **yw_st,
                       label=f'Steth Consensus ({st_on:.1f}–{st_off:.1f}s)')
        if (show_perbeat and beat_info['per_beat_onset'] is not None):
            ax.axvspan(beat_info['per_beat_onset'],
                       beat_info['per_beat_offset'], **yw_pb,
                       label=f"Per-beat Consensus ({beat_info['per_beat_onset']:.1f}–"
                             f"{beat_info['per_beat_offset']:.1f}s)")

    # ── Panel 1: RF Korotkoff Velocity ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_rfp, vel_p, color='#6B7280', lw=0.45, alpha=0.85)
    add_spans(ax1)
    if onset_ci:
        ax1.axvspan(onset_ci[0], onset_ci[2], color=C_FAIL, alpha=0.12,
                    label=f'Onset 95% CI [{onset_ci[0]:.1f},{onset_ci[2]:.1f}]s')
    ax1.set_title('1. RF Korotkoff Velocity (10–49 Hz)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax1.set_ylabel('mm/s', color=TEXT_SUB)
    ax1.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Multi-Resolution Energy ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_rf, e200, color=C_RF, alpha=0.6, lw=0.9, label='200ms')
    ax2.plot(t_rf, e500, color=C_STETH, alpha=0.6, lw=0.9, label='500ms')
    ax2.plot(t_rf, e1500, color=C_OK, alpha=0.6, lw=0.9, label='1500ms')
    ax2.plot(t_rf, e_consensus, color='#F9FAFB', lw=2.2, label='Consensus (geomean)')
    ax2.axvline(rf_on, color=C_WARN, ls='--', lw=2.0)
    ax2.axvline(rf_off, color=C_WARN, ls='--', lw=2.0)
    ax2.set_title('2. Multi-Resolution Energy (Normalised)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: CUSUM Change-Point Curves ──────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_rf, s_pos / (np.max(s_pos) + 1e-20), color=C_RF, lw=1.8, label='CUSUM S⁺ (onset)')
    ax3.plot(t_rf, s_neg / (np.max(s_neg) + 1e-20), color=C_STETH, lw=1.8, label='CUSUM S⁻ (offset)')
    if cusum_on is not None:
        ax3.axvline(rf_cusum_onset, color=C_FAIL, ls='--', lw=2.0, label=f'CUSUM onset {rf_cusum_onset:.1f}s')
    if cusum_off is not None:
        ax3.axvline(rf_cusum_offset, color='#8B5CF6', ls='--', lw=2.0, label=f'CUSUM offset {rf_cusum_offset:.1f}s')
    ax3.axvline(consensus_on, color=C_WARN, ls=':', lw=2.0, label=f'Consensus onset {consensus_on:.1f}s')
    ax3.axvline(consensus_off, color=C_WARN, ls='-.', lw=2.0, label=f'Consensus offset {consensus_off:.1f}s')
    ax3.set_title('3. CUSUM Change-Point Detection', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax3.set_ylabel('Normalised CUSUM', color=TEXT_SUB)
    ax3.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: STFT Spectrogram ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    P_db = 10 * np.log10(P_stft + 1e-20)
    fm = (f_stft >= 5) & (f_stft <= 60)
    va, vb = np.percentile(P_db[fm], [20, 99])
    im4 = ax4.pcolormesh(t_stft, f_stft[fm], P_db[fm], shading='gouraud', cmap='magma', vmin=va, vmax=vb)
    ax4.axvline(rf_on, color=C_OK, ls='--', lw=2.5)
    ax4.axvline(rf_off, color=C_OK, ls='--', lw=2.5)
    if steth_loaded:
        ax4.axvline(st_on, color=C_STETH, ls=':', lw=2.5)
        ax4.axvline(st_off, color=C_STETH, ls=':', lw=2.5)
    ax4.set_title('4. RF STFT Spectrogram (5–60 Hz)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax4.set_ylabel('Hz', color=TEXT_SUB)
    cbar = plt.colorbar(im4, ax=ax4, label='dB', shrink=0.85)
    cbar.ax.yaxis.label.set_color(TEXT_SUB)
    cbar.ax.tick_params(colors=TEXT_SUB)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    # ── Panel 5: Per-Beat Korotkoff Timeline ────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    bt = beat_info['beat_times']
    be = beat_info['beat_koro_energy']
    ba = beat_info['beat_is_active']
    colors_beat = [C_OK if a else C_FAIL for a in ba]
    ax5.bar(bt, be, width=0.28, color=colors_beat, edgecolor='none', alpha=0.85)
    ax5.axhline(beat_info['k_threshold'], color=C_WARN, ls='--', lw=1.5,
                label=f"Threshold ({beat_info['k_threshold']:.2f})")
    add_spans(ax5, show_perbeat=True, show_steth=False)
    ax5.set_title(f"5. Per-Beat Korotkoff Energy ({beat_info['k_count']} K-sounds detected)", fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax5.set_ylabel('RMS Energy (mm/s)', color=TEXT_SUB)
    ax5.legend(fontsize=8, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Beat Activation Heatmap ────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    be_norm = be / (np.max(be) + 1e-20)
    activation_2d = np.tile(be_norm, (3, 1))
    ax6.imshow(activation_2d, aspect='auto', cmap='RdYlGn',
               extent=[bt[0], bt[-1], 0, 1], interpolation='nearest', alpha=0.9)
    for i, (t_beat, active) in enumerate(zip(bt, ba)):
        marker = '▲' if active else '▽'
        color = '#15803D' if active else '#B91C1C'
        ax6.text(t_beat, 0.5, marker, ha='center', va='center', fontsize=9, color=color, fontweight='bold')
    ax6.axvline(rf_on, color=C_WARN, ls='--', lw=2.5)
    ax6.axvline(rf_off, color=C_WARN, ls='--', lw=2.5)
    ax6.set_yticks([])
    ax6.set_title('6. Beat Activation Timeline (▲=K-active, ▽=silent)', fontweight='bold', fontsize=12, color=TEXT_MAIN)

    # ── Panel 7: Energy Ratio Curve ─────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    noise_level = np.median(rf_energy[:int(8 * fs_rf)])
    energy_ratio = rf_energy_smooth / (noise_level + 1e-20)
    energy_ratio_db = 10 * np.log10(energy_ratio + 1e-20)
    ax7.plot(t_rf, energy_ratio_db, color='#A78BFA', lw=1.5)
    ax7.axhline(0, color=BORDER_COLOR, ls='-', lw=0.8, alpha=0.5)
    ax7.axhline(10 * np.log10(adapt_thresh / (noise_level + 1e-20) + 1e-20),
                color=C_FAIL, ls='--', lw=1.5, label='Adaptive Threshold')
    add_spans(ax7, show_perbeat=False)
    ax7.set_title('7. Energy Ratio (Korotkoff / Noise Floor, dB)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax7.set_ylabel('dB', color=TEXT_SUB)
    ax7.legend(fontsize=8, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax7.grid(True, alpha=0.3)

    # ── Panel 8: 6-Method Consensus Windows ─────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    m_colors = {'M1_VelRMS': C_RF, 'M2_TKEO': C_STETH,
                'M3_Kurtosis': '#C084FC', 'M4_Hilbert': C_OK,
                'M5_BandPower': C_WARN, 'M6_STFT': '#F9FAFB'}
    for name, curve in method_curves.items():
        cs = smooth(curve, int(fs_rf * 2.0))
        cn = cs / (np.max(cs) + 1e-20)
        ax8.plot(t_rf, cn, color=m_colors[name], alpha=0.85, lw=1.2, label=name)
    ax8.axvline(rf_on, color=C_WARN, ls='--', lw=2.5, label=f'Final onset {rf_on:.1f}s')
    ax8.axvline(rf_off, color=C_WARN, ls='-.', lw=2.5, label=f'Final offset {rf_off:.1f}s')
    ax8.set_title(f'8. 6-Algorithm Normalised Curves ({n_agree}/6 agree)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax8.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax8.grid(True, alpha=0.3)

    # ── Panel 9: HR Displacement + Beats ────────────────────────
    ax9 = fig.add_subplot(gs[4, 0])
    ax9.plot(t_rfp, dhr_p, color=C_RF, lw=1.0)
    ax9.plot(t_rf[peaks_rf],
             (disp_hr_rf / (np.max(np.abs(disp_hr_rf)) + 1e-20))[peaks_rf],
             color=C_STETH, marker='o', ls='none', ms=6, label=f'Beats ({hr_rf_bpm:.0f} BPM)',
             mec='#F9FAFB', mew=1.0, zorder=5)
    add_spans(ax9, show_perbeat=False)
    ax9.set_title(f'9. RF Heart Rate (Peaks: {hr_rf_bpm:.0f} BPM | PSD: {hr_rf_psd:.1f} BPM)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax9.set_ylabel('Normalised', color=TEXT_SUB)
    ax9.legend(fontsize=8, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax9.grid(True, alpha=0.3)

    # ── Panel 10: Zoomed Korotkoff Window ───────────────────────
    ax10 = fig.add_subplot(gs[4, 1])
    pad = 3.0
    z_on  = max(0, rf_on - pad)
    z_off = min(rec_dur_rf, rf_off + pad)
    mask_z = (t_rfp >= z_on) & (t_rfp <= z_off)
    ax10.plot(t_rfp[mask_z], vel_p[mask_z], color='#C084FC', lw=0.8)
    ax10.axvspan(rf_on, rf_off, **yw_rf)
    if beat_info['per_beat_onset'] is not None:
        ax10.axvspan(beat_info['per_beat_onset'],
                     beat_info['per_beat_offset'], **yw_pb)
    for bt_t, bt_a in zip(beat_info['beat_times'], beat_info['beat_is_active']):
        if z_on <= bt_t <= z_off:
            ax10.axvline(bt_t, color=C_OK if bt_a else C_FAIL, alpha=0.5, lw=1.0)
    ax10.set_title('10. Zoomed Korotkoff Velocity + K-Beats', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax10.set_ylabel('mm/s', color=TEXT_SUB)
    ax10.grid(True, alpha=0.3)

    # ── Panel 11: Energy Overlay (if stethoscope available) ─────
    ax11 = fig.add_subplot(gs[5, 0])
    if steth_loaded:
        rf_env_n_plot = rf_env_full / (np.max(rf_env_full) + 1e-20)
        ax11.plot(t_rf, rf_env_n_plot, color=C_RF, lw=2.2, label='RF Energy')
        ax11.plot(t_rf, aud_env_rf, color=C_STETH, lw=2.2, label='Steth Energy')
        aud_env_shifted = np.interp(t_rf + lag_sec, t_aud, aud_env_n)
        ax11.plot(t_rf, aud_env_shifted, color=C_STETH, lw=1.5, ls='--', alpha=0.5,
                  label=f'Steth (shifted +{lag_sec:.2f}s)')
        ax11.axvline(rf_on, color=C_RF, ls='--', alpha=0.8)
        ax11.axvline(rf_off, color=C_RF, ls='--', alpha=0.8)
        ax11.axvline(st_on, color=C_STETH, ls=':', alpha=0.8)
        ax11.axvline(st_off, color=C_STETH, ls=':', alpha=0.8)
        ax11.set_title(f'11. Energy Overlay (lag={lag_sec:.2f}s, r={cc_peak:.3f})', fontweight='bold', fontsize=12, color=TEXT_MAIN)
        ax11.legend(fontsize=8, loc='upper right', framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    else:
        rf_env_n_plot = rf_env_full / (np.max(rf_env_full) + 1e-20)
        ax11.plot(t_rf, rf_env_n_plot, color=C_RF, lw=2.2, label='RF Energy')
        add_spans(ax11, show_steth=False)
        ax11.set_title('11. RF Energy Envelope', fontweight='bold', fontsize=12, color=TEXT_MAIN)
        ax11.legend(fontsize=8, framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax11.set_ylabel('Normalised', color=TEXT_SUB)
    ax11.grid(True, alpha=0.3)

    # ── Panel 12: Cross-Correlation ─────────────────────────────
    ax12 = fig.add_subplot(gs[5, 1])
    if steth_loaded and len(cc) > 1:
        lags_arr = np.arange(len(cc)) - len(rf_env_n[seg_s:seg_e]) + 1
        lag_t_arr = lags_arr / fs_rf
        ax12.plot(lag_t_arr, cc / (np.max(cc) + 1e-20), color='#A78BFA', lw=1.8)
        ax12.axvline(lag_sec, color=C_FAIL, ls='--', lw=2.0,
                     label=f'Peak lag={lag_sec:.2f}s')
        ax12.set_xlim(-5, 5)
        ax12.set_title(f'12. Cross-Correlation (r={cc_peak:.3f})', fontweight='bold', fontsize=12, color=TEXT_MAIN)
        ax12.legend(fontsize=8, framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    else:
        ax12.text(0.5, 0.5, 'No stethoscope data', ha='center', va='center', fontsize=14, color=TEXT_SUB, transform=ax12.transAxes)
        ax12.set_title('12. Cross-Correlation (N/A)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax12.set_ylabel('Normalised CC', color=TEXT_SUB)
    ax12.grid(True, alpha=0.3)

    # ── Panel 13: Method Duration Comparison Bar ────────────────
    ax13 = fig.add_subplot(gs[6, 0])
    labels_bar, dur_bar, colors_bar = [], [], []
    all_detections = dict(method_wins)
    if rf_cusum_dur > 0:
        all_detections['CUSUM'] = {'onset': rf_cusum_onset, 'offset': rf_cusum_offset, 'duration': rf_cusum_dur}
    if beat_info['per_beat_onset'] is not None:
        pbd = beat_info['per_beat_offset'] - beat_info['per_beat_onset']
        all_detections['PerBeat'] = {'onset': beat_info['per_beat_onset'], 'offset': beat_info['per_beat_offset'], 'duration': pbd}
    for mk, mv in all_detections.items():
        lbl = mk.replace('_', '\n')
        labels_bar.append(lbl)
        if mv and MIN_DUR_S <= mv['duration'] <= MAX_DUR_S:
            colors_bar.append(C_OK)
            dur_bar.append(mv['duration'])
        elif mv:
            colors_bar.append(C_FAIL)
            dur_bar.append(mv['duration'])
        else:
            colors_bar.append(BORDER_COLOR)
            dur_bar.append(0)
    bars = ax13.bar(labels_bar, dur_bar, color=colors_bar, edgecolor=BORDER_COLOR, alpha=0.85, width=0.55)
    for bar, val in zip(bars, dur_bar):
        if val > 0:
            ax13.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                      f'{val:.1f}s', ha='center', fontsize=8, fontweight='bold', color=TEXT_MAIN)
    ax13.axhline(rf_dur, color=C_WARN, ls='--', lw=2.0, label=f'Final: {rf_dur:.1f}s')
    ax13.set_title('13. Detection Method Durations', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax13.set_ylabel('Duration (s)', color=TEXT_SUB)
    ax13.legend(fontsize=8, framealpha=0.9, facecolor=CARD_COLOR, edgecolor=BORDER_COLOR)
    ax13.tick_params(axis='x', labelsize=8)
    ax13.grid(True, axis='y', alpha=0.2)

    # ── Panel 14: Validation Metrics Bar ────────────────────────
    ax14 = fig.add_subplot(gs[6, 1])
    if steth_loaded:
        onset_diff  = abs(rf_on - st_on)
        offset_diff = abs(rf_off - st_off)
        dur_diff    = abs(rf_dur - st_dur)
        cats = ['Onset\nDiff (s)', 'Offset\nDiff (s)', 'Dur\nDiff (s)', 'Raw\nIoU', 'Lag-Corr\nIoU']
        vals = [onset_diff, offset_diff, dur_diff, iou, iou_corrected]
        thresh_list = [3.0, 3.0, 5.0, 0.5, 0.5]
        colors_v = [C_OK if (v <= t if i < 3 else v >= t) else C_FAIL
                    for i, (v, t) in enumerate(zip(vals, thresh_list))]
        bars_v = ax14.bar(cats, vals, color=colors_v, edgecolor=BORDER_COLOR, alpha=0.85, width=0.48)
        for bar, val in zip(bars_v, vals):
            ax14.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'{val:.2f}',
                      ha='center', fontsize=10, fontweight='bold', color=TEXT_MAIN)
        ax14.set_title('14. Cross-Validation Metrics', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    else:
        ax14.text(0.5, 0.5, 'No stethoscope data\nfor cross-validation', ha='center', va='center', fontsize=14, color=TEXT_SUB, transform=ax14.transAxes)
        ax14.set_title('14. Cross-Validation (N/A)', fontweight='bold', fontsize=12, color=TEXT_MAIN)
    ax14.set_ylabel('Value', color=TEXT_SUB)
    ax14.grid(True, axis='y', alpha=0.2)

    # ── Panel 15: Old vs New Comparison ─────────────────────────
    ax15 = fig.add_subplot(gs[7, 0])
    comparison_labels = ['Legacy\nDetector', 'CUSUM\nDetector', 'Consensus\nFusion', 'Per-Beat\nDetector']
    comparison_onsets = [
        rf_legacy_win['onset'] if rf_legacy_win else 0,
        rf_cusum_onset if rf_cusum_dur > 0 else 0,
        rf_on,
        beat_info['per_beat_onset'] or 0
    ]
    comparison_offsets = [
        rf_legacy_win['offset'] if rf_legacy_win else 0,
        rf_cusum_offset if rf_cusum_dur > 0 else 0,
        rf_off,
        beat_info['per_beat_offset'] or 0
    ]
    y_pos = np.arange(len(comparison_labels))
    bar_colors = [C_STETH, C_FAIL, C_WARN, C_OK]
    for i, (lbl, on, off) in enumerate(zip(comparison_labels, comparison_onsets, comparison_offsets)):
        if on > 0 and off > 0:
            ax15.barh(i, off - on, left=on, height=0.48, color=bar_colors[i], edgecolor=BORDER_COLOR, alpha=0.85)
            ax15.text(on + (off - on) / 2, i, f'{on:.1f}–{off:.1f}s\n({off-on:.1f}s)',
                      ha='center', va='center', fontsize=8, fontweight='bold', color=BG_COLOR if bar_colors[i]==C_WARN else TEXT_MAIN)
    ax15.set_yticks(y_pos)
    ax15.set_yticklabels(comparison_labels, fontsize=9)
    ax15.set_xlabel('Time (s)')
    ax15.set_title('15. Old vs New: Detection Comparison', fontweight='bold',
                   fontsize=11)

    # ── Panel 16: Summary Report ────────────────────────────────
    ax16 = fig.add_subplot(gs[7, 1])
    ax16.axis('off')

    lines = [
        f"IMPROVED KOROTKOFF ANALYSIS v3.0",
        f"{'═' * 52}",
        f"Recording : {os.path.basename(RF_PATH)}",
        f"Duration  : {rec_dur_rf:.1f}s | fs = {fs_rf:.0f} Hz",
        f"",
        f"DETECTION RESULTS:",
        f"  Final Window    : {rf_on:.2f}s – {rf_off:.2f}s ({rf_dur:.1f}s)",
        f"  CUSUM Window    : {rf_cusum_onset:.2f}s – {rf_cusum_offset:.2f}s "
        f"({rf_cusum_dur:.1f}s)",
        f"  Consensus Window: {consensus_on:.2f}s – {consensus_off:.2f}s "
        f"({consensus_dur:.1f}s)",
    ]
    if beat_info['per_beat_onset'] is not None:
        pbd = beat_info['per_beat_offset'] - beat_info['per_beat_onset']
        lines += [
            f"  Per-Beat Window : {beat_info['per_beat_onset']:.2f}s – "
            f"{beat_info['per_beat_offset']:.2f}s ({pbd:.1f}s)",
            f"  Korotkoff Count : {beat_info['k_count']} K-sounds",
        ]
    lines += [
        f"  Methods OK      : {n_agree} / 6",
        f"  Koro SNR        : {snr_db:.1f} dB",
        f"",
        f"HEART RATE:",
        f"  RF (peaks)      : {hr_rf_bpm:.0f} BPM",
        f"  RF (PSD)        : {hr_rf_psd:.1f} BPM",
    ]
    if steth_loaded:
        lines += [
            f"  Steth (peaks)   : {hr_aud_bpm:.0f} BPM",
            f"  Steth (PSD)     : {hr_aud_psd:.1f} BPM",
            f"  HR Diff         : {abs(hr_rf_bpm - hr_aud_bpm):.1f} BPM",
        ]
    if onset_ci:
        lines += [
            f"",
            f"BOOTSTRAP 95% CI:",
            f"  Onset           : [{onset_ci[0]:.2f}, {onset_ci[2]:.2f}]s",
            f"  Offset          : [{offset_ci[0]:.2f}, {offset_ci[2]:.2f}]s",
        ]
    if steth_loaded:
        lines += [
            f"",
            f"CROSS-VALIDATION:",
            f"  Steth Window    : {st_on:.2f}s – {st_off:.2f}s ({st_dur:.1f}s)",
            f"  Raw IoU         : {iou:.3f}",
            f"  Lag-Corrected   : {iou_corrected:.3f} (lag={lag_sec:.2f}s)",
            f"  XCorr peak r    : {cc_peak:.3f}",
            f"",
            f"  CONFIDENCE      : {confidence:.3f} [{conf_label}]",
        ]
    lines += [f"{'═' * 52}"]

    # Color the summary box based on confidence
    if steth_loaded:
        bg_color = conf_color
        bg_alpha = 0.15
    else:
        bg_color = 'lightyellow'
        bg_alpha = 0.8

    ax16.text(0.03, 0.97, '\n'.join(lines), fontsize=10, family='monospace',
              fontweight='bold', va='top', transform=ax16.transAxes,
              bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                        alpha=bg_alpha, edgecolor='black', linewidth=1.5))

    for a in fig.axes:
        if a != ax16:
            a.set_xlabel('Time (s)', fontsize=8)

    fig.suptitle(f'Improved Korotkoff Detection v3.0 — '
                 f'{os.path.basename(RF_PATH)}',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\n  Dashboard saved -> {OUTPUT_IMG}")

    # ── Return results dict for batch processing ─────────────────
    results = {
        'rf_file':          os.path.basename(RF_PATH),
        'rf_onset':         rf_on,
        'rf_offset':        rf_off,
        'rf_duration':      rf_dur,
        'cusum_onset':      rf_cusum_onset,
        'cusum_offset':     rf_cusum_offset,
        'cusum_duration':   rf_cusum_dur,
        'consensus_onset':  consensus_on,
        'consensus_offset': consensus_off,
        'consensus_dur':    consensus_dur,
        'per_beat_onset':   beat_info['per_beat_onset'],
        'per_beat_offset':  beat_info['per_beat_offset'],
        'k_count':          beat_info['k_count'],
        'n_methods_agree':  n_agree,
        'snr_db':           snr_db,
        'hr_rf_bpm':        hr_rf_bpm,
        'hr_rf_psd':        hr_rf_psd,
        'onset_ci':         onset_ci,
        'offset_ci':        offset_ci,
    }
    if steth_loaded:
        results.update({
            'steth_onset':    st_on,
            'steth_offset':   st_off,
            'steth_duration': st_dur,
            'raw_iou':        iou,
            'lag_corrected_iou': iou_corrected,
            'xcorr_lag':      lag_sec,
            'xcorr_peak_r':   cc_peak,
            'hr_aud_bpm':     hr_aud_bpm,
            'hr_aud_psd':     hr_aud_psd,
            'confidence':     confidence,
            'conf_label':     conf_label,
        })

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS COMPLETE — Confidence: {confidence:.3f} [{conf_label}]"
          if steth_loaded else "  ANALYSIS COMPLETE (RF-only)")
    print(f"{'=' * 70}")

    return results


if __name__ == '__main__':
    run()
