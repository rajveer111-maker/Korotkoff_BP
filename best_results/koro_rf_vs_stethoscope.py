"""
RF vs Stethoscope Korotkoff Cross-Validation
=============================================
Loads both RF (rec_koro_may15.h5) and stethoscope audio (korotoff_audio_stethoscope.mp4),
detects Korotkoff windows independently on both, then cross-validates.
"""
import h5py, numpy as np, os, pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, welch, stft, medfilt
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
# ── CONFIG ──────────────────────────────────────────────────────
RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope.mp4'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_rf_vs_stethoscope_new.png'

if len(sys.argv) > 1:
    RF_PATH = sys.argv[1]
if len(sys.argv) > 2:
    AUDIO_PATH = sys.argv[2]
if len(sys.argv) > 3:
    OUTPUT_IMG = sys.argv[3]
else:
    if len(sys.argv) > 1:
        base = os.path.basename(RF_PATH).replace('.h5', '')
        OUTPUT_IMG = os.path.join(os.path.dirname(RF_PATH), f'koro_rf_vs_stethoscope_{base}.png')

FS_RF      = 10_000
FC_HZ      = 0.9e9
IQ_MODE    = '-I+jQ'

MIN_ONSET_S = 10.0
MIN_TAIL_S  = 10.0

C = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

# ── HELPERS ─────────────────────────────────────────────────────
def smooth(x, w):
    k = max(1, w); return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True).mean().fillna(0).values)

def apply_iq(i, q):
    return -i + 1j * q  # IQ_MODE = '-I+jQ'

def iq_condition(iq):
    ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc = (qc - sp*ic) / (al*cp + 1e-15)
    return ic + 1j*qc

def robust_phase(iq):
    dphi = np.angle(iq[1:]*np.conj(iq[:-1]))
    h, b = np.histogram(dphi, 512)
    co = b[np.argmax(h)] + (b[1]-b[0])/2
    dc = dphi - co
    iqr = np.percentile(dc,75)-np.percentile(dc,25)
    dc = np.clip(dc, -max(3*iqr,0.017), max(3*iqr,0.017))
    return signal.detrend(np.insert(np.cumsum(dc),0,0.0))

def find_sustained(curve, time, fs, rec_dur, min_dur=5.0, max_dur=18.0):
    ss = int(MIN_ONSET_S*fs); se = int((rec_dur-MIN_TAIL_S)*fs)
    if se <= ss + int(min_dur*fs): return None
    sw = max(3, int(fs*0.5))|1
    cc = medfilt(curve, min(sw, len(curve) if len(curve)%2==1 else len(curve)-1))
    cc = smooth(cc, int(fs*1.0))
    best_score, best_on, best_off = -1, 0, 0
    for dt in np.arange(min_dur, min(max_dur, rec_dur-MIN_ONSET_S-MIN_TAIL_S)+0.5, 0.5):
        ws = int(dt*fs)
        dw = np.exp(-0.5*((dt-10.0)/3.0)**2)
        for s in range(ss, se-ws, int(fs*0.25)):
            e = s+ws
            if e > se: break
            sc = np.sum(cc[s:e])*dw
            if sc > best_score:
                best_score=sc; best_on=time[s]; best_off=time[min(e,len(time)-1)]
    d = best_off-best_on
    return {'onset':best_on,'offset':best_off,'duration':d} if d>2 else None

# ── LOAD RF ─────────────────────────────────────────────────────
def load_rf():
    with h5py.File(RF_PATH,'r') as f: data=f['data'][:]
    fs = FS_RF
    ir, qr = data[0,:], data[1,:]
    N = len(ir); t = np.arange(N)/fs
    iq = iq_condition(apply_iq(ir, qr))
    phase = robust_phase(iq)
    # Koro velocity
    sos_k = butter(4,[10,49],btype='band',fs=fs,output='sos')
    pk = sosfiltfilt(sos_k, phase)
    vk = np.append(np.diff(pk)*fs, 0)*SCALE
    # HR displacement
    sos_h = butter(4,[0.5,3.0],btype='band',fs=fs,output='sos')
    dh = sosfiltfilt(sos_h, phase)*SCALE
    return t, vk, dh, fs

# ── LOAD STETHOSCOPE ────────────────────────────────────────────
def load_stethoscope():
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(AUDIO_PATH)
        audio = clip.to_soundarray()
        fs_audio = clip.fps
        clip.close()
    except:
        fs_audio, audio = wavfile.read(AUDIO_PATH.replace('.mp4','.wav'))
        audio = audio.astype(np.float64) / 32768.0

    # Mix to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    N = len(audio); t = np.arange(N)/fs_audio
    print(f"  Stethoscope: {N} samples, {t[-1]:.1f}s, fs={fs_audio} Hz")

    # Filter to Korotkoff band 20-200 Hz (stethoscope range)
    sos_k = butter(4, [20, 200], btype='band', fs=fs_audio, output='sos')
    koro_audio = sosfiltfilt(sos_k, audio)

    # Also filter to heart sound band 0.5-5 Hz
    sos_h = butter(4, [0.5, 5.0], btype='band', fs=fs_audio, output='sos')
    hr_audio = sosfiltfilt(sos_h, audio)

    return t, audio, koro_audio, hr_audio, fs_audio

# ── MAIN ────────────────────────────────────────────────────────
def run():
    print("Loading RF...")
    t_rf, vel_koro_rf, disp_hr_rf, fs_rf = load_rf()
    rec_dur_rf = t_rf[-1]
    print(f"  RF: {len(t_rf)} samples, {rec_dur_rf:.1f}s")

    print("Loading Stethoscope...")
    t_aud, audio_raw, koro_aud, hr_aud, fs_aud = load_stethoscope()
    rec_dur_aud = t_aud[-1]

    # ── RF WINDOW DETECTION (velocity energy) ───────────────────
    print("Detecting RF Koro window...")
    rf_curve = sliding_rms(vel_koro_rf, int(fs_rf*0.5))**2
    rf_win = find_sustained(rf_curve, t_rf, fs_rf, rec_dur_rf)
    print(f"  RF window: {rf_win}")

    # ── STETHOSCOPE WINDOW DETECTION ────────────────────────────
    print("Detecting Stethoscope Koro window...")
    # Method A: Envelope energy of koro band
    aud_env = np.abs(hilbert(koro_aud))
    aud_curve_a = sliding_rms(aud_env, int(fs_aud*0.5))**2
    aud_win_a = find_sustained(aud_curve_a, t_aud, fs_aud, rec_dur_aud)
    print(f"  Steth M-A (Envelope): {aud_win_a}")

    # Method B: RMS energy of raw koro band
    aud_curve_b = sliding_rms(koro_aud, int(fs_aud*0.3))**2
    aud_curve_b = smooth(aud_curve_b, int(fs_aud*1.0))
    aud_win_b = find_sustained(aud_curve_b, t_aud, fs_aud, rec_dur_aud)
    print(f"  Steth M-B (RMS):      {aud_win_b}")

    # Method C: STFT sub-band energy
    nps = 4096
    f_s, t_s, Zs = stft(koro_aud, fs=fs_aud, nperseg=nps, noverlap=nps*3//4)
    Ps = np.abs(Zs)**2
    km = (f_s>=20)&(f_s<=200)
    se = np.mean(Ps[km,:], axis=0)
    aud_curve_c = np.interp(t_aud, t_s, se)
    aud_win_c = find_sustained(aud_curve_c, t_aud, fs_aud, rec_dur_aud)
    print(f"  Steth M-C (STFT):     {aud_win_c}")

    # Stethoscope consensus
    steth_wins = [w for w in [aud_win_a, aud_win_b, aud_win_c] if w is not None]
    if steth_wins:
        st_on  = float(np.median([w['onset'] for w in steth_wins]))
        st_off = float(np.median([w['offset'] for w in steth_wins]))
    else:
        st_on, st_off = 15.0, 25.0
    st_dur = st_off - st_on

    # RF consensus (from previous run)
    rf_on  = rf_win['onset']  if rf_win else 10.0
    rf_off = rf_win['offset'] if rf_win else 20.0
    rf_dur = rf_off - rf_on

    # ── CROSS-VALIDATION ────────────────────────────────────────
    onset_diff  = abs(rf_on - st_on)
    offset_diff = abs(rf_off - st_off)
    dur_diff    = abs(rf_dur - st_dur)
    overlap_start = max(rf_on, st_on)
    overlap_end   = min(rf_off, st_off)
    overlap_dur   = max(0, overlap_end - overlap_start)
    union_dur     = max(rf_off, st_off) - min(rf_on, st_on)
    iou = overlap_dur / union_dur if union_dur > 0 else 0

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"  RF Window       : {rf_on:.2f}s - {rf_off:.2f}s ({rf_dur:.1f}s)")
    print(f"  Stethoscope     : {st_on:.2f}s - {st_off:.2f}s ({st_dur:.1f}s)")
    print(f"  Onset diff      : {onset_diff:.2f}s")
    print(f"  Offset diff     : {offset_diff:.2f}s")
    print(f"  Duration diff   : {dur_diff:.1f}s")
    print(f"  Overlap (IoU)   : {iou:.2f} ({overlap_dur:.1f}s)")
    match = onset_diff < 3.0 and offset_diff < 3.0 and iou > 0.5
    print(f"  MATCH           : {'YES [PASS]' if match else 'NO [FAIL]'}")
    print(f"{'='*60}")

    # ── STFT for stethoscope ────────────────────────────────────
    nps2 = 2048
    f_s2, t_s2, Zs2 = stft(koro_aud, fs=fs_aud, nperseg=nps2, noverlap=nps2*3//4)
    Ps2_db = 10*np.log10(np.abs(Zs2)**2 + 1e-20)

    # ── STFT for RF ─────────────────────────────────────────────
    nps3 = 2048
    f_rf, t_rf2, Zrf = stft(vel_koro_rf, fs=fs_rf, nperseg=nps3, noverlap=nps3*3//4)
    Prf_db = 10*np.log10(np.abs(Zrf)**2 + 1e-20)

    # ── HR detection on stethoscope ─────────────────────────────
    peaks_aud, _ = signal.find_peaks(np.abs(hr_aud), distance=int(fs_aud*0.4),
                                      prominence=np.std(hr_aud)*0.5)
    if len(peaks_aud)>2:
        iv = np.diff(t_aud[peaks_aud]); viv = iv[(iv>0.3)&(iv<2.0)]
        hr_aud_bpm = 60.0/np.median(viv) if len(viv)>0 else 0
    else: hr_aud_bpm = 0

    # HR on RF
    t_stable = disp_hr_rf[int(10*fs_rf):int(20*fs_rf)]
    pth = np.std(t_stable)*0.8
    peaks_rf, _ = signal.find_peaks(-disp_hr_rf, distance=int(fs_rf*0.5), prominence=pth)
    if len(peaks_rf)>1:
        iv = np.diff(t_rf[peaks_rf]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_rf_bpm = 60.0/np.median(viv) if len(viv)>0 else 0
    else: hr_rf_bpm = 0

    # ── PSD HEART RATE VERIFICATION ──────────────────────────────
    print("Calculating PSD for RF and Stethoscope Heart Rate...")
    # For RF: Detrend first to remove low-frequency drift, then Welch PSD
    disp_hr_rf_detrend = signal.detrend(disp_hr_rf)
    f_rf_psd, p_rf_psd = welch(disp_hr_rf_detrend, fs=fs_rf, nperseg=min(len(disp_hr_rf_detrend), int(fs_rf * 20)))
    mask_rf_psd = (f_rf_psd >= 0.5) & (f_rf_psd <= 3.0)
    if np.any(mask_rf_psd):
        hr_rf_psd_hz = f_rf_psd[mask_rf_psd][np.argmax(p_rf_psd[mask_rf_psd])]
        hr_rf_psd_bpm = hr_rf_psd_hz * 60.0
    else:
        hr_rf_psd_bpm = 0.0

    # For Stethoscope: Detrend first, then Welch PSD
    hr_aud_detrend = signal.detrend(hr_aud)
    f_aud_psd, p_aud_psd = welch(hr_aud_detrend, fs=fs_aud, nperseg=min(len(hr_aud_detrend), int(fs_aud * 20)))
    mask_aud_psd = (f_aud_psd >= 0.5) & (f_aud_psd <= 3.0)
    if np.any(mask_aud_psd):
        hr_aud_psd_hz = f_aud_psd[mask_aud_psd][np.argmax(p_aud_psd[mask_aud_psd])]
        hr_aud_psd_bpm = hr_aud_psd_hz * 60.0
    else:
        hr_aud_psd_bpm = 0.0

    print(f"  RF PSD Heart Rate: {hr_rf_psd_bpm:.1f} BPM")
    print(f"  Steth PSD Heart Rate: {hr_aud_psd_bpm:.1f} BPM")

    # ── NORMALISED ENERGY CURVES for overlay ────────────────────
    # RF energy at 1-second resolution
    rf_env = smooth(rf_curve, int(fs_rf*2.0))
    rf_env_n = rf_env / (np.max(rf_env)+1e-20)

    # Steth energy at 1-second resolution, resampled to RF time
    aud_env_sm = smooth(aud_curve_a, int(fs_aud*2.0))
    aud_env_n = aud_env_sm / (np.max(aud_env_sm)+1e-20)
    aud_env_rf = np.interp(t_rf, t_aud, aud_env_n)

    # Cross-correlation to check time alignment
    cc = np.correlate(rf_env_n[int(5*fs_rf):int(50*fs_rf)],
                      aud_env_rf[int(5*fs_rf):int(50*fs_rf)], mode='full')
    lag_samples = np.argmax(cc) - len(rf_env_n[int(5*fs_rf):int(50*fs_rf)]) + 1
    lag_sec = lag_samples / fs_rf
    cc_peak = np.max(cc) / (np.sqrt(np.sum(rf_env_n[int(5*fs_rf):int(50*fs_rf)]**2) *
                                     np.sum(aud_env_rf[int(5*fs_rf):int(50*fs_rf)]**2)) + 1e-20)
    print(f"  Cross-corr lag  : {lag_sec:.2f}s, peak r={cc_peak:.3f}")

    # ── PLOTTING (12 panels) ────────────────────────────────────
    fig, axes = plt.subplots(6, 2, figsize=(26, 38))
    plt.subplots_adjust(hspace=0.50, wspace=0.25)
    yw_rf = dict(color='gold', alpha=0.25)
    yw_st = dict(color='cyan', alpha=0.15)

    def spans(ax):
        ax.axvspan(rf_on, rf_off, **yw_rf, label=f'RF {rf_on:.1f}-{rf_off:.1f}s')
        ax.axvspan(st_on, st_off, **yw_st, label=f'Steth {st_on:.1f}-{st_off:.1f}s')

    # Row 1: Raw signals
    ax = axes[0,0]
    ax.plot(t_rf, vel_koro_rf, 'gray', lw=0.4); spans(ax)
    ax.set_title('1. RF Korotkoff Velocity (10-49 Hz)', fontweight='bold'); ax.set_ylabel('mm/s')
    ax.legend(fontsize=7)

    ax = axes[0,1]
    ax.plot(t_aud, koro_aud, 'steelblue', lw=0.3); spans(ax)
    ax.set_title('2. Stethoscope Korotkoff (20-200 Hz)', fontweight='bold'); ax.set_ylabel('Amplitude')
    ax.legend(fontsize=7)

    # Row 2: STFT spectrograms
    ax = axes[1,0]
    fm_rf = (f_rf>=5)&(f_rf<=60)
    va,vb = np.percentile(Prf_db[fm_rf],[20,99])
    im1 = ax.pcolormesh(t_rf2, f_rf[fm_rf], Prf_db[fm_rf], shading='gouraud', cmap='magma', vmin=va, vmax=vb)
    ax.axvline(rf_on, color='lime', ls='--', lw=2); ax.axvline(rf_off, color='lime', ls='--', lw=2)
    ax.axvline(st_on, color='cyan', ls=':', lw=2); ax.axvline(st_off, color='cyan', ls=':', lw=2)
    ax.set_title('3. RF STFT (5-60 Hz)', fontweight='bold'); ax.set_ylabel('Hz')
    plt.colorbar(im1, ax=ax, label='dB')

    ax = axes[1,1]
    fm_aud = (f_s2>=10)&(f_s2<=300)
    va2,vb2 = np.percentile(Ps2_db[fm_aud],[20,99])
    im2 = ax.pcolormesh(t_s2, f_s2[fm_aud], Ps2_db[fm_aud], shading='gouraud', cmap='inferno', vmin=va2, vmax=vb2)
    ax.axvline(rf_on, color='lime', ls='--', lw=2); ax.axvline(rf_off, color='lime', ls='--', lw=2)
    ax.axvline(st_on, color='cyan', ls=':', lw=2); ax.axvline(st_off, color='cyan', ls=':', lw=2)
    ax.set_title('4. Stethoscope STFT (10-300 Hz)', fontweight='bold'); ax.set_ylabel('Hz')
    plt.colorbar(im2, ax=ax, label='dB')

    # Row 3: Energy envelopes overlay
    ax = axes[2,0]
    ax.plot(t_rf, rf_env_n, 'red', lw=2, label='RF Energy')
    ax.plot(t_rf, aud_env_rf, 'blue', lw=2, label='Stethoscope Energy')
    ax.axvline(rf_on, color='red', ls='--', alpha=0.7); ax.axvline(rf_off, color='red', ls='--', alpha=0.7)
    ax.axvline(st_on, color='blue', ls=':', alpha=0.7); ax.axvline(st_off, color='blue', ls=':', alpha=0.7)
    ax.set_title('5. Energy Envelope Overlay (Normalized)', fontweight='bold')
    ax.set_ylabel('Normalized Amplitude'); ax.legend(fontsize=8)

    ax = axes[2,1]
    lags = np.arange(len(cc)) - len(rf_env_n[int(5*fs_rf):int(50*fs_rf)]) + 1
    lag_t = lags / fs_rf
    ax.plot(lag_t, cc / (np.max(cc)+1e-20), 'purple', lw=1)
    ax.axvline(lag_sec, color='red', ls='--', label=f'Peak lag={lag_sec:.2f}s')
    ax.set_xlim(-5, 5); ax.set_title(f'6. Cross-Correlation (r={cc_peak:.3f})', fontweight='bold')
    ax.set_ylabel('Normalized CC'); ax.legend(fontsize=8)

    # Row 4: Zoomed koro windows
    pad = 3.0
    z_on  = max(0, min(rf_on, st_on) - pad)
    z_off = min(max(rec_dur_rf, rec_dur_aud), max(rf_off, st_off) + pad)

    ax = axes[3,0]
    mask_rf = (t_rf>=z_on)&(t_rf<=z_off)
    ax.plot(t_rf[mask_rf], vel_koro_rf[mask_rf], 'firebrick', lw=0.6); spans(ax)
    ax.set_title('7. RF Zoomed', fontweight='bold'); ax.set_ylabel('mm/s'); ax.legend(fontsize=7)

    ax = axes[3,1]
    mask_aud = (t_aud>=z_on)&(t_aud<=z_off)
    ax.plot(t_aud[mask_aud], koro_aud[mask_aud], 'steelblue', lw=0.4); spans(ax)
    ax.set_title('8. Stethoscope Zoomed', fontweight='bold'); ax.set_ylabel('Amplitude'); ax.legend(fontsize=7)

    # Row 5: HR comparison
    ax = axes[4,0]
    disp_hr_rf_n = disp_hr_rf / (np.max(np.abs(disp_hr_rf)) + 1e-20)
    ax.plot(t_rf, disp_hr_rf_n, 'firebrick', lw=0.8)
    ax.plot(t_rf[peaks_rf], disp_hr_rf_n[peaks_rf], 'bo', ms=4, label=f'RF Beats ({hr_rf_bpm:.0f} BPM)')
    spans(ax)
    ax.set_title(f'9. RF Heart Rate (Peaks: {hr_rf_bpm:.0f} BPM | PSD: {hr_rf_psd_bpm:.1f} BPM)', fontweight='bold')
    ax.set_ylabel('Normalized Amplitude'); ax.legend(fontsize=7)

    ax = axes[4,1]
    hr_aud_n = hr_aud / (np.max(np.abs(hr_aud)) + 1e-20)
    ax.plot(t_aud, hr_aud_n, 'steelblue', lw=0.5)
    ds = max(1, len(peaks_aud)//200)
    ax.plot(t_aud[peaks_aud[::ds]], hr_aud_n[peaks_aud[::ds]], 'ro', ms=3, label=f'Steth Beats ({hr_aud_bpm:.0f} BPM)')
    spans(ax)
    ax.set_title(f'10. Stethoscope Heart Rate (Peaks: {hr_aud_bpm:.0f} BPM | PSD: {hr_aud_psd_bpm:.1f} BPM)', fontweight='bold')
    ax.set_ylabel('Normalized Amplitude'); ax.legend(fontsize=7)

    # Row 6: Validation summary
    ax = axes[5,0]
    categories = ['Onset\nDiff (s)', 'Offset\nDiff (s)', 'Duration\nDiff (s)', 'Overlap\nIoU']
    values = [onset_diff, offset_diff, dur_diff, iou]
    thresholds = [3.0, 3.0, 5.0, 0.5]
    colors = ['limegreen' if (v <= t if i < 3 else v >= t) else 'salmon'
              for i, (v, t) in enumerate(zip(values, thresholds))]
    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.2f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_title('11. Cross-Validation Metrics', fontweight='bold')
    ax.set_ylabel('Value')

    ax = axes[5,1]; ax.axis('off')
    summary = [
        "RF vs STETHOSCOPE CROSS-VALIDATION",
        "="*50,
        f"RF Recording   : {os.path.basename(RF_PATH)}",
        f"Stethoscope    : {os.path.basename(AUDIO_PATH)}",
        "",
        "DETECTED WINDOWS:",
        f"  RF           : {rf_on:.2f}s - {rf_off:.2f}s ({rf_dur:.1f}s)",
        f"  Stethoscope  : {st_on:.2f}s - {st_off:.2f}s ({st_dur:.1f}s)",
        "",
        "CROSS-VALIDATION:",
        f"  Onset diff   : {onset_diff:.2f} s {'[OK]' if onset_diff<3 else '[X]'}",
        f"  Offset diff  : {offset_diff:.2f} s {'[OK]' if offset_diff<3 else '[X]'}",
        f"  Duration diff: {dur_diff:.1f} s {'[OK]' if dur_diff<5 else '[X]'}",
        f"  Overlap IoU  : {iou:.2f} {'[OK]' if iou>0.5 else '[X]'}",
        f"  XCorr lag    : {lag_sec:.2f} s",
        f"  XCorr peak   : {cc_peak:.3f}",
        "",
        "HEART RATE (PEAK DETECTION):",
        f"  RF           : {hr_rf_bpm:.0f} BPM",
        f"  Stethoscope  : {hr_aud_bpm:.0f} BPM",
        f"  Diff         : {abs(hr_rf_bpm-hr_aud_bpm):.0f} BPM",
        "",
        "HEART RATE (PSD SPECTRAL):",
        f"  RF           : {hr_rf_psd_bpm:.1f} BPM",
        f"  Stethoscope  : {hr_aud_psd_bpm:.1f} BPM",
        f"  Diff         : {abs(hr_rf_psd_bpm-hr_aud_psd_bpm):.1f} BPM",
        "="*50,
        f"  STATUS: {'VALIDATED [OK]' if match else 'CHECK MANUALLY [X]'}",
    ]
    ax.text(0.05, 0.95, '\n'.join(summary), fontsize=11, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    for a in axes.flat: a.set_xlabel('Time (s)')
    fig.suptitle('RF vs Stethoscope Korotkoff Cross-Validation', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\nDashboard saved -> {OUTPUT_IMG}")

if __name__ == '__main__':
    run()
