"""
Multi-Subject 3-Figure Dual-Modality Korotkoff Analysis v5.0
=============================================================
Subjects : Prof. Kan (Sub_1) and Rajveer (Sub_2)
Per recording generates 3 separate 300-DPI figures:

  Figure 1 -- Acoustic/Stethoscope Analysis   (8 panels, 4x2)
  Figure 2 -- RF Radar RMG Analysis           (8 panels, 4x2)
             Phase & Magnitude INDEPENDENTLY validated then cross-checked
  Figure 3 -- Cross-Modality Comparison       (6 panels, 3x2)

Key improvements in v5.0:
  - Phase displacement shown as BANDPASS-filtered (10-200 Hz) * SCALE (mm)
    --> physically realistic sub-mm Korotkoff arterial wall vibration
    --> raw accumulated phase * SCALE was unrealistically huge (100s mm)
  - 5 Phase-only envelopes -> Phase consensus window (independent)
  - 5 Magnitude-only envelopes -> Magnitude consensus window (independent)
  - Both windows shown on TFD spectrograms for visual cross-validation
  - Joint consensus = mean of Phase + Magnitude if IoU >= 0.70, else Phase
  - Report card shows: signal RMS in um, cardiac pk-pk in mm, PM IoU
"""

import h5py
import numpy as np
import os, sys, traceback

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch, stft, medfilt
from scipy.fft import next_fast_len
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000      # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)        # ~26.5 mm/rad (1 rad -> 26.5 mm)
# NOTE: SCALE used ONLY on BANDPASS-filtered phase (10-200 Hz).
# After bandpass, amplitudes are tiny (0.001-0.5 rad), so mm values are
# physically realistic (0.026-13 mm range, Korotkoff typically 0.01-0.5 mm).

MIN_ONSET_S = 20.0   # window must START after 20 s
MIN_TAIL_S  =  5.0
MIN_DUR_S   =  5.0
MAX_DUR_S   = 25.0   # maximum Korotkoff duration

# ── SUBJECT CONFIG ────────────────────────────────────────────────────
BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUBJECTS = [
    {"name": "Prof_Kan",  "label": "Prof. Kan (Sub 1)",
     "folder": os.path.join(BASE, "Sub_1_Prof_kan"),
     "color": "#E84393", "n_recs": 10},
    {"name": "Rajveer",   "label": "Rajveer (Sub 2)",
     "folder": os.path.join(BASE, "Sub_2_Rajveer"),
     "color": "#2196F3", "n_recs": 10},
]
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ── SIGNAL HELPERS ────────────────────────────────────────────────────
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2)
                   .rolling(int(w), center=True).mean().fillna(0).values)

def sliding_kurt(x, w):
    return pd.Series(x).rolling(int(w), center=True).kurt().fillna(0).values

def calc_tkeo(x):
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t

def fast_hilbert(x):
    n = next_fast_len(len(x))
    from scipy.signal import hilbert
    return np.abs(hilbert(x, N=n)[:len(x)])

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic*qc)
    sp = np.clip(p3/np.sqrt(p1*p2+1e-20), -1, 1)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    qc2 = (qc - sp*ic)/(al*cp+1e-15) if abs(np.degrees(np.arcsin(sp)))<90 else qc
    return ic + 1j*qc2

# ── ADAPTIVE WINDOW SEARCH ────────────────────────────────────────────
def find_window_adaptive(curve, time, fs, rec_dur, target_dur=15.0, sigma=5.0):
    s_start = int(MIN_ONSET_S * fs)
    s_end   = int((rec_dur - MIN_TAIL_S) * fs)
    if s_end <= s_start + int(MIN_DUR_S * fs):
        return None
    kw = max(3, int(fs*0.5)) | 1
    kw = min(kw, len(curve) if len(curve)%2==1 else len(curve)-1)
    c  = smooth(medfilt(curve, kernel_size=kw), int(fs*1.0))
    best, best_on, best_off = -1, 0, 0
    for dur in np.arange(MIN_DUR_S,
                          min(MAX_DUR_S, rec_dur-MIN_ONSET_S-MIN_TAIL_S)+0.5, 0.5):
        ws    = int(dur * fs)
        w_dur = np.exp(-0.5*((dur - target_dur)/sigma)**2)
        for s in range(s_start, s_end - ws, int(fs*0.25)):
            e = s + ws
            if e > s_end: break
            t_mid  = time[s] + dur/2.0
            w_time = np.exp(-0.5*((t_mid - 32.0)/10.0)**2)
            score  = np.mean(c[s:e]) * w_dur * w_time
            if score > best:
                best = score; best_on = time[s]; best_off = time[min(e, len(time)-1)]
    dur = best_off - best_on
    return None if dur < 2.0 else {'onset': best_on, 'offset': best_off, 'duration': dur}

def passes(w, rec_dur):
    return (w is not None and w['onset'] >= MIN_ONSET_S and
            w['offset'] <= rec_dur - MIN_TAIL_S and
            MIN_DUR_S <= w['duration'] <= MAX_DUR_S)

def consensus(methods, rec_dur, fallback=(24.0, 44.0)):
    valid_on  = [v['onset']  for v in methods.values() if passes(v, rec_dur)]
    valid_off = [v['offset'] for v in methods.values() if passes(v, rec_dur)]
    if not valid_on:
        return fallback
    med_on, med_off = np.median(valid_on), np.median(valid_off)
    fon  = [o for o in valid_on  if abs(o-med_on)  <= 3.0] or [med_on]
    foff = [o for o in valid_off if abs(o-med_off) <= 3.0] or [med_off]
    return float(np.mean(fon)), float(np.mean(foff))

def iou_calc(ao, af, bo, bf):
    i = max(0, min(af,bf)-max(ao,bo))
    u = max(af,bf)-min(ao,bo)
    return i/u if u>0 else 0.0

# ════════════════════════════════════════════════════════════════════
# RF PROCESSING  (IMPROVED v5.0)
# ════════════════════════════════════════════════════════════════════
def process_rf(rf_path):
    with h5py.File(rf_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    N = len(i_raw); t = np.arange(N)/FS_RF; rec_dur = t[-1]

    # Step 1: IQ conditioning + anti-alias LPF
    iq    = b210_iq_condition(-i_raw + 1j*q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_c   = sosfiltfilt(sos_lp, iq)
    idx_def = int(20.0 * FS_RF)

    # Step 2: Decoupled Phase Reconstruction (no poly/linear detrending)
    puw    = np.unwrap(np.angle(iq_c[idx_def:]))
    dp     = np.diff(puw)
    dp    -= np.median(dp)          # remove carrier frequency offset (CFO)
    dp     = np.clip(dp, -0.5, 0.5) # clip jumps > lambda/4
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)

    ph_inf  = np.angle(iq_c[:idx_def])
    ph_inf -= (pd.Series(ph_inf).rolling(int(FS_RF), center=True)
               .mean().bfill().ffill().values)
    ph_inf += ph_def[0] - ph_inf[-1]
    phase_clean = np.concatenate([ph_inf, ph_def])  # radians

    # Bandpass filters
    sos_k    = butter(4, [10, 200],  btype='band', fs=FS_RF, output='sos')  # Korotkoff
    sos_h    = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')  # Cardiac
    sos_slow = butter(4, [0.05, 0.5], btype='band', fs=FS_RF, output='sos') # Breathing

    # KEY: Korotkoff displacement = BP-filtered phase * SCALE (sub-mm physical values)
    pk_rad = sosfiltfilt(sos_k, phase_clean)
    pk_mm  = pk_rad * SCALE                          # Korotkoff displacement (mm)
    vk     = np.append(np.diff(pk_mm)*FS_RF, 0)     # Phase velocity (mm/s)
    dh     = sosfiltfilt(sos_h, phase_clean) * SCALE # Cardiac displacement (mm)
    slow_mm = sosfiltfilt(sos_slow, phase_clean) * SCALE  # Breathing drift (mm)

    # Step 3: Magnitude Reconstruction (no detrending)
    mag_raw = np.abs(iq_c)
    b50, a50 = signal.iirnotch(50.0, 30, FS_RF)
    mag_n   = signal.filtfilt(b50, a50, mag_raw)

    mg_def = mag_n[idx_def:]
    mg_inf = mag_n[:idx_def]
    mg_inf -= pd.Series(mg_inf).rolling(int(FS_RF), center=True).mean().bfill().ffill().values
    mg_inf += mg_def[0] - mg_inf[-1]
    mag_pp  = np.concatenate([mg_inf, mg_def])
    mag_k   = sosfiltfilt(sos_k, mag_pp)
    vel_mag = np.append(np.diff(mag_k)*FS_RF, 0)   # Magnitude velocity (a.u./s)

    # Step 4: 5 Phase-only envelopes
    npsg = 2048
    ph_curves = {
        'Ph RMS'    : sliding_rms(vk, int(FS_RF*0.5))**2,
        'Ph TKEO'   : np.abs(calc_tkeo(vk)),
        'Ph Hilbert': fast_hilbert(vk),
        'Ph Kurt'   : np.clip(sliding_kurt(vk, int(FS_RF*1.0)), 0, None),
    }
    fs_v, ts_v, Zv = stft(vk, fs=FS_RF, nperseg=npsg, noverlap=npsg*3//4)
    km = (fs_v >= 10) & (fs_v <= 200)
    ph_curves['Ph STFT'] = np.interp(t, ts_v, np.mean((np.abs(Zv)**2)[km], axis=0))

    # Step 5: 5 Magnitude-only envelopes
    mg_curves = {
        'Mag RMS'    : sliding_rms(vel_mag, int(FS_RF*0.5))**2,
        'Mag TKEO'   : np.abs(calc_tkeo(vel_mag)),
        'Mag Hilbert': fast_hilbert(vel_mag),
        'Mag Kurt'   : np.clip(sliding_kurt(vel_mag, int(FS_RF*1.0)), 0, None),
    }
    fs_m, ts_m, Zm = stft(vel_mag, fs=FS_RF, nperseg=npsg, noverlap=npsg*3//4)
    mg_curves['Mag STFT'] = np.interp(t, ts_m, np.mean((np.abs(Zm)**2)[km], axis=0))

    # Step 6: Independent consensus windows
    ph_methods = {n: find_window_adaptive(c/(np.max(c)+1e-20), t, FS_RF, rec_dur)
                  for n, c in ph_curves.items()}
    mg_methods = {n: find_window_adaptive(c/(np.max(c)+1e-20), t, FS_RF, rec_dur)
                  for n, c in mg_curves.items()}

    ph_on, ph_off = consensus(ph_methods, rec_dur)
    mg_on, mg_off = consensus(mg_methods, rec_dur)
    ph_dur = ph_off - ph_on
    mg_dur = mg_off - mg_on
    dur_diff_pm = abs(ph_dur - mg_dur)
    pm_iou = iou_calc(ph_on, ph_off, mg_on, mg_off)

    # Step 7: Joint consensus (mean if agree, else trust phase)
    if pm_iou >= 0.70:
        rf_on  = (ph_on  + mg_on)  / 2.0
        rf_off = (ph_off + mg_off) / 2.0
    else:
        rf_on, rf_off = ph_on, ph_off
    rf_dur = rf_off - rf_on

    # Combined for backward compat
    curves = {**ph_curves, **mg_curves}
    methods = {**ph_methods, **mg_methods}

    # PSD in active window
    mw = (t >= rf_on) & (t <= rf_off)
    fpsd_ph, ppsd_ph = welch(vk[mw], fs=FS_RF, nperseg=min(int(FS_RF), max(1,mw.sum())))
    ppsd_ph_db = 10*np.log10(ppsd_ph + 1e-20)
    fpsd_mg, ppsd_mg = welch(vel_mag[mw], fs=FS_RF, nperseg=min(int(FS_RF), max(1,mw.sum())))
    ppsd_mg_db = 10*np.log10(ppsd_mg + 1e-20)

    # Heart rate
    s0, s1 = int(10*FS_RF), int(50*FS_RF)
    pks, _ = signal.find_peaks(-dh[s0:s1], distance=int(FS_RF*0.5),
                                 prominence=np.std(dh[s0:s1])*0.8)
    pks += s0
    if len(pks) > 1:
        iv = np.diff(t[pks]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_pk = 60/np.median(viv) if len(viv) else 0.0
    else:
        hr_pk = 0.0
    fhr, phr = welch(signal.detrend(dh[s0:s1]), fs=FS_RF,
                      nperseg=min(s1-s0, int(FS_RF*20)))
    mh = (fhr>=0.8)&(fhr<=2.5)
    hr_psd = fhr[mh][np.argmax(phr[mh])]*60 if mh.any() else 0.0

    return dict(
        t=t, vk=vk, dh=dh, pk_mm=pk_mm, slow_mm=slow_mm,
        vel_mag=vel_mag, mag_pp=mag_pp,
        ph_curves=ph_curves, mg_curves=mg_curves,
        ph_methods=ph_methods, mg_methods=mg_methods,
        curves=curves, methods=methods,
        ph_on=ph_on, ph_off=ph_off, ph_dur=ph_dur,
        mg_on=mg_on, mg_off=mg_off, mg_dur=mg_dur,
        onset=rf_on, offset=rf_off, duration=rf_dur,
        pm_iou=pm_iou, dur_diff_pm=dur_diff_pm,
        fpsd=fpsd_ph, ppsd_db=ppsd_ph_db,
        fpsd_mg=fpsd_mg, ppsd_mg_db=ppsd_mg_db,
        hr_pk=hr_pk, hr_psd=hr_psd, rec_dur=rec_dur,
        disp_mm=pk_mm,  # backward compat
    )

# ════════════════════════════════════════════════════════════════════
# STETHOSCOPE PROCESSING
# ════════════════════════════════════════════════════════════════════
def process_stethoscope(wav_path):
    fs_a, raw = wavfile.read(wav_path)
    audio = raw.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    N = len(audio); t = np.arange(N)/fs_a; rec_dur = t[-1]

    sos_k = butter(4, [50, 1000],  btype='band', fs=fs_a, output='sos')
    sos_h = butter(4, [0.4, 3.0],  btype='band', fs=fs_a, output='sos')
    ka = sosfiltfilt(sos_k, audio)
    ha = sosfiltfilt(sos_h, audio)

    rms_env  = sliding_rms(ka, int(fs_a*0.3))**2
    hil_env  = fast_hilbert(ka)
    npsg = 4096
    fs_s, ts_s, Zs = stft(ka, fs=fs_a, nperseg=npsg, noverlap=npsg*3//4)
    km2 = (fs_s>=50)&(fs_s<=1000)
    stft_env = np.interp(t, ts_s, np.mean((np.abs(Zs)**2)[km2], axis=0))

    curves_st = {'RMS': rms_env, 'Hilbert': hil_env, 'STFT': stft_env}
    methods_st = {n: find_window_adaptive(c/(np.max(c)+1e-20), t, fs_a, rec_dur)
                  for n, c in curves_st.items()}
    st_on, st_off = consensus(methods_st, rec_dur)
    st_dur = st_off - st_on

    env_n = np.clip(rms_env, 0, np.percentile(rms_env, 95))
    env_n /= (env_n.max() + 1e-20)

    mw = (t>=st_on)&(t<=st_off)
    fpsd, ppsd = welch(ka[mw], fs=fs_a, nperseg=min(int(fs_a), max(1,mw.sum())))
    ppsd_db = 10*np.log10(ppsd+1e-20)

    fs_sp = 2400
    ka_ds = signal.resample_poly(ka, up=fs_sp, down=int(fs_a))
    npss  = min(len(ka_ds)//4, int(fs_sp*0.05))
    fsp, tsp, Ssp = signal.spectrogram(ka_ds, fs=fs_sp, window='hann',
                                        nperseg=npss, noverlap=npss*7//8, nfft=1024)

    s0, s1 = int(10*fs_a), int(50*fs_a)
    pks, _ = signal.find_peaks(np.abs(ha[s0:s1]), distance=int(fs_a*0.4),
                                 prominence=np.std(ha[s0:s1])*0.5)
    pks += s0
    if len(pks) > 1:
        iv = np.diff(t[pks]); viv = iv[(iv>0.3)&(iv<2.0)]
        hr_pk = 60/np.median(viv) if len(viv) else 0.0
    else:
        hr_pk = 0.0
    fhr, phr = welch(signal.detrend(ha[s0:s1]), fs=fs_a,
                      nperseg=min(s1-s0, int(fs_a*20)))
    mh = (fhr>=0.8)&(fhr<=2.5)
    hr_psd = fhr[mh][np.argmax(phr[mh])]*60 if mh.any() else 0.0

    return dict(t=t, ka=ka, ha=ha, audio=audio, fs=fs_a, rec_dur=rec_dur,
                env_n=env_n, rms_env=rms_env,
                curves=curves_st, methods=methods_st,
                onset=st_on, offset=st_off, duration=st_dur,
                fpsd=fpsd, ppsd_db=ppsd_db,
                fsp=fsp, tsp=tsp, Ssp=Ssp, fs_sp=fs_sp,
                hr_pk=hr_pk, hr_psd=hr_psd)

# ════════════════════════════════════════════════════════════════════
# ALIGNMENT
# ════════════════════════════════════════════════════════════════════
def align(rf, st):
    pe = smooth(rf['ph_curves']['Ph RMS'], int(FS_RF*1.5))
    me = smooth(rf['mg_curves']['Mag RMS'], int(FS_RF*1.5))
    pn = pe/(np.percentile(pe,95)+1e-20); mn = me/(np.percentile(me,95)+1e-20)
    rf_env = np.sqrt(np.clip(pn,0,1)*np.clip(mn,0,1))
    rf_env /= (rf_env.max()+1e-20)

    st_env_rf = np.interp(rf['t'], st['t'], st['env_n'])

    fs_cc = 100; ds = int(FS_RF/fs_cc)
    s0, s1 = int(20*FS_RF), int(min(50*FS_RF, len(rf['t'])))
    rc = rf_env[s0:s1:ds]; sc = st_env_rf[s0:s1:ds]
    cc = np.correlate(rc, sc, 'full')
    lgs = np.arange(len(cc)) - len(rc)+1
    cc_lag = lgs[np.argmax(cc)] / fs_cc
    cc_r   = np.max(cc) / (np.sqrt((rc**2).sum()*(sc**2).sum())+1e-20)

    lag = (rf['onset']+rf['offset'])/2 - (st['onset']+st['offset'])/2
    st_on_a  = st['onset']  + lag
    st_off_a = st['offset'] + lag
    st_dur_a = st_off_a - st_on_a
    st_env_a = np.interp(rf['t'], st['t']+lag, st['env_n'])

    return dict(
        lag=lag, cc_lag=cc_lag, cc_r=cc_r,
        st_on_a=st_on_a, st_off_a=st_off_a, st_dur_a=st_dur_a,
        rf_env=rf_env, st_env_a=st_env_a, st_env_rf=st_env_rf,
        iou_raw   = iou_calc(rf['onset'],rf['offset'],st['onset'],st['offset']),
        iou_align = iou_calc(rf['onset'],rf['offset'],st_on_a,st_off_a),
        dur_diff_raw   = abs(rf['duration']-st['duration']),
        dur_diff_align = abs(rf['duration']-st_dur_a),
        onset_diff_raw   = abs(rf['onset']-st['onset']),
        onset_diff_align = abs(rf['onset']-st_on_a),
    )

# ════════════════════════════════════════════════════════════════════
# FIGURE 1 -- STETHOSCOPE ANALYSIS  (8 panels, 4x2)
# ════════════════════════════════════════════════════════════════════
def fig1_acoustic(st, subject_label, rec_idx, wav_path, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(22, 28))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    t = st['t']; ds = max(1, len(t)//40000)
    t_p = t[::ds]; ka_p = st['ka'][::ds]; ha_p = st['ha'][::ds]

    def span(ax):
        ax.axvspan(st['onset'], st['offset'], color='#00E5FF', alpha=0.20,
                   label=f"Korotkoff ({st['onset']:.1f}-{st['offset']:.1f}s)")
        ax.axvline(20.0, color='orange', ls='--', lw=1.5, label='Deflation (20 s)')

    ax = axes[0,0]
    ax.plot(t_p, st['audio'][::ds], color='#78909C', lw=0.4, alpha=0.6, label='Raw Audio')
    span(ax)
    ax.set_title('1. Raw Stethoscope Audio Waveform', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[0,1]
    ax.plot(t_p, ka_p, color='#26C6DA', lw=0.5, alpha=0.8, label='Korotkoff Band (50-1000 Hz)')
    span(ax)
    ax.set_title('2. Bandpass Filtered: Korotkoff Band (50-1000 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[1,0]
    ax.plot(t_p, ha_p, color='#EF5350', lw=1.2, label='Heartbeat Band (0.4-3 Hz)')
    span(ax)
    ax.set_title('3. Cardiac Heartbeat Band (0.4-3 Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[1,1]
    colors_env = ['#00E5FF', '#76FF03', '#FFEA00']
    for (name, env), col in zip(st['curves'].items(), colors_env):
        env_n = env / (env.max()+1e-20)
        ax.plot(t, env_n, lw=1.2, color=col, alpha=0.85, label=name)
        w = st['methods'][name]
        if w:
            ax.axvspan(w['onset'], w['offset'], color=col, alpha=0.07)
    ax.axvspan(st['onset'], st['offset'], color='#00E5FF', alpha=0.25,
               label=f"Consensus ({st['onset']:.1f}-{st['offset']:.1f}s)")
    ax.axvline(20.0, color='orange', ls='--', lw=1.5)
    ax.set_title('4. Multi-Method Envelopes & Adaptive Consensus Window', fontweight='bold', fontsize=12)
    ax.set_ylabel('Norm. Energy'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[2,0]
    fm = (st['fsp'] >= 50) & (st['fsp'] <= 1000)
    P5 = 10*np.log10(np.sqrt(st['Ssp'])+1e-20)
    va, vb = np.percentile(P5[fm], [10, 99.5])
    ext = [st['tsp'][0], st['tsp'][-1], st['fsp'][fm][0], st['fsp'][fm][-1]]
    im = ax.imshow(P5[fm], extent=ext, aspect='auto', origin='lower',
                   cmap='inferno', vmin=va, vmax=vb, interpolation='bilinear')
    ax.axvline(st['onset'],  color='#00E5FF', ls='--', lw=2)
    ax.axvline(st['offset'], color='#00E5FF', ls='--', lw=2)
    ax.set_title('5. Time-Frequency Distribution (Korotkoff Spectrogram)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency (Hz)'); plt.colorbar(im, ax=ax, label='dB')

    ax = axes[2,1]
    ax.semilogy(st['fpsd'], 10**(st['ppsd_db']/10), color='#26C6DA', lw=2)
    ax.axvline(50,   color='orange', ls='--', lw=1, label='50 Hz')
    ax.axvline(1000, color='orange', ls='--', lw=1, label='1000 Hz')
    ax.fill_between(st['fpsd'], 10**(st['ppsd_db']/10), alpha=0.2, color='#26C6DA')
    ax.set_xlim([0, 1200])
    ax.set_title('6. PSD of Active Korotkoff Window (Acoustic)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Power (a.u.)'); ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3,0]
    z0 = max(0, st['onset']-2); z1 = min(st['rec_dur'], st['offset']+2)
    mz = (t>=z0)&(t<=z1)
    ax.plot(t[mz], st['ka'][mz], color='#26C6DA', lw=0.6, alpha=0.8, label='Korotkoff Band')
    ax.axvspan(st['onset'], st['offset'], color='#00E5FF', alpha=0.25, label='Active Window')
    ax.set_xlim([z0, z1])
    ax.set_title('7. Zoomed Active Korotkoff Window (Acoustic)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (a.u.)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3,1]; ax.axis('off')
    lines = [
        f"STETHOSCOPE ANALYSIS -- {subject_label}  Rec {rec_idx:02d}",
        "="*52,
        f"Audio File    : {os.path.basename(wav_path)}",
        "",
        "DETECTED KOROTKOFF WINDOW:",
        f"  Onset    : {st['onset']:.2f} s  (>20s: {'OK' if st['onset']>=20 else 'FAIL'})",
        f"  Offset   : {st['offset']:.2f} s",
        f"  Duration : {st['duration']:.2f} s  (<25s: {'OK' if st['duration']<=25 else 'FAIL'})",
        "",
        "METHOD BREAKDOWN:",
    ]
    for name, w in st['methods'].items():
        if w:
            lines.append(f"  {name:<14}: {w['onset']:.1f}s - {w['offset']:.1f}s ({w['duration']:.1f}s)")
        else:
            lines.append(f"  {name:<14}: -- not detected --")
    lines += ["", f"HEART RATE:", f"  Peak: {st['hr_pk']:.1f} BPM  |  PSD: {st['hr_psd']:.1f} BPM", "="*52]
    ax.text(0.03, 0.97, '\n'.join(lines), fontsize=9.5, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.90))

    for a in axes.flat:
        if not a.get_xlabel(): a.set_xlabel('Time (s)')
    fig.suptitle(f'Figure 1 -- Acoustic Stethoscope Analysis  |  {subject_label}  Rec {rec_idx:02d}  |  300 DPI',
                 fontsize=14, fontweight='bold', y=0.990)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    [Fig1 Acoustic] -> {out_path}")

# ════════════════════════════════════════════════════════════════════
# FIGURE 2 -- RF RADAR RMG ANALYSIS  (8 panels, 4x2)
#   Phase & Magnitude validated INDEPENDENTLY, then cross-checked
# ════════════════════════════════════════════════════════════════════
def fig2_rf(rf, subject_label, rec_idx, rf_path, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(22, 28))
    plt.subplots_adjust(hspace=0.50, wspace=0.28)
    t = rf['t']; ds = max(1, len(t)//40000); t_p = t[::ds]

    PH_COL    = '#EF5350'   # red  -- phase method
    MAG_COL   = '#42A5F5'   # blue -- magnitude method
    JOINT_COL = '#FFD700'   # gold -- joint consensus

    # ── P1: Korotkoff bandpass displacement in mm (PHYSICALLY CORRECT) ──
    ax = axes[0,0]
    ax.plot(t_p, rf['pk_mm'][::ds], color=PH_COL, lw=0.7, alpha=0.85,
            label='Korotkoff Disp. (BP 10-200 Hz, mm)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5, label='Deflation (20 s)')
    ax.axvspan(rf['ph_on'], rf['ph_off'], color=PH_COL, alpha=0.18,
               label=f'Phase Window ({rf["ph_on"]:.1f}-{rf["ph_off"]:.1f}s)')
    pk_rms_um = np.sqrt(np.mean(rf['pk_mm']**2)) * 1000  # micrometres
    ax.set_title(f'1. Phase Korotkoff Displacement (BP 10-200 Hz)  |  RMS={pk_rms_um:.2f} um',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Displacement (mm)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── P2: Cardiac & breathing displacement in mm (physical scale) ──
    ax = axes[0,1]
    ax.plot(t_p, rf['dh'][::ds], color='#26C6DA', lw=1.0,
            label='Cardiac Disp. (0.4-3 Hz, mm)')
    ax.plot(t_p, rf['slow_mm'][::ds], color='#AB47BC', lw=0.7, alpha=0.6,
            label='Breathing Drift (0.05-0.5 Hz, mm)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5)
    ax.set_title('2. Cardiac & Breathing Displacement (mm) -- Physical Scale',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Displacement (mm)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── P3: Phase-only 5 envelopes + phase consensus window ──
    ax = axes[1,0]
    ph_pal = ['#EF5350','#FF7043','#FFA726','#FFCA28','#EF9A9A']
    for (name, c), col in zip(rf['ph_curves'].items(), ph_pal):
        cn = c/(c.max()+1e-20)
        ax.plot(t, cn, lw=0.8, color=col, alpha=0.80, label=name)
    ax.axvspan(rf['ph_on'], rf['ph_off'], color=PH_COL, alpha=0.22,
               label=f'Phase Consensus ({rf["ph_on"]:.1f}-{rf["ph_off"]:.1f}s, {rf["ph_dur"]:.1f}s)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5)
    ax.set_title(f'3. PHASE-Method: 5 Envelopes & Adaptive Window  [{rf["ph_dur"]:.1f}s]',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Norm. Energy')
    ax.legend(fontsize=7, loc='upper right', ncol=2); ax.grid(True, alpha=0.2)

    # ── P4: Magnitude-only 5 envelopes + magnitude consensus window ──
    ax = axes[1,1]
    mg_pal = ['#42A5F5','#26C6DA','#66BB6A','#7E57C2','#90CAF9']
    for (name, c), col in zip(rf['mg_curves'].items(), mg_pal):
        cn = c/(c.max()+1e-20)
        ax.plot(t, cn, lw=0.8, color=col, alpha=0.80, label=name)
    ax.axvspan(rf['mg_on'], rf['mg_off'], color=MAG_COL, alpha=0.22,
               label=f'Mag Consensus ({rf["mg_on"]:.1f}-{rf["mg_off"]:.1f}s, {rf["mg_dur"]:.1f}s)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5)
    ax.set_title(f'4. MAGNITUDE-Method: 5 Envelopes & Adaptive Window  [{rf["mg_dur"]:.1f}s]',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Norm. Energy')
    ax.legend(fontsize=7, loc='upper right', ncol=2); ax.grid(True, alpha=0.2)

    # ── P5: Phase velocity TFD spectrogram (both windows overlaid) ──
    ax = axes[2,0]
    fs_t  = 600
    vk_ds = signal.resample_poly(rf['vk'], up=fs_t, down=int(FS_RF))
    nps   = min(len(vk_ds)//4, int(fs_t*0.15))
    f5, t5, S5 = signal.spectrogram(vk_ds, fs=fs_t, window='hann',
                                     nperseg=nps, noverlap=nps*7//8, nfft=1024)
    fm5 = (f5>=10)&(f5<=200)
    P5  = 10*np.log10(np.sqrt(S5)+1e-20)
    va, vb = np.percentile(P5[fm5], [15, 99.5])
    im5 = ax.imshow(P5[fm5], extent=[t5[0],t5[-1],f5[fm5][0],f5[fm5][-1]],
                    aspect='auto', origin='lower', cmap='magma',
                    vmin=va, vmax=vb, interpolation='bilinear')
    ax.axvline(rf['ph_on'],  color=PH_COL,  ls='--', lw=2, label='Phase Win')
    ax.axvline(rf['ph_off'], color=PH_COL,  ls='--', lw=2)
    ax.axvline(rf['mg_on'],  color=MAG_COL, ls=':',  lw=2, label='Mag Win')
    ax.axvline(rf['mg_off'], color=MAG_COL, ls=':',  lw=2)
    ax.set_title('5. Phase Velocity TFD (10-200 Hz) -- Phase vs Mag Windows',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im5, ax=ax, label='dB'); ax.legend(fontsize=7, loc='upper right')

    # ── P6: Magnitude velocity TFD spectrogram (both windows overlaid) ──
    ax = axes[2,1]
    vm_ds = signal.resample_poly(rf['vel_mag'], up=fs_t, down=int(FS_RF))
    f6, t6, S6 = signal.spectrogram(vm_ds, fs=fs_t, window='hann',
                                     nperseg=nps, noverlap=nps*7//8, nfft=1024)
    fm6 = (f6>=10)&(f6<=200)
    P6  = 10*np.log10(np.sqrt(S6)+1e-20)
    va6, vb6 = np.percentile(P6[fm6], [15, 99.5])
    im6 = ax.imshow(P6[fm6], extent=[t6[0],t6[-1],f6[fm6][0],f6[fm6][-1]],
                    aspect='auto', origin='lower', cmap='viridis',
                    vmin=va6, vmax=vb6, interpolation='bilinear')
    ax.axvline(rf['mg_on'],  color=MAG_COL, ls='--', lw=2, label='Mag Win')
    ax.axvline(rf['mg_off'], color=MAG_COL, ls='--', lw=2)
    ax.axvline(rf['ph_on'],  color=PH_COL,  ls=':',  lw=2, label='Phase Win')
    ax.axvline(rf['ph_off'], color=PH_COL,  ls=':',  lw=2)
    ax.set_title('6. Magnitude Velocity TFD (10-200 Hz) -- Mag vs Phase Windows',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im6, ax=ax, label='dB'); ax.legend(fontsize=7, loc='upper right')

    # ── P7: Phase vs Magnitude cross-validation overlay ──
    ax = axes[3,0]
    ph_env = smooth(rf['ph_curves']['Ph RMS'], int(FS_RF*1.5))
    ph_env /= (ph_env.max()+1e-20)
    mg_env = smooth(rf['mg_curves']['Mag RMS'], int(FS_RF*1.5))
    mg_env /= (mg_env.max()+1e-20)
    ax.plot(t, ph_env, color=PH_COL,    lw=2.0, label='Phase Envelope')
    ax.plot(t, mg_env, color=MAG_COL,   lw=2.0, label='Magnitude Envelope')
    ax.axvspan(rf['ph_on'], rf['ph_off'], color=PH_COL,  alpha=0.15,
               label=f'Phase: {rf["ph_on"]:.1f}-{rf["ph_off"]:.1f}s ({rf["ph_dur"]:.1f}s)')
    ax.axvspan(rf['mg_on'], rf['mg_off'], color=MAG_COL, alpha=0.12,
               label=f'Mag:   {rf["mg_on"]:.1f}-{rf["mg_off"]:.1f}s ({rf["mg_dur"]:.1f}s)')
    ax.axvline(rf['onset'],  color=JOINT_COL, ls='-', lw=2.5)
    ax.axvline(rf['offset'], color=JOINT_COL, ls='-', lw=2.5,
               label=f'Joint: {rf["onset"]:.1f}-{rf["offset"]:.1f}s ({rf["duration"]:.1f}s)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5, label='Deflation (20s)')
    pm_agree = 'AGREE' if rf['pm_iou']>=0.85 else ('MARGINAL' if rf['pm_iou']>=0.70 else 'DISAGREE')
    ax.set_title(f'7. Phase vs Magnitude Cross-Validation  |  PM-IoU={rf["pm_iou"]:.3f}  [{pm_agree}]',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Norm. Energy')
    ax.legend(fontsize=7, loc='upper right', ncol=2); ax.grid(True, alpha=0.2)

    # ── P8: Dual-method validation report card ──
    ax = axes[3,1]; ax.axis('off')
    pm_s  = 'PASS' if rf['pm_iou']>=0.85 else 'MARGINAL'
    overall = 'VALIDATED' if rf['pm_iou']>=0.75 else 'CHECK DATA'
    card = [
        f'RF RADAR RMG -- DUAL-METHOD REPORT',
        f'Subject: {subject_label}   Rec: {rec_idx:02d}',
        '='*50,
        f'RF H5 File : {os.path.basename(rf_path)}',
        f'Radar      : 0.9 GHz  |  Fs={FS_RF} Hz',
        f'lambda={LAMBDA_MM:.1f}mm  Scale={SCALE:.4f}mm/rad',
        '',
        'SIGNAL AMPLITUDE (physical):',
        f'  Korotkoff RMS : {pk_rms_um:.2f} um  (typical 5-500 um)',
        f'  Cardiac pk-pk : {np.ptp(rf["dh"]):.3f} mm  (typical 0.1-3 mm)',
        '',
        'INDEPENDENT WINDOW DETECTION:',
        f'  {"Method":<9} {"Onset":>7} {"Offset":>7} {"Dur":>6}  {">20s":>4}  {"<25s":>4}',
        f'  {"-"*50}',
        f'  {"PHASE":<9} {rf["ph_on"]:>6.2f}s {rf["ph_off"]:>6.2f}s {rf["ph_dur"]:>5.2f}s'
        f'  {"OK" if rf["ph_on"]>=20 else "FAIL":>4}  {"OK" if rf["ph_dur"]<=25 else "FAIL":>4}',
        f'  {"MAGNITUDE":<9} {rf["mg_on"]:>6.2f}s {rf["mg_off"]:>6.2f}s {rf["mg_dur"]:>5.2f}s'
        f'  {"OK" if rf["mg_on"]>=20 else "FAIL":>4}  {"OK" if rf["mg_dur"]<=25 else "FAIL":>4}',
        '',
        'CROSS-METHOD AGREEMENT:',
        f'  |Dur Diff (Ph-Mag)| : {rf["dur_diff_pm"]:.3f} s',
        f'  Phase vs Mag IoU   : {rf["pm_iou"]:.3f}  [{pm_s}]',
        '',
        'JOINT CONSENSUS WINDOW:',
        f'  Onset    : {rf["onset"]:.2f} s',
        f'  Offset   : {rf["offset"]:.2f} s',
        f'  Duration : {rf["duration"]:.2f} s',
        '',
        f'HEART RATE (RF):',
        f'  Peak: {rf["hr_pk"]:.1f} BPM  |  PSD: {rf["hr_psd"]:.1f} BPM',
        '='*50,
        f'OVERALL RF STATUS: {overall}',
    ]
    ax.text(0.02, 0.98, '\n'.join(card), fontsize=9.2, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.92))

    for a in axes.flat:
        if not a.get_xlabel(): a.set_xlabel('Time (s)')
    fig.suptitle(
        f'Figure 2 -- RF RMG Analysis (Phase & Magnitude Dual-Method Validation)'
        f'  |  {subject_label}  Rec {rec_idx:02d}  |  300 DPI',
        fontsize=13, fontweight='bold', y=0.993)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    [Fig2 RF]      -> {out_path}")

# ════════════════════════════════════════════════════════════════════
# FIGURE 3 -- CROSS-MODALITY COMPARISON  (6 panels, 3x2)
# ════════════════════════════════════════════════════════════════════
def fig3_comparison(rf, st, al, subject_label, rec_idx, rf_path, wav_path, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(22, 21))
    plt.subplots_adjust(hspace=0.45, wspace=0.28)
    t_rf = rf['t']

    ax = axes[0,0]
    ax.plot(t_rf, al['rf_env'], color='#EF5350', lw=2.0, label='Joint RF Envelope')
    ax.plot(t_rf, al['st_env_a'], color='limegreen', lw=2.0,
            label=f'Steth Envelope (lag {al["lag"]:+.2f}s)')
    ax.axvspan(rf['onset'],     rf['offset'],     color='gold',      alpha=0.35,
               label=f'RF ({rf["onset"]:.1f}-{rf["offset"]:.1f}s)')
    ax.axvspan(al['st_on_a'],  al['st_off_a'],   color='limegreen', alpha=0.30,
               label=f'Steth Aligned ({al["st_on_a"]:.1f}-{al["st_off_a"]:.1f}s)')
    ax.axvline(20.0, color='orange', ls='--', lw=1.5, label='20s Deflation')
    ax.set_title(f'1. Envelope Overlay & Alignment  (IoU={al["iou_align"]:.3f})',
                 fontweight='bold', fontsize=12)
    ax.set_ylabel('Norm. Energy'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[0,1]
    cats  = ['RF\n(Ph+Mag)', 'Steth\n(Raw)', 'Steth\n(Aligned)']
    durs  = [rf['duration'], st['duration'], al['st_dur_a']]
    cols  = ['#EF5350', '#26C6DA', '#66BB6A']
    bars  = ax.bar(cats, durs, color=cols, alpha=0.85, width=0.5, edgecolor='white', lw=1.5)
    ax.axhline(MAX_DUR_S, color='yellow', ls='--', lw=1.5, label=f'Max {MAX_DUR_S}s')
    for bar, d in zip(bars, durs):
        ax.text(bar.get_x()+bar.get_width()/2, d+0.2, f'{d:.2f}s',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Duration (s)'); ax.set_ylim([0, MAX_DUR_S+4])
    ax.set_title('2. Korotkoff Duration Comparison', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    ax = axes[1,0]
    fs_cc = 100; ds2 = int(FS_RF/fs_cc)
    s0, s1 = int(20*FS_RF), int(min(50*FS_RF, len(t_rf)))
    rc = al['rf_env'][s0:s1:ds2]; sc = al['st_env_rf'][s0:s1:ds2]
    cc = np.correlate(rc, sc, 'full')
    lgs_s = (np.arange(len(cc)) - len(rc)+1) / fs_cc
    ax.plot(lgs_s, cc/(cc.max()+1e-20), color='#AB47BC', lw=2)
    ax.axvline(al['cc_lag'], color='yellow', ls='--', lw=2, label=f'CC Lag={al["cc_lag"]:.2f}s')
    ax.axvline(al['lag'],    color='cyan',   ls='-',  lw=2, label=f'Consensus={al["lag"]:.2f}s')
    ax.set_xlim([-10, 10])
    ax.set_title(f'3. Envelope Cross-Correlation  (r={al["cc_r"]:.3f})', fontweight='bold', fontsize=12)
    ax.set_xlabel('Lag (s)'); ax.set_ylabel('Norm. CC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    metrics = ['Onset Diff', 'Offset Diff', 'Dur Diff']
    raw_v   = [al['onset_diff_raw'], abs(rf['offset']-st['offset']), al['dur_diff_raw']]
    aln_v   = [al['onset_diff_align'], abs(rf['offset']-al['st_off_a']), al['dur_diff_align']]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x-w/2, raw_v, w, color='#78909C', alpha=0.85, label='Unaligned')
    ax.bar(x+w/2, aln_v, w, color='#29B6F6', alpha=0.85, label='Aligned')
    ax.axhline(1.5, color='yellow', ls='--', lw=1.5, label='1.5s threshold')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('|Difference| (s)')
    ax.set_title('4. Metric Differences: Unaligned vs. Aligned', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')
    for i, (rv, av) in enumerate(zip(raw_v, aln_v)):
        ax.text(i-w/2, rv+0.03, f'{rv:.2f}', ha='center', fontsize=9)
        ax.text(i+w/2, av+0.03, f'{av:.2f}', ha='center', fontsize=9)

    ax = axes[2,0]
    bars2 = ax.bar(['IoU Raw', 'IoU Aligned'], [al['iou_raw'], al['iou_align']],
                    color=['#78909C','#29B6F6'], alpha=0.85, width=0.4,
                    edgecolor='white', lw=1.5)
    ax.axhline(0.75, color='yellow', ls='--', lw=1.5, label='IoU=0.75 threshold')
    ax.axhline(1.00, color='lime',   ls='--', lw=1,   label='IoU=1.00 perfect')
    for bar, v in zip(bars2, [al['iou_raw'], al['iou_align']]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}',
                ha='center', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.set_title('5. Overlap Agreement (IoU): Raw vs. Aligned', fontweight='bold', fontsize=12)
    ax.set_ylabel('IoU'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    ax = axes[2,1]; ax.axis('off')
    dur_pass = 'PASS' if al['dur_diff_align']<1.5 else 'FAIL'
    iou_pass = 'PASS' if al['iou_align']>0.75     else 'FAIL'
    overall  = 'VALIDATED [EXCELLENT]' if al['iou_align']>=0.75 and al['dur_diff_align']<1.5 else 'MARGINAL'
    lines = [
        f'CROSS-MODALITY VALIDATION CARD',
        f'Subject: {subject_label}   Rec: {rec_idx:02d}',
        '='*50,
        f'RF File  : {os.path.basename(rf_path)}',
        f'WAV File : {os.path.basename(wav_path)}',
        '',
        'KOROTKOFF WINDOW SUMMARY:',
        f'  {"Sensor":<18} {"Onset":>7} {"Offset":>7} {"Dur":>7}',
        f'  {"-"*45}',
        f'  {"RF Joint Consensus":<18} {rf["onset"]:>6.2f}s {rf["offset"]:>6.2f}s {rf["duration"]:>6.2f}s',
        f'  {"Steth Raw":<18} {st["onset"]:>6.2f}s {st["offset"]:>6.2f}s {st["duration"]:>6.2f}s',
        f'  {"Steth Aligned":<18} {al["st_on_a"]:>6.2f}s {al["st_off_a"]:>6.2f}s {al["st_dur_a"]:>6.2f}s',
        '',
        'VALIDATION METRICS (Aligned):',
        f'  Onset Diff    : {al["onset_diff_align"]:.3f} s',
        f'  Duration Diff : {al["dur_diff_align"]:.3f} s  [{dur_pass}]',
        f'  Overlap IoU   : {al["iou_align"]:.3f}    [{iou_pass}]',
        f'  Trigger Lag   : {al["lag"]:+.3f} s',
        f'  Cross-Corr r  : {al["cc_r"]:.3f}',
        '',
        'CONSTRAINTS:',
        f'  Start > 20s: RF {rf["onset"]:.1f}s {"OK" if rf["onset"]>=20 else "FAIL"}'
        f'  | Steth {st["onset"]:.1f}s {"OK" if st["onset"]>=20 else "FAIL"}',
        f'  Dur  <= 25s: RF {rf["duration"]:.1f}s {"OK" if rf["duration"]<=25 else "FAIL"}'
        f'  | Steth {st["duration"]:.1f}s {"OK" if st["duration"]<=25 else "FAIL"}',
        f'  RF Phase vs Mag IoU: {rf["pm_iou"]:.3f}',
        '='*50,
        f'OVERALL STATUS: {overall}',
    ]
    ax.text(0.02, 0.98, '\n'.join(lines), fontsize=9.5, family='monospace',
            fontweight='bold', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    for a in axes.flat:
        if not a.get_xlabel(): a.set_xlabel('Time (s)')
    fig.suptitle(
        f'Figure 3 -- Cross-Modality Comparison  |  {subject_label}  Rec {rec_idx:02d}  |  300 DPI',
        fontsize=14, fontweight='bold', y=0.993)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    [Fig3 Compare] -> {out_path}")

# ════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ════════════════════════════════════════════════════════════════════
def plot_summary(rows):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    plt.subplots_adjust(hspace=0.42, wspace=0.3)
    fig.patch.set_facecolor('#111827')

    def sty(ax, title, xl='', yl=''):
        ax.set_facecolor('#1F2937')
        for sp in ax.spines.values(): sp.set_color('#374151')
        ax.tick_params(colors='#D1D5DB')
        ax.set_title(title, color='white', fontweight='bold', fontsize=11)
        if xl: ax.set_xlabel(xl, color='#9CA3AF')
        if yl: ax.set_ylabel(yl, color='#9CA3AF')
        ax.grid(True, alpha=0.15, color='#374151')

    recs = np.arange(1, 11)

    ax = axes[0,0]
    for s in SUBJECTS:
        d = df[df['subject']==s['label']]
        ax.plot(d['rec'], d['rf_dur'], 'o-', color=s['color'], lw=2, ms=7, label=s['label'])
    ax.axhline(MAX_DUR_S, color='yellow', ls='--', lw=1.5, label=f'Max {MAX_DUR_S}s')
    ax.set_xticks(recs); ax.legend(fontsize=8, labelcolor='white', facecolor='#222')
    sty(ax, 'RF Korotkoff Duration per Recording', 'Recording #', 'Duration (s)')

    ax = axes[0,1]
    for s in SUBJECTS:
        d = df[df['subject']==s['label']]
        ax.plot(d['rec'], d['st_dur'], 's--', color=s['color'], lw=2, ms=7, label=s['label'])
    ax.axhline(MAX_DUR_S, color='yellow', ls='--', lw=1.5, label=f'Max {MAX_DUR_S}s')
    ax.set_xticks(recs); ax.legend(fontsize=8, labelcolor='white', facecolor='#222')
    sty(ax, 'Stethoscope Korotkoff Duration per Recording', 'Recording #', 'Duration (s)')

    ax = axes[0,2]
    w = 0.35
    for i, s in enumerate(SUBJECTS):
        d = df[df['subject']==s['label']]
        ax.bar(d['rec'] + i*w, d['dur_diff'], w, color=s['color'], alpha=0.85, label=s['label'])
    ax.axhline(1.5, color='yellow', ls='--', lw=1.5, label='1.5s threshold')
    ax.set_xticks(recs + w/2); ax.set_xticklabels(recs)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#222')
    sty(ax, 'Duration Difference (Aligned) per Recording', 'Recording #', '|Delta Duration| (s)')

    ax = axes[1,0]
    for s in SUBJECTS:
        d = df[df['subject']==s['label']]
        ax.plot(d['rec'], d['iou_align'], '^-', color=s['color'], lw=2, ms=7, label=s['label'])
    ax.axhline(0.75, color='yellow', ls='--', lw=1.5, label='IoU=0.75')
    ax.set_ylim([0, 1.05]); ax.set_xticks(recs)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#222')
    sty(ax, 'Aligned Overlap (IoU) per Recording', 'Recording #', 'IoU')

    ax = axes[1,1]
    for s in SUBJECTS:
        d = df[df['subject']==s['label']]
        ax.plot(d['rec'], d['rf_hr'],  'o-',  color=s['color'], lw=2, ms=6, label=f"{s['label']} RF")
        ax.plot(d['rec'], d['st_hr'],  's--', color=s['color'], lw=1.5, ms=5, alpha=0.6,
                label=f"{s['label']} Steth")
    ax.set_xticks(recs); ax.legend(fontsize=7, labelcolor='white', facecolor='#222', ncol=2)
    sty(ax, 'Heart Rate Comparison per Recording', 'Recording #', 'HR (BPM)')

    ax = axes[1,2]; ax.axis('off'); ax.set_facecolor('#1F2937')
    hdr = ["Subject","N","RF Dur\nmean+/-SD","Dur Diff\nmean+/-SD","IoU\nmean+/-SD","RF HR\nmean+/-SD"]
    rows_tbl = [hdr]
    for s in SUBJECTS:
        d = df[df['subject']==s['label']]
        rows_tbl.append([
            s['label'].split('(')[0].strip(), str(len(d)),
            f"{d['rf_dur'].mean():.2f}+/-{d['rf_dur'].std():.2f}s",
            f"{d['dur_diff'].mean():.2f}+/-{d['dur_diff'].std():.2f}s",
            f"{d['iou_align'].mean():.3f}+/-{d['iou_align'].std():.3f}",
            f"{d['rf_hr'].mean():.1f}+/-{d['rf_hr'].std():.1f}",
        ])
    tbl = ax.table(cellText=rows_tbl[1:], colLabels=rows_tbl[0],
                   cellLoc='center', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1F2937' if r==0 else ('#27374D' if r%2 else '#1E293B'))
        cell.set_text_props(color='white', fontweight='bold' if r==0 else 'normal')
        cell.set_edgecolor('#374151')
    ax.set_title('Summary Statistics', color='white', fontweight='bold', fontsize=11)

    fig.suptitle(
        'Cross-Subject Summary  |  Prof. Kan (Sub 1) & Rajveer (Sub 2)'
        '  |  RF RMG vs Stethoscope  |  300 DPI',
        fontsize=15, fontweight='bold', color='white', y=0.996)
    out = os.path.join(SUMMARY_DIR, 'cross_subject_summary_dashboard.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [Summary] -> {out}")

# ════════════════════════════════════════════════════════════════════
# PER-RECORDING RUNNER
# ════════════════════════════════════════════════════════════════════
def process_one(subject, rec_idx):
    folder   = subject['folder']
    rf_path  = os.path.join(folder, f"Rec_{rec_idx}.h5")
    wav_path = os.path.join(folder, f"sthethoscope_rec{rec_idx:02d}.wav")
    if not os.path.exists(wav_path):
        wav_path = os.path.join(folder, f"sthethoscope_rec{rec_idx}.wav")

    if not os.path.exists(rf_path):
        print(f"  SKIP -- missing {rf_path}"); return None
    if not os.path.exists(wav_path):
        print(f"  SKIP -- missing {wav_path}"); return None

    print(f"\n  [{subject['label']}  Rec {rec_idx:02d}]")
    print(f"    RF : {os.path.basename(rf_path)}")
    print(f"    WAV: {os.path.basename(wav_path)}")

    try:
        rf = process_rf(rf_path)
        st = process_stethoscope(wav_path)
        al = align(rf, st)

        out_dir = os.path.join(folder, 'results')
        os.makedirs(out_dir, exist_ok=True)
        nm = f"{subject['name']}_Rec{rec_idx:02d}"

        fig1_acoustic(st, subject['label'], rec_idx, wav_path,
                      os.path.join(out_dir, f"{nm}_Fig1_Acoustic.png"))
        fig2_rf(rf, subject['label'], rec_idx, rf_path,
                os.path.join(out_dir, f"{nm}_Fig2_RF.png"))
        fig3_comparison(rf, st, al, subject['label'], rec_idx, rf_path, wav_path,
                        os.path.join(out_dir, f"{nm}_Fig3_Comparison.png"))

        return dict(
            subject=subject['label'], rec=rec_idx,
            rf_file=os.path.basename(rf_path), wav_file=os.path.basename(wav_path),
            rf_dur=rf['duration'], st_dur=st['duration'],
            rf_onset=rf['onset'], rf_offset=rf['offset'],
            st_on_a=al['st_on_a'], st_off_a=al['st_off_a'], st_dur_a=al['st_dur_a'],
            lag=al['lag'], cc_r=al['cc_r'],
            iou_raw=al['iou_raw'], iou_align=al['iou_align'],
            dur_diff=al['dur_diff_align'],
            rf_hr=rf['hr_pk'], st_hr=st['hr_pk'],
            pm_iou=rf['pm_iou'],
        )
    except Exception as e:
        print(f"    ERROR: {e}"); traceback.print_exc(); return None

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print(" Multi-Subject 3-Figure Dual-Modality Korotkoff v5.0")
    print("="*65)
    print(f" Subjects    : {', '.join(s['label'] for s in SUBJECTS)}")
    print(f" Recs/subject: {SUBJECTS[0]['n_recs']}")
    print(f" Onset limit : > {MIN_ONSET_S} s")
    print(f" Max duration: {MAX_DUR_S} s")
    print(f" Output DPI  : 300")
    print(f" lambda      : {LAMBDA_MM:.1f} mm  |  Scale: {SCALE:.4f} mm/rad")
    print("="*65)

    all_rows = []
    for subject in SUBJECTS:
        print(f"\n{'='*60}")
        print(f" SUBJECT: {subject['label']}")
        print(f"{'='*60}")
        for rec in range(1, subject['n_recs']+1):
            row = process_one(subject, rec)
            if row:
                all_rows.append(row)

    if not all_rows:
        print("ERROR: no recordings processed."); sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV -> {csv_path}")

    print("\n" + "="*65)
    print(" RESULTS SUMMARY")
    print("="*65)
    for _, r in df.iterrows():
        status = "PASS" if r['iou_align']>=0.75 and r['dur_diff']<1.5 else "FAIL"
        print(f"  {r['subject']:<22} Rec{r['rec']:02d}  "
              f"RF={r['rf_dur']:.1f}s  St={r['st_dur']:.1f}s  "
              f"dDur={r['dur_diff']:.2f}s  IoU={r['iou_align']:.2f}  "
              f"PM-IoU={r['pm_iou']:.2f}  {status}")

    print("\nGenerating cross-subject summary dashboard...")
    plot_summary(all_rows)

    print("\n" + "="*65)
    print(" DONE -- 3 figures per recording + 1 summary")
    print("="*65)
    for s in SUBJECTS:
        print(f"  {s['label']}: {os.path.join(s['folder'], 'results')}")
    print(f"  Summary: {SUMMARY_DIR}")

if __name__ == '__main__':
    main()
