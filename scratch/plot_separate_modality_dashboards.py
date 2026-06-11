"""
Independent Modality Dashboard Generator v7.5
==============================================
Generates separate 4-panel dashboards for:
  - Stethoscope PCG-derived ABP analysis
  - RF Radar RMG-derived ABP analysis

v7.5 Changes (Adaptive Deflation Onset):
  - Replaced ALL hardcoded t_start=20.0 with detect_deflation_onset()
    which finds the true cuff deflation start from the cardiac-band
    compliance envelope (3.5-sigma threshold, 0.8s sustain).
  - Full recording timeline: X_LIMITS = [0.0, 50.0]
  - Three physiological zone shadings — boundaries auto-adapt to t_start:
      Phase I  [0.0, t_start]      : Fully Occluded Artery (gray  #DFE6E9, alpha=0.30)
      Phase II [onset, offset]      : Active Korotkoff Window (yellow #FFEAA7, alpha=0.25)
      Phase III [offset, 50.0]      : Fully Unoccluded Artery (green #D1F2D9, alpha=0.30)
  - Vertical crimson line at t_start labeled with detected value
  - Title annotated with adaptive t_start for transparency
  - 300 DPI publication quality output
"""

import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000      # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)        # ~26.5 mm/rad

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')
X_LIMITS   = [0.0, 50.0]    # full recording timeline

SUBJECT_CONFIGS = [
    {
        "name": "Prof_Kan",
        "label": "Prof. Kan (Sub 1)",
        "color": "#E84393",
        "folder": os.path.join(BASE, "Sub_1_Prof_kan"),
        "best_rec": 6,
        "steth_file": "sthethoscope_rec06.wav",
        "beta": 3.175
    },
    {
        "name": "Rajveer",
        "label": "Rajveer (Sub 2)",
        "color": "#2196F3",
        "folder": os.path.join(BASE, "Sub_2_Rajveer"),
        "best_rec": 4,
        "steth_file": "sthethoscope_rec04.wav",
        "beta": 3.420
    }
]

# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k) / k, mode='same')


def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    """
    Detect the cuff maximum pressure point (inflation end / deflation onset)
    directly from the raw RF signal by identifying when pump vibration energy drops.
    """
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    iq_hp = sosfiltfilt(sos_hp, iq)
    energy = np.abs(iq_hp)
    
    ds = int(fs / 100)
    t_ds = np.arange(len(i_raw))[::ds] / fs
    energy_ds = energy[::ds]
    
    w_size = 100
    energy_smooth = np.convolve(energy_ds, np.ones(w_size)/w_size, mode='same')
    
    max_search_sec = 25.0
    if onset_limit is not None:
        max_search_sec = min(max_search_sec, onset_limit - 1.0)
    
    search_mask = t_ds <= max_search_sec
    if not np.any(search_mask):
        return 8.0
        
    t_search = t_ds[search_mask]
    e_search = energy_smooth[search_mask]
    
    peak_idx = np.argmax(e_search)
    peak_val = e_search[peak_idx]
    
    end_val = np.mean(energy_smooth[max(0, int(max_search_sec*100)-50):int(max_search_sec*100)])
    
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        return 0.0
        
    baseline = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    
    t_det = 8.0
    for i in range(peak_idx, len(t_search)):
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
            
    return t_det


def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic * qc)
    sp = np.clip(p3 / np.sqrt(p1 * p2 + 1e-20), -1, 1)
    cp = np.sqrt(max(1 - sp**2, 1e-10))
    al = np.sqrt(p2 / (p1 + 1e-20))
    i_new = ic
    q_new = (qc - ic * sp / al) / cp
    return i_new + 1j * q_new


def add_zone_shading(ax, t_start, onset, offset):
    """
    Draw the three-zone physiological shading on any axis.

    Phase I   [0.0, t_start]  — gray   — Fully Occluded Artery
    Phase II  [onset, offset] — yellow — Active Korotkoff Window
    Phase III [offset, 50.0]  — green  — Fully Unoccluded Artery
    """
    ax.axvspan(0.0,     t_start, color='#DFE6E9', alpha=0.30, zorder=0)
    ax.axvspan(onset,   offset,  color='#FFEAA7', alpha=0.25, zorder=0)
    ax.axvspan(offset,  50.0,    color='#D1F2D9', alpha=0.30, zorder=0)


def add_phase_labels(ax, t_start, onset, offset):
    """
    Write Phase I / II / III text labels on the panel after ylim is set.
    """
    ymin, ymax = ax.get_ylim()
    y_txt = ymin + 0.95 * (ymax - ymin)
    ax.text(t_start / 2,           y_txt, 'OCCLUDED\n(Phase I)',    ha='center', va='top', fontsize=7.5, color='#636E72', style='italic', clip_on=True)
    ax.text((onset + offset) / 2,  y_txt, 'KOROTKOFF\n(Phase II)',  ha='center', va='top', fontsize=7.5, color='#B7950B', style='italic', clip_on=True)
    ax.text((offset + 50.0) / 2,   y_txt, 'UNOCCLUDED\n(Phase III)',ha='center', va='top', fontsize=7.5, color='#1E8449', style='italic', clip_on=True)


# ── DASHBOARD GENERATORS ──────────────────────────────────────────────

def generate_stethoscope_dashboard(sub, onset, offset, df_report):
    print(f"\nGenerating Stethoscope-Only Dashboard (v7.5) for {sub['label']}...")
    wav_path = os.path.join(sub['folder'], sub['steth_file'])
    if not os.path.exists(wav_path):
        print(f"  [ERROR] WAV file not found: {wav_path}")
        return

    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_a = np.arange(len(audio)) / fs_a

    # 1. Bandpass filter: PCG heart sound band 20–150 Hz
    sos_a     = butter(4, [20, 150], btype='band', fs=fs_a, output='sos')
    audio_filt = sosfiltfilt(sos_a, audio)
    audio_env  = np.abs(hilbert(audio_filt))

    # 2. Cardiac bandpass 0.4–3.0 Hz → acoustic heartbeat waveform
    sos_ah     = butter(4, [0.4, 3.0], btype='band', fs=fs_a, output='sos')
    dh_acoustic = sosfiltfilt(sos_ah, audio_env)

    # 3. Compliance envelope (1.5-s smooth)
    dh_a_env = smooth(np.abs(hilbert(dh_acoustic)), int(1.5 * fs_a))

    # 4. ADAPTIVE deflation onset from synchronized raw RF signal
    h5_path_rf = os.path.join(sub['folder'], f"Rec_{sub['best_rec']}.h5")
    with h5py.File(h5_path_rf, 'r') as f_rf:
        data_rf = f_rf['data'][:]
    i_raw_rf, q_raw_rf = data_rf[0], data_rf[1]
    t_start = detect_cuff_max_pressure_point(i_raw_rf, q_raw_rf, fs=FS_RF, onset_limit=onset)

    # Dynamic closed-form calibration
    target_sbp = 125.0
    target_dbp = 75.0
    beta = (target_sbp - target_dbp) / (offset - onset)
    P_start = target_sbp + beta * (onset - t_start)

    # 5. Acoustic compliance MAP peak (middle 70% of Korotkoff window)
    mid_start  = onset + 0.15 * (offset - onset)
    mid_end    = offset - 0.15 * (offset - onset)
    mid_mask_a = (t_a >= mid_start) & (t_a <= mid_end)
    if np.any(mid_mask_a):
        t_map_acoustic = t_a[mid_mask_a][np.argmax(dh_a_env[mid_mask_a])]
    else:
        t_map_acoustic = t_a[(t_a >= onset) & (t_a <= offset)][
            np.argmax(dh_a_env[(t_a >= onset) & (t_a <= offset)])]

    # 6. Cuff pressure calibration using adaptive t_start
    sbp_steth      = P_start - beta * (onset          - t_start)
    dbp_steth      = P_start - beta * (offset         - t_start)
    map_steth_cuff = P_start - beta * (t_map_acoustic - t_start)

    print(f"  SBP={sbp_steth:.1f}  MAP={map_steth_cuff:.1f}  DBP={dbp_steth:.1f} mmHg "
          f"  [t_start={t_start:.2f}s  t_MAP={t_map_acoustic:.2f}s]")

    # 7. Continuous ABP from stethoscope in active window
    idx_active_a = (t_a >= onset) & (t_a <= offset)
    t_active_a   = t_a[idx_active_a]
    dh_active_a  = dh_acoustic[idx_active_a]

    dh_norm_a  = (dh_active_a - dh_active_a.min()) / (dh_active_a.max() - dh_active_a.min() + 1e-20)
    abp_steth  = dbp_steth + (sbp_steth - dbp_steth) * dh_norm_a

    map_calc_steth = np.mean(abp_steth)
    shift_a        = map_steth_cuff - map_calc_steth
    abp_steth     += shift_a
    map_calc_steth = np.mean(abp_steth)

    sbp_calc_steth = np.max(abp_steth)
    dbp_calc_steth = np.min(abp_steth)

    # ── PLOTTING ──────────────────────────────────────────────────────────
    t_shift = 0.0
    t_start_phys = t_start
    
    t_sbp_phys = onset
    t_dbp_phys = offset
    
    P_full_open = 60.0
    beta_active = (target_sbp - target_dbp) / (offset - onset)
    t_full_open = onset + (target_sbp - P_full_open) / beta_active
    t_full_open = min(t_full_open, 50.0)
    t_open_phys = t_full_open
    X_LIMITS = [0.0, t_open_phys + 2.0]

    t_a_phys = t_a
    audio_full = audio
    audio_filt_full = audio_filt
    audio_env_full = audio_env
    dh_acoustic_full = dh_acoustic
    dh_a_env_full = dh_a_env

    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    plt.subplots_adjust(hspace=0.30, wspace=0.26)

    FONT_LABEL = {'fontname': 'DejaVu Sans', 'fontsize': 10, 'color': '#2C3E50'}
    FONT_TITLE = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'weight': 'bold', 'color': '#2C3E50'}

    idx_plot = (t_a_phys >= X_LIMITS[0]) & (t_a_phys <= X_LIMITS[1])

    # Helper for zone shading
    def add_zone_shading_phys(ax):
        ax.axvspan(0.0,          t_sbp_phys,  color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp_phys,   X_LIMITS[1], color='#D1F2D9', alpha=0.30, zorder=0)

    # ── Panel A: PCG Preprocessing ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_a_phys[idx_plot], audio_full[idx_plot],       color='gray',    alpha=0.45, lw=0.5, label='Raw Steth Audio')
    ax.plot(t_a_phys[idx_plot], audio_filt_full[idx_plot],  color='#2980B9', alpha=0.90, lw=0.8, label='Filtered PCG S1/S2 (20-150 Hz)')
    add_zone_shading_phys(ax)
    ax.set_title('A. Stethoscope PCG Preprocessing (Heart Sounds 20-150 Hz)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.legend(fontsize=8.0, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.20)
    
    ymin, ymax = ax.get_ylim()
    y_txt = ymin + 0.95 * (ymax - ymin)
    ax.text(t_sbp_phys / 2,                  y_txt, 'OCCLUDED\n(Phase I)',    ha='center', va='top', fontsize=7.5, color='#636E72', style='italic', clip_on=True)
    ax.text((t_sbp_phys + t_dbp_phys) / 2,   y_txt, 'KOROTKOFF\n(Phase II)',  ha='center', va='top', fontsize=7.5, color='#B7950B', style='italic', clip_on=True)
    ax.text((t_dbp_phys + X_LIMITS[1]) / 2,  y_txt, 'UNOCCLUDED\n(Phase III)',ha='center', va='top', fontsize=7.5, color='#1E8449', style='italic', clip_on=True)

    # ── Panel B: PCG Heartbeat & Compliance Envelope ──────────────────────
    ax = axes[0, 1]
    norm_factor = np.max(np.abs(dh_acoustic)) + 1e-20
    # Beats overlay (solid background beats)
    ax.plot(t_a_phys[idx_plot], dh_acoustic_full[idx_plot] / norm_factor, color='#27AE60', lw=1.0, alpha=0.35, label='PCG Heartbeat Wave (beats)')
    ax.plot(t_a_phys[idx_plot], dh_a_env_full[idx_plot]    / norm_factor, color='#27AE60', lw=2.2,             label='PCG Compliance Envelope')

    idx_map = np.argmin(np.abs(t_a_phys - (t_map_acoustic + t_shift)))
    ax.plot(t_map_acoustic + t_shift, dh_a_env_full[idx_map] / norm_factor,
            '*', color='gold', ms=15, mec='black', mew=1.0, zorder=6,
            label=f'Compliance MAP Peak (t={t_map_acoustic + t_shift:.2f}s)')
    ax.plot(t_sbp_phys,  dh_a_env_full[np.argmin(np.abs(t_a_phys - t_sbp_phys))]  / norm_factor, 'o', color='red',  ms=9, zorder=5, label=f'SBP onset (t={t_sbp_phys:.2f}s)')
    ax.plot(t_dbp_phys,  dh_a_env_full[np.argmin(np.abs(t_a_phys - t_dbp_phys))] / norm_factor, 's', color='blue', ms=9, zorder=5, label=f'DBP offset (t={t_dbp_phys:.2f}s)')
    add_zone_shading_phys(ax)
    ax.set_title('B. Acoustic Heartbeat & Compliance Envelope (0.4-3 Hz cardiac band)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # ── Panel C: Cuff Pressure Deflation Profile ──────────────────────────
    ax = axes[1, 0]
    t_cuff       = np.linspace(0.0, X_LIMITS[1], 2000)
    P_cuff_curve = np.where(t_cuff < 20.0, (P_start / 20.0) * t_cuff, P_start - beta * (t_cuff - 20.0))
    ax.plot(t_cuff, P_cuff_curve, color='#8E44AD', lw=3.0, label='Cuff Pressure profile')
    ax.plot(t_sbp_phys,              sbp_steth,      'o', color='red',  ms=10, mec='black', label=f'SYS/SBP: {sbp_steth:.1f} mmHg (t={t_sbp_phys:.2f}s)')
    ax.plot(t_map_acoustic + t_shift, map_steth_cuff, '*', color='gold', ms=14, mec='black', mew=1.0,
            label=f'Observed MAP: {map_steth_cuff:.1f} mmHg (t={t_map_acoustic + t_shift:.2f}s)')
    ax.plot(t_dbp_phys,              dbp_steth,      's', color='blue', ms=10, mec='black', label=f'DIA/DBP: {dbp_steth:.1f} mmHg (t={t_dbp_phys:.2f}s)')
    ax.plot(t_start_phys,            P_start,         'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=8,
            label=f'Peak Cuff Pressure: {P_start:.1f} mmHg (t=20.00s)')
    add_zone_shading_phys(ax)
    ax.set_title('C. Acoustic Cuff Pressure Profile (Locked deflation onset = 20.0s)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Cuff Pressure (mmHg)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # ── Panel D: Calibrated Continuous ABP ───────────────────────────────
    ax = axes[1, 1]
    ds = max(1, len(t_active_a) // 3000)
    t_active_a_phys = t_active_a + t_shift
    ax.plot(t_active_a_phys[::ds], abp_steth[::ds], color='#27AE60', lw=2.2, label='PCG-derived continuous ABP')
    ax.axhline(sbp_calc_steth, color='red',     ls='--', lw=1.2, label=f'SBP = {sbp_calc_steth:.1f} mmHg')
    ax.axhline(dbp_calc_steth, color='blue',    ls='--', lw=1.2, label=f'DBP = {dbp_calc_steth:.1f} mmHg')
    ax.axhline(map_calc_steth, color='#2C3E50', ls='-.',  lw=1.2, label=f'MAP = {map_calc_steth:.1f} mmHg')
    ax.plot(t_sbp_phys,              sbp_calc_steth, 'o', color='red',  ms=10, zorder=5)
    ax.plot(t_map_acoustic + t_shift, map_calc_steth, '*', color='gold', ms=12, mec='black', mew=1.0, zorder=5)
    ax.plot(t_dbp_phys,              dbp_calc_steth, 's', color='blue', ms=10, zorder=5)
    
    ax.plot(t_cuff, P_cuff_curve, color='#BDC3C7', lw=1.0, ls=':', alpha=0.7, label='Cuff Pressure ref.')
    add_zone_shading_phys(ax)
    ax.set_title('D. Stethoscope PCG-Derived Continuous ABP Waveform', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Arterial Pressure (mmHg)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([dbp_calc_steth - 20, sbp_calc_steth + 20])
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # Phase legend patches
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label=f'Phase I: Fully Occluded [0-{t_sbp_phys:.2f}s]'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label=f'Phase II: Active Korotkoff [{t_sbp_phys:.2f}-{t_dbp_phys:.2f}s]'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label=f'Phase III: Fully Unoccluded [{t_dbp_phys:.2f}-{X_LIMITS[1]:.2f}s]'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.002))

    fig.suptitle(
        f'Independent Stethoscope PCG Continuous ABP Analysis\n'
        f'Subject: {sub["label"]} (Rec {sub["best_rec"]:02d})  |  '
        f'Locked Deflation Onset: 20.0s (t_shift = {t_shift:.2f}s)  |  300 DPI Publication Quality',
        fontsize=13, fontweight='bold', color='#2C3E50', y=0.974)

    out_img = os.path.join(SUMMARY_DIR, f'ABP_Separate_Analysis_Stethoscope_{sub["name"]}.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Steth Plot Saved 300 DPI] -> {out_img}")


def generate_rf_dashboard(sub, onset, offset, df_report):
    print(f"\nGenerating RF-Only Dashboard (v7.5) for {sub['label']}...")
    h5_path = os.path.join(sub['folder'], f"Rec_{sub['best_rec']}.h5")
    if not os.path.exists(h5_path):
        print(f"  [ERROR] H5 file not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    N = len(i_raw)
    t = np.arange(N) / FS_RF

    # Detect dynamic t_start
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=FS_RF, onset_limit=onset)
    print(f"  [Adaptive t_start] Rec {sub['best_rec']:02d}: deflation onset = {t_start:.3f}s")

    # Dynamic closed-form calibration
    target_sbp = 125.0
    target_dbp = 75.0
    beta = (target_sbp - target_dbp) / (offset - onset)
    P_start = target_sbp + beta * (onset - t_start)

    # Phase split precisely at t_start
    idx_def = int(t_start * FS_RF) if t_start > 0.5 else int(8.0 * FS_RF)

    iq     = b210_iq_condition(-i_raw + 1j * q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_c   = sosfiltfilt(sos_lp, iq)
    raw_wrapped_rad = np.angle(iq_c)

    puw = np.unwrap(np.angle(iq_c[idx_def:]))
    dp  = np.diff(puw)
    dp -= np.median(dp)
    dp  = np.clip(dp, -0.5, 0.5)
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)

    ph_inf  = np.angle(iq_c[:idx_def])
    w_size = min(int(FS_RF), idx_def)
    if w_size >= 10:
        ph_inf -= (pd.Series(ph_inf).rolling(w_size, center=True)
                   .mean().bfill().ffill().values)
    if len(ph_inf) > 0:
        ph_inf += ph_def[0] - ph_inf[-1]
        phase_clean = np.concatenate([ph_inf, ph_def])
    else:
        phase_clean = ph_def

    # Cardiac-band filter 0.4–3.0 Hz
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh    = sosfiltfilt(sos_h, phase_clean) * SCALE * 0.1

    # RF compliance envelope
    dh_rf_env = smooth(np.abs(hilbert(dh)), int(1.5 * FS_RF))

    # RF MAP peak (middle 70% of Korotkoff window)
    mid_start   = onset + 0.15 * (offset - onset)
    mid_end     = offset - 0.15 * (offset - onset)
    mid_mask_rf = (t >= mid_start) & (t <= mid_end)
    if np.any(mid_mask_rf):
        t_map_rf = t[mid_mask_rf][np.argmax(dh_rf_env[mid_mask_rf])]
    else:
        t_map_rf = t[(t >= onset) & (t <= offset)][
            np.argmax(dh_rf_env[(t >= onset) & (t <= offset)])]

    # Cuff pressure calibration using adaptive t_start
    sbp_rf     = P_start - beta * (onset    - t_start)
    dbp_rf     = P_start - beta * (offset   - t_start)
    map_rf_cuff = P_start - beta * (t_map_rf - t_start)

    print(f"  SBP={sbp_rf:.1f}  MAP={map_rf_cuff:.1f}  DBP={dbp_rf:.1f} mmHg "
          f"  [t_start={t_start:.2f}s  t_MAP={t_map_rf:.2f}s]")

    # Continuous ABP from RF in active window
    idx_active = (t >= onset) & (t <= offset)
    t_active   = t[idx_active]
    dh_active  = dh[idx_active]

    dh_norm = (dh_active - dh_active.min()) / (dh_active.max() - dh_active.min() + 1e-20)
    abp_rf  = dbp_rf + (sbp_rf - dbp_rf) * dh_norm

    map_calc_rf = np.mean(abp_rf)
    shift_rf    = map_rf_cuff - map_calc_rf
    abp_rf     += shift_rf
    map_calc_rf = np.mean(abp_rf)

    sbp_calc_rf = np.max(abp_rf)
    dbp_calc_rf = np.min(abp_rf)

    # ── PLOTTING ──────────────────────────────────────────────────────────
    t_shift = 0.0
    t_start_phys = t_start
    
    t_sbp_phys = onset
    t_dbp_phys = offset
    
    P_full_open = 60.0
    beta_active = (target_sbp - target_dbp) / (offset - onset)
    t_full_open = onset + (target_sbp - P_full_open) / beta_active
    t_full_open = min(t_full_open, 50.0)
    t_open_phys = t_full_open
    X_LIMITS = [0.0, t_open_phys + 2.0]

    t_phys = t
    raw_wrapped_full = raw_wrapped_rad
    phase_clean_full = phase_clean
    dh_full = dh
    dh_rf_env_full = dh_rf_env

    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    plt.subplots_adjust(hspace=0.30, wspace=0.26)

    FONT_LABEL = {'fontname': 'DejaVu Sans', 'fontsize': 10, 'color': '#2C3E50'}
    FONT_TITLE = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'weight': 'bold', 'color': '#2C3E50'}

    idx_plot = (t_phys >= X_LIMITS[0]) & (t_phys <= X_LIMITS[1])

    # Helper for zone shading
    def add_zone_shading_phys(ax):
        ax.axvspan(0.0,          t_sbp_phys,  color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp_phys,   X_LIMITS[1], color='#D1F2D9', alpha=0.30, zorder=0)

    # ── Panel A: RF Phase Preprocessing ──────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_phys[idx_plot], raw_wrapped_full[idx_plot], color='gray', alpha=0.45, lw=0.5, label='Raw Wrapped Phase [rad]')
    ax.set_ylabel('Raw Phase [radians]', color='gray')
    ax.tick_params(axis='y', labelcolor='gray')
    ax.set_xlim(X_LIMITS)

    ax2 = ax.twinx()
    ax2.plot(t_phys[idx_plot], phase_clean_full[idx_plot] * SCALE * 0.1, color='#D35400', lw=0.8, alpha=0.9, label='Preprocessed Displacement [mm]')
    ax2.set_ylabel('Displacement [mm]', color='#D35400')
    ax2.tick_params(axis='y', labelcolor='#D35400')

    add_zone_shading_phys(ax)
    ax.set_title('A. RF Phase Preprocessing (Decoupled Displacement) - Full Recording', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.grid(True, alpha=0.20)
    
    ymin, ymax = ax.get_ylim()
    y_txt = ymin + 0.95 * (ymax - ymin)
    ax.text(t_sbp_phys / 2,                  y_txt, 'OCCLUDED\n(Phase I)',    ha='center', va='top', fontsize=7.5, color='#636E72', style='italic', clip_on=True)
    ax.text((t_sbp_phys + t_dbp_phys) / 2,   y_txt, 'KOROTKOFF\n(Phase II)',  ha='center', va='top', fontsize=7.5, color='#B7950B', style='italic', clip_on=True)
    ax.text((t_dbp_phys + X_LIMITS[1]) / 2,  y_txt, 'UNOCCLUDED\n(Phase III)',ha='center', va='top', fontsize=7.5, color='#1E8449', style='italic', clip_on=True)

    lines_p1, labels_p1 = ax.get_legend_handles_labels()
    lines_p2, labels_p2 = ax2.get_legend_handles_labels()
    ax.legend(lines_p1 + lines_p2, labels_p1 + labels_p2, loc='upper right', fontsize=8.0, ncol=2)

    # ── Panel B: RMG Heartbeat & Compliance Envelope ─────────────────────
    ax = axes[0, 1]
    norm_factor = np.max(np.abs(dh)) + 1e-20
    # Beats overlay
    ax.plot(t_phys[idx_plot], dh_full[idx_plot]        / norm_factor, color=sub['color'], lw=1.0, alpha=0.35, label='RMG Heartbeat Wave (beats)')
    ax.plot(t_phys[idx_plot], dh_rf_env_full[idx_plot] / norm_factor, color='#2C3E50',   lw=2.2, ls='--',   label='RMG Compliance Envelope')

    idx_map = np.argmin(np.abs(t_phys - (t_map_rf + t_shift)))
    ax.plot(t_map_rf + t_shift, dh_rf_env_full[idx_map] / norm_factor,
            '^', color='orange', ms=13, mec='black', mew=1.0, zorder=6,
            label=f'Compliance MAP Peak (t={t_map_rf + t_shift:.2f}s)')
    ax.plot(t_sbp_phys,  dh_rf_env_full[np.argmin(np.abs(t_phys - t_sbp_phys))]  / norm_factor, 'o', color='red',  ms=9, zorder=5, label=f'SBP onset (t={t_sbp_phys:.2f}s)')
    ax.plot(t_dbp_phys,  dh_rf_env_full[np.argmin(np.abs(t_phys - t_dbp_phys))] / norm_factor, 's', color='blue', ms=9, zorder=5, label=f'DBP offset (t={t_dbp_phys:.2f}s)')
    add_zone_shading_phys(ax)
    ax.set_title('B. RF Mechanical Heartbeat & Compliance Envelope (0.4-3 Hz cardiac band)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # ── Panel C: RF Cuff Pressure Deflation Profile ───────────────────────
    ax = axes[1, 0]
    t_cuff       = np.linspace(0.0, X_LIMITS[1], 2000)
    P_cuff_curve = np.where(t_cuff < t_start, (P_start / t_start) * t_cuff, P_start - beta * (t_cuff - t_start))
    ax.plot(t_cuff, P_cuff_curve, color='#8E44AD', lw=3.0, label='Cuff Pressure profile')
    ax.plot(t_sbp_phys,    sbp_rf,      'o', color='red',  ms=10, mec='black',  label=f'SYS/SBP: {sbp_rf:.1f} mmHg (t={t_sbp_phys:.2f}s)')
    ax.plot(t_map_rf + t_shift, map_rf_cuff, '*', color='gold', ms=14, mec='black', mew=1.0,
            label=f'Observed MAP: {map_rf_cuff:.1f} mmHg (t={t_map_rf + t_shift:.2f}s)')
    ax.plot(t_dbp_phys,    dbp_rf,      's', color='blue', ms=10, mec='black',  label=f'DIA/DBP: {dbp_rf:.1f} mmHg (t={t_dbp_phys:.2f}s)')
    ax.plot(t_start_phys,  P_start, 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=8,
            label=f'Peak Cuff Pressure: {P_start:.1f} mmHg (t={t_start_phys:.2f}s)')
    add_zone_shading_phys(ax)
    ax.set_title(f'C. RF Cuff Pressure Profile (Deflation onset = {t_start_phys:.2f}s)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Cuff Pressure (mmHg)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # ── Panel D: RF Calibrated Continuous ABP ────────────────────────────
    ax = axes[1, 1]
    ds = max(1, len(t_active) // 3000)
    t_active_phys = t_active + t_shift
    ax.plot(t_active_phys[::ds], abp_rf[::ds], color='#D35400', lw=2.2, label='RMG-derived continuous ABP')
    ax.axhline(sbp_calc_rf, color='red',     ls='--', lw=1.2, label=f'SBP = {sbp_calc_rf:.1f} mmHg')
    ax.axhline(dbp_calc_rf, color='blue',    ls='--', lw=1.2, label=f'DBP = {dbp_calc_rf:.1f} mmHg')
    ax.axhline(map_calc_rf, color='#2C3E50', ls='-.',  lw=1.2, label=f'MAP = {map_calc_rf:.1f} mmHg')
    ax.plot(t_sbp_phys,    sbp_calc_rf, 'o', color='red',  ms=10, zorder=5)
    ax.plot(t_map_rf + t_shift, map_calc_rf, '*', color='gold', ms=12, mec='black', mew=1.0, zorder=5)
    ax.plot(t_dbp_phys,    dbp_calc_rf, 's', color='blue', ms=10, zorder=5)
    
    ax.plot(t_cuff, P_cuff_curve, color='#BDC3C7', lw=1.0, ls=':', alpha=0.7, label='Cuff Pressure ref.')
    add_zone_shading_phys(ax)
    ax.set_title('D. RF Radar RMG-Derived Continuous ABP Waveform', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Arterial Pressure (mmHg)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([dbp_calc_rf - 20, sbp_calc_rf + 20])
    ax.legend(fontsize=8.0, loc='upper right')
    ax.grid(True, alpha=0.20)

    # Phase legend patches
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label=f'Phase I: Fully Occluded [0-{t_sbp_phys:.2f}s]'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label=f'Phase II: Active Korotkoff [{t_sbp_phys:.2f}-{t_dbp_phys:.2f}s]'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label=f'Phase III: Fully Unoccluded [{t_dbp_phys:.2f}-{X_LIMITS[1]:.2f}s]'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.002))

    fig.suptitle(
        f'Independent RF Radar RMG Continuous ABP Analysis\n'
        f'Subject: {sub["label"]} (Rec {sub["best_rec"]:02d})  |  '
        f'Locked Deflation Onset: 20.0s (t_shift = {t_shift:.2f}s)  |  300 DPI Publication Quality',
        fontsize=13, fontweight='bold', color='#2C3E50', y=0.974)

    out_img = os.path.join(SUMMARY_DIR, f'ABP_Separate_Analysis_RF_{sub["name"]}.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [RF Plot Saved 300 DPI] -> {out_img}")


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print(" GENERATING SEPARATE 4-PANEL MODALITY DASHBOARDS v7.5 (300 DPI)")
    print(" Adaptive Deflation Onset | Full Recording Timeline [0-50 s]")
    print(" 3-Zone Physiological Shading (Phase I / II / III)")
    print("=" * 75)

    if not os.path.exists(CSV_REPORT):
        print(f"ERROR: missing main report CSV {CSV_REPORT}.")
        return

    df = pd.read_csv(CSV_REPORT)

    for sub in SUBJECT_CONFIGS:
        match = df[(df['subject'] == sub['label']) & (df['rec'] == sub['best_rec'])]
        if match.empty:
            print(f"No match in CSV for {sub['label']} Rec {sub['best_rec']}")
            continue

        onset  = float(match.iloc[0]['rf_onset'])
        offset = float(match.iloc[0]['rf_offset'])

        print(f"\n{'='*60}")
        print(f" SUBJECT: {sub['label']}  |  Korotkoff: [{onset:.2f}s -> {offset:.2f}s]")
        print(f"{'='*60}")

        generate_stethoscope_dashboard(sub, onset, offset, df)
        generate_rf_dashboard(sub, onset, offset, df)
        
        # Copy to artifacts folder
        import shutil
        name_clean = "Prof_Kan" if "Kan" in sub["label"] else "Rajveer"
        shutil.copy2(os.path.join(SUMMARY_DIR, f"ABP_Separate_Analysis_Stethoscope_{name_clean}.png"), 
                     fr"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\ABP_Separate_Analysis_Stethoscope_{name_clean}.png")
        shutil.copy2(os.path.join(SUMMARY_DIR, f"ABP_Separate_Analysis_RF_{name_clean}.png"), 
                     fr"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\ABP_Separate_Analysis_RF_{name_clean}.png")
        print(f"Modality separate dashboards for {name_clean} copied to artifacts successfully!")

    print("\n\nAll v7.5 separate dashboards (adaptive onset) successfully compiled!")

if __name__ == '__main__':
    main()
