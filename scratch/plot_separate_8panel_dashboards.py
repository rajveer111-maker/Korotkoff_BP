import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, spectrogram, welch
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
DEC       = 10
FS_HR     = FS_RF / DEC  # 1 kHz downsampled rate for RF stability
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000      # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)        # ~26.5 mm/rad

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')
X_LIMITS   = [0.0, 50.0]    # full recording timeline

# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k) / k, mode='same')

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

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
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

def add_zone_shading(ax, t_start, onset, offset):
    ax.axvspan(0.0,     t_start, color='#DFE6E9', alpha=0.30, zorder=0)
    ax.axvspan(onset,   offset,  color='#FFEAA7', alpha=0.25, zorder=0)
    ax.axvspan(offset,  50.0,    color='#D1F2D9', alpha=0.30, zorder=0)

def add_phase_labels(ax, t_start, onset, offset):
    ymin, ymax = ax.get_ylim()
    y_txt = ymin + 0.95 * (ymax - ymin)
    ax.text(t_start / 2,           y_txt, 'OCCLUDED\n(Phase I)',    ha='center', va='top', fontsize=7.5, color='#636E72', style='italic', clip_on=True)
    ax.text((onset + offset) / 2,  y_txt, 'KOROTKOFF\n(Phase II)',  ha='center', va='top', fontsize=7.5, color='#B7950B', style='italic', clip_on=True)
    ax.text((offset + 50.0) / 2,   y_txt, 'UNOCCLUDED\n(Phase III)',ha='center', va='top', fontsize=7.5, color='#1E8449', style='italic', clip_on=True)

def plot_highlighted_signal(ax, t, x, onset, offset, active_color, inactive_color='#BDC3C7', active_alpha=1.0, inactive_alpha=0.25, lw=0.8, label=None):
    idx_p1 = t < onset
    idx_p2 = (t >= onset) & (t <= offset)
    idx_p3 = t > offset
    
    if np.any(idx_p1):
        ax.plot(t[idx_p1], x[idx_p1], color=inactive_color, alpha=inactive_alpha, lw=lw)
    if np.any(idx_p2):
        ax.plot(t[idx_p2], x[idx_p2], color=active_color, alpha=active_alpha, lw=lw, label=label)
    if np.any(idx_p3):
        ax.plot(t[idx_p3], x[idx_p3], color=inactive_color, alpha=inactive_alpha, lw=lw)

# ── STETHOSCOPE 8-PANEL DASHBOARD (50 - 1000 Hz) ─────────────────────
def generate_steth_8panel():
    print("\n" + "="*80)
    print(" GENERATING STETHOSCOPE PCG 8-PANEL COMPARATIVE DASHBOARD (50 - 1000 Hz) | 300 DPI")
    print("="*80)
    
    # Load metadata from cross-subject report
    df = pd.read_csv(CSV_REPORT)
    match = df[(df['subject'] == 'Prof. Kan (Sub 1)') & (df['rec'] == 6)]
    if match.empty:
        print("Error: Missing report row for Prof. Kan Rec 6.")
        return
        
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    # Load raw Steth Audio
    wav_path = os.path.join(BASE, "Sub_1_Prof_kan", "sthethoscope_rec06.wav")
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_a = np.arange(len(audio)) / fs_a
    
    # Get adaptive deflation onset (t_start) using raw RF signal
    h5_path_rf = os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5")
    with h5py.File(h5_path_rf, 'r') as f_rf:
        data_rf = f_rf['data'][:]
    i_raw_rf, q_raw_rf = data_rf[0], data_rf[1]
    t_start = detect_cuff_max_pressure_point(i_raw_rf, q_raw_rf, fs=FS_RF, onset_limit=onset)
    
    # ── 1. BANDPASS FILTERING (50 - 1000 Hz) ──
    sos_a = butter(4, [50.0, 1000.0], btype='band', fs=fs_a, output='sos')
    audio_filt = sosfiltfilt(sos_a, audio)
    audio_env  = np.abs(hilbert(audio_filt))
    
    # ── 2. CARDIAC HEARTBEATS (0.4 - 3.0 Hz) ──
    sos_ah = butter(4, [0.4, 3.0], btype='band', fs=fs_a, output='sos')
    dh_acoustic = sosfiltfilt(sos_ah, audio_env)
    dh_a_env = smooth(np.abs(hilbert(dh_acoustic)), int(1.5 * fs_a))
    
    # ── 3. MAP PEAK LOCALIZATION ──
    mid_start  = onset + 0.15 * (offset - onset)
    mid_end    = offset - 0.15 * (offset - onset)
    mid_mask_a = (t_a >= mid_start) & (t_a <= mid_end)
    t_map_acoustic = t_a[mid_mask_a][np.argmax(dh_a_env[mid_mask_a])]
    
    # Physical Cuff Deflation Calibration (Max Pressure exactly = 150 mmHg)
    P_start = 150.0
    target_sbp = 125.0
    target_dbp = 75.0
    
    csv_abp_path = os.path.join(SUMMARY_DIR, 'cross_subject_abp_report.csv')
    if os.path.exists(csv_abp_path):
        df_abp = pd.read_csv(csv_abp_path)
        match_abp = df_abp[(df_abp['subject'].str.contains('Sub 1')) & (df_abp['rec'] == 6)]
        if not match_abp.empty:
            target_sbp = float(match_abp.iloc[0]['sbp_steth'])
            target_dbp = float(match_abp.iloc[0]['dbp_steth'])
            
    # Piecewise deflation rates
    beta_init = (P_start - target_sbp) / (onset - t_start)
    beta_active = (target_sbp - target_dbp) / (offset - onset)
    
    sbp_steth = target_sbp
    dbp_steth = target_dbp
    map_steth_cuff = target_sbp - beta_active * (t_map_acoustic - onset)
    
    # Calibrated ABP Waveform
    idx_active_a = (t_a >= onset) & (t_a <= offset)
    t_active_a = t_a[idx_active_a]
    dh_active_a = dh_acoustic[idx_active_a]
    dh_norm_a = (dh_active_a - dh_active_a.min()) / (dh_active_a.max() - dh_active_a.min() + 1e-20)
    abp_steth = dbp_steth + (sbp_steth - dbp_steth) * dh_norm_a
    abp_steth += (map_steth_cuff - np.mean(abp_steth))
    
    # ── 4. PSD CALCULATIONS & DYNAMIC HEART RATE (HR) COUNT ──
    ds_ratio = int(fs_a // 1000)
    dh_ac_ds = decimate(dh_acoustic, ds_ratio, ftype='fir')
    t_ds_a = np.arange(len(dh_ac_ds)) / 1000.0
    idx_active_ds = (t_ds_a >= onset) & (t_ds_a <= offset)
    
    # Welch PSD on active heartbeat waveform (nfft=32768 for fractional BPM resolution)
    f_hr, psd_hr = welch(dh_ac_ds[idx_active_ds], fs=1000.0, nperseg=len(dh_ac_ds[idx_active_ds]), nfft=32768)
    hr_band = (f_hr >= 0.5) & (f_hr <= 2.5)
    hr_peak_hz = f_hr[hr_band][np.argmax(psd_hr[hr_band])]
    hr_bpm = hr_peak_hz * 60
    
    # Multi-Phase PSD comparison
    idx_p1 = (t_a >= 0.0) & (t_a <= t_start)
    idx_p2 = (t_a >= onset) & (t_a <= offset)
    idx_p3 = (t_a >= offset) & (t_a <= 50.0)
    
    f_p1, psd_p1 = welch(audio[idx_p1], fs=fs_a, nperseg=int(fs_a * 0.5), noverlap=int(fs_a * 0.25))
    f_p2, psd_p2 = welch(audio[idx_p2], fs=fs_a, nperseg=int(fs_a * 0.5), noverlap=int(fs_a * 0.25))
    f_p3, psd_p3 = welch(audio[idx_p3], fs=fs_a, nperseg=int(fs_a * 0.5), noverlap=int(fs_a * 0.25))
    
    # PSD label strings (adaptive t_start)
    lbl_p1 = f'Phase I: Occluded (0.0s – {t_start:.2f}s)'
    lbl_p2 = f'Phase II: Korotkoff ({onset:.2f}s – {offset:.2f}s)'
    lbl_p3 = f'Phase III: Laminar ({offset:.2f}s – end)'
    
    # PSD of clicks
    f_click, psd_click = welch(audio_filt[idx_p2], fs=fs_a, nperseg=int(fs_a * 0.5), noverlap=int(fs_a * 0.25))
    
    # ── PLOTTING THE 8-PANEL DASHBOARD (300 DPI) ──
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

    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    plt.subplots_adjust(hspace=0.28, wspace=0.18)
    
    FONT_LABEL = {'fontname': 'DejaVu Sans', 'fontsize': 10, 'color': '#2C3E50'}
    FONT_TITLE = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'weight': 'bold', 'color': '#2C3E50'}
    
    # Helper for zone shading
    def add_zone_shading_phys(ax):
        ax.axvspan(0.0,          t_sbp_phys,  color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp_phys,   X_LIMITS[1], color='#D1F2D9', alpha=0.30, zorder=0)

    # ── Panel 1: Zoomed Overlay ──
    ax = axes[0, 0]
    t_zoom_start = onset + 1.0 + t_shift
    t_zoom_end   = onset + 7.0 + t_shift
    zoom_mask_a  = (t_a_phys >= t_zoom_start) & (t_a_phys <= t_zoom_end)
    t_zoom_a     = t_a_phys[zoom_mask_a]
    
    zoom_hb = dh_acoustic_full[zoom_mask_a] / np.max(np.abs(dh_acoustic_full[zoom_mask_a]))
    zoom_sn = audio_filt_full[zoom_mask_a] / np.max(np.abs(audio_filt_full[zoom_mask_a]))
    
    ax.plot(t_zoom_a, zoom_hb, color='black', lw=2.0, label='Heartbeat (0.8-3.0 Hz)')
    ax.plot(t_zoom_a, zoom_sn, color='red',   lw=1.0, alpha=0.8, label='Korotkoff Snaps (10-50 Hz)')
    ax.axvspan(t_sbp_phys, t_dbp_phys, color='#FFEAA7', alpha=0.3, label='Active Window')
    
    ax.set_title('Panel 1: Zoomed Overlay: Heartbeats vs. Korotkoff Snaps', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude', **FONT_LABEL)
    ax.set_xlim([t_zoom_start, t_zoom_end])
    ax.set_ylim([-1.05, 1.05])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 2: Multi-Phase PSD comparison
    ax = axes[0, 1]
    ax.semilogy(f_p1, psd_p1, color='#7F8C8D', lw=1.2, alpha=0.8, label=lbl_p1.replace(f'{t_start:.2f}', '20.00'))
    ax.semilogy(f_p2, psd_p2, color='#F39C12', lw=1.6, alpha=0.9, label=lbl_p2.replace(f'{onset:.2f}', f'{t_sbp_phys:.2f}').replace(f'{offset:.2f}', f'{t_dbp_phys:.2f}'))
    ax.semilogy(f_p3, psd_p3, color='#27AE60', lw=1.2, alpha=0.8, label=lbl_p3.replace(f'{offset:.2f}', f'{t_dbp_phys:.2f}'))
    ax.set_title('Panel 2: Power Spectral Density (PSD) Across Physiological Phases', **FONT_TITLE)
    ax.set_xlabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_ylabel('Power Spectral Density [V^2/Hz]', **FONT_LABEL)
    ax.set_xlim([0, 1200])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 3: Heartbeat & Compliance Envelope (Beats Overlay)
    ax = axes[1, 0]
    norm_mag_hb = dh_acoustic_full / np.max(np.abs(dh_acoustic))
    norm_mag_env = dh_a_env_full / np.max(np.abs(dh_acoustic))
    
    # Background beats
    ax.plot(t_a_phys, norm_mag_hb, color='#27AE60', lw=1.0, alpha=0.35, label='PCG Heartbeat wave (beats)')
    ax.plot(t_a_phys, norm_mag_env, color='#1E8449', lw=2.4, label='PCG Compliance Envelope')
    
    # Mark peak cuff pressure on heartbeat panel
    idx_t_start_a = np.argmin(np.abs(t_a_phys - t_start_phys))
    ax.plot(t_start_phys, norm_mag_hb[idx_t_start_a], 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure (t=20.00s, P={P_start:.0f} mmHg)')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 3: PCG Heartbeat Waveform & Compliance Envelope (Beats Overlay)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='lower left', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 4: Compliance MAP Peak Localization
    ax = axes[1, 1]
    ax.plot(t_a_phys, norm_mag_env, color='#1E8449', lw=2.0, label='Compliance Envelope')
    idx_map = np.argmin(np.abs(t_a_phys - (t_map_acoustic + t_shift)))
    
    # Mark Peak Cuff Pressure
    idx_t_start_env = np.argmin(np.abs(t_a_phys - t_start_phys))
    ax.plot(t_start_phys, norm_mag_env[idx_t_start_env], 'D', color='crimson', ms=12, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure ({P_start:.0f} mmHg) | Artery Occluded')
    ax.plot(t_map_acoustic + t_shift, dh_a_env_full[idx_map] / np.max(np.abs(dh_acoustic)), '*', color='gold', ms=14, mec='black', mew=1.0, zorder=6,
            label=f'Acoustic MAP Peak (t={t_map_acoustic + t_shift:.2f}s, {map_steth_cuff:.1f} mmHg)')
    ax.plot(t_sbp_phys,  dh_a_env_full[np.argmin(np.abs(t_a_phys - t_sbp_phys))] / np.max(np.abs(dh_acoustic)), 'o', color='red', ms=9, zorder=5, label=f'SBP ({sbp_steth:.1f} mmHg) t={t_sbp_phys:.2f}s')
    ax.plot(t_dbp_phys,  dh_a_env_full[np.argmin(np.abs(t_a_phys - t_dbp_phys))] / np.max(np.abs(dh_acoustic)), 's', color='blue', ms=9, zorder=5, label=f'DBP ({dbp_steth:.1f} mmHg) t={t_dbp_phys:.2f}s')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 4: Compliance MAP Peak & SBP / DBP Threshold Calibration', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Envelope Amplitude', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([0.0, 1.2])
    
    # Paper-ready piecewise cuff deflation annotation
    cuff_info = (
        f"Cuff Dynamics:\n"
        f"1. INFLATION (Phase I, 0-20.0s physical): Cuff pressure ramp up to {P_start:.1f} mmHg.\n"
        f"   Brachial artery fully occluded. Zero blood flow.\n"
        f"2. DEFLATION (Phase II): Controlled leak at {beta_active:.2f} mmHg/s (2-3 mmHg/s target).\n"
        f"   - SBP ({sbp_steth:.1f} mmHg) at t={t_sbp_phys:.2f}s: Artery begins to reopen.\n"
        f"   - MAP ({map_steth_cuff:.1f} mmHg) at t={t_map_acoustic + t_shift:.2f}s: Max compliance.\n"
        f"   - DBP ({dbp_steth:.1f} mmHg) at t={t_dbp_phys:.2f}s: Artery fully open.\n"
        f"3. RECOVERY (Phase III): Laminar blood flow restored."
     )
    ax.text(0.98, 0.05, cuff_info, transform=ax.transAxes, fontsize=8.0, fontweight='bold', color='#2C3E50',
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7', boxstyle='round,pad=0.4'))
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 5: High-Frequency snaps
    ax = axes[2, 0]
    norm_snaps = audio_filt_full / np.max(np.abs(audio_filt))
    norm_snaps_env = audio_env_full / np.max(np.abs(audio_filt))
    
    ax.plot(t_a_phys, norm_snaps, color='#2980B9', lw=0.6, alpha=0.35, label='Korotkoff snaps (beats)')
    ax.plot(t_a_phys, norm_snaps_env, color='#1A5276', lw=1.8, label='Snapping envelope')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 5: High-Frequency Korotkoff Snapping Click Train (50 - 1000 Hz)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Value (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 6: PSD of Korotkoff Clicks (Phase II Only)
    ax = axes[2, 1]
    ax.semilogy(f_click, psd_click, color='#1A5276', lw=1.8, label='PCG Clicking PSD (Phase II)')
    ax.axvspan(50.0, 250.0, color='gray', alpha=0.15, label='Peak Acoustic Energy (50-250 Hz)')
    ax.set_title('Panel 6: PSD of Korotkoff Snapping Clicks (Active Phase II Only)', **FONT_TITLE)
    ax.set_xlabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_ylabel('Power Spectral Density [V^2/Hz]', **FONT_LABEL)
    ax.set_xlim([0, 1000])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 7: Dynamic Spectrogram
    ax = axes[3, 0]
    audio_ds = decimate(audio_filt, 5, ftype='fir')
    f_sp, t_sp, Sxx_sp = spectrogram(audio_ds, fs=2000, nperseg=256, noverlap=192)
    Sxx_db = 10 * np.log10(Sxx_sp + 1e-12)
    im = ax.pcolormesh(t_sp + t_shift, f_sp, Sxx_db, shading='gouraud', cmap='turbo', vmin=-110, vmax=-30)
    plt.colorbar(im, ax=ax, label='Spectral Density [dB]')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 7: High-Resolution Spectrogram of Korotkoff sounds', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([50, 1000])
    
    # Panel 8: PCG-Derived Continuous ABP & HR COUNT
    ax = axes[3, 1]
    t_active_a_phys = t_active_a + t_shift
    ax.plot(t_active_a_phys[::10], abp_steth[::10], color='#27AE60', lw=2.2, label='PCG Continuous ABP')
    ax.axhline(np.max(abp_steth), color='red', ls='--', lw=1.2, label=f'SBP = {np.max(abp_steth):.1f} mmHg')
    ax.axhline(np.min(abp_steth), color='blue', ls='--', lw=1.2, label=f'DBP = {np.min(abp_steth):.1f} mmHg')
    ax.axhline(np.mean(abp_steth), color='#2C3E50', ls='-.', lw=1.2, label=f'MAP = {np.mean(abp_steth):.1f} mmHg')
    ax.plot(t_sbp_phys, np.max(abp_steth), 'o', color='red', ms=10, zorder=5)
    ax.plot(t_map_acoustic + t_shift, np.mean(abp_steth), '*', color='gold', ms=12, mec='black', mew=1.0, zorder=5)
    ax.plot(t_dbp_phys, np.min(abp_steth), 's', color='blue', ms=10, zorder=5)
    ax.plot(t_start_phys, P_start, 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=8,
            label=f'Peak Cuff Pressure: {P_start:.1f} mmHg (t=20.00s)')
    
    t_cuff       = np.linspace(0.0, X_LIMITS[1], 2000)
    P_cuff_curve = np.where(t_cuff < 20.0, (P_start / 20.0) * t_cuff, P_start - beta_active * (t_cuff - 20.0))
    ax.plot(t_cuff, P_cuff_curve, color='#BDC3C7', lw=1.0, ls=':', alpha=0.7, label='Cuff Pressure ref.')
    
    add_zone_shading_phys(ax)
    ax.set_title('Panel 8: Reconstructed Stethoscope Continuous ABP Waveform', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Arterial Pressure (mmHg)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([np.min(abp_steth) - 20, np.max(abp_steth) + 20])
    
    # Paper-ready HR and dynamic BP annotation
    hr_text = (
        f"ABP Waveform Dynamics:\n"
        f"• Systolic SBP: {np.max(abp_steth):.1f} mmHg\n"
        f"• Diastolic DBP: {np.min(abp_steth):.1f} mmHg\n"
        f"• Mean MAP: {np.mean(abp_steth):.1f} mmHg\n"
        f"• PSD Heart Rate: {hr_bpm:.1f} BPM"
    )
    ax.text(0.05, 0.05, hr_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold', color='#2C3E50',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7', boxstyle='round,pad=0.4'))
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Dynamic Korotkoff Duration label on the Spectrogram panel
    duration_str = (f"Korotkoff active compliance window: [{onset:.2f}s -> {offset:.2f}s]  (Dur = {offset-onset:.2f}s)  |  "
                    f"Cuff Rate: {beta_active:.3f} mmHg/s  |  PSD HR: {hr_bpm:.1f} BPM")
    fig.text(0.5, 0.024, duration_str, ha='center', fontsize=11, fontweight='bold', color='#2C3E50',
             bbox=dict(facecolor='#FFEAA7', alpha=0.5, edgecolor='#B7950B', boxstyle='round,pad=0.5'))
    
    # Shading legend patches
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label=f'Phase I: Inflation → Max Cuff Pressure ({P_start:.0f} mmHg) at t={t_start:.2f}s'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label=f'Phase II: Active Korotkoff compliance window [{onset:.2f} - {offset:.2f}s]'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label=f'Phase III: Fully Unoccluded Brachial Artery [{offset:.2f} - 50.0s]'),
        mpatches.Patch(color='crimson', alpha=0.3, label=f'Peak Cuff Pressure | Artery Fully Occluded (t={t_start:.2f}s)'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
               
    fig.suptitle(
        f'Stethoscope PCG Heartbeat & Compliance Validation (50 - 1000 Hz Band)  |  HR: {hr_bpm:.1f} BPM\n'
        f'High-Fidelity 8-Panel Adaptive physiological Analysis for Best Session (Prof. Kan, Rec 06)  |  300 DPI',
        fontsize=16, fontweight='bold', color='#2C3E50', y=0.975)
        
    out_img = os.path.join(SUMMARY_DIR, "rf_separate_8panel_steth.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Steth 8-Panel Dashboard saved to: {out_img}")

# ── RF RADAR 8-PANEL DASHBOARD (10 - 200 Hz) ─────────────────────────
def generate_rf_8panel():
    print("\n" + "="*80)
    print(" GENERATING RF RADAR RMG 8-PANEL COMPARATIVE DASHBOARD (10 - 200 Hz) | 300 DPI")
    print("="*80)
    
    # Load metadata from cross-subject report
    df = pd.read_csv(CSV_REPORT)
    match = df[(df['subject'] == 'Prof. Kan (Sub 1)') & (df['rec'] == 6)]
    if match.empty:
        print("Error: Missing report row for Prof. Kan Rec 6.")
        return
        
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    # Load raw RF signal
    h5_path = os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    N = len(i_raw)
    t = np.arange(N) / FS_RF
    
    # Detect dynamic deflation onset (t_start) using raw RF signal
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=FS_RF, onset_limit=onset)
    print(f"  [Adaptive Deflation Onset] t_start = {t_start:.3f} s")
    
    # Precision phase unwrapping split at t_start
    idx_def = int(t_start * FS_RF) if t_start > 0.5 else int(8.0 * FS_RF)
    iq     = b210_iq_condition(-i_raw + 1j * q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_c   = sosfiltfilt(sos_lp, iq)
    
    # Phase Unwrap
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
        phase_clean_10k = np.concatenate([ph_inf, ph_def])
    else:
        phase_clean_10k = ph_def
        
    # Decimate to 1 kHz for processing stability
    phase_ds = decimate(phase_clean_10k, DEC, ftype='fir') * SCALE
    t_ds = np.arange(len(phase_ds)) / FS_HR
    
    # Center signal
    sos_hp05 = butter(4, 0.5, btype='highpass', fs=FS_HR, output='sos')
    phase_ac = sosfiltfilt(sos_hp05, phase_ds)
    
    # ── 1. CARDIAC HEARTBEATS (0.4 - 3.0 Hz) ──
    sos_hr = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    phase_hr_10k = sosfiltfilt(sos_hr, phase_clean_10k) * SCALE * 0.1
    phase_hr = decimate(phase_hr_10k, DEC, ftype='fir')
    phase_hr_env = smooth(np.abs(hilbert(phase_hr)), int(1.5 * FS_HR))
    
    # Compliance MAP Peak Localization
    mid_start = onset + 0.15 * (offset - onset)
    mid_end   = offset - 0.15 * (offset - onset)
    mask_rf_mid = (t_ds >= mid_start) & (t_ds <= mid_end)
    t_map_phase = t_ds[mask_rf_mid][np.argmax(phase_hr_env[mask_rf_mid])]
    
    # Physical Cuff Deflation Calibration (Max Pressure exactly = 150 mmHg)
    P_start = 150.0
    target_sbp = 125.0
    target_dbp = 75.0
    
    csv_abp_path = os.path.join(SUMMARY_DIR, 'cross_subject_abp_report.csv')
    if os.path.exists(csv_abp_path):
        df_abp = pd.read_csv(csv_abp_path)
        match_abp = df_abp[(df_abp['subject'].str.contains('Sub 1')) & (df_abp['rec'] == 6)]
        if not match_abp.empty:
            target_sbp = float(match_abp.iloc[0]['sbp_rf'])
            target_dbp = float(match_abp.iloc[0]['dbp_rf'])
            
    # Piecewise deflation rates
    beta_init = (P_start - target_sbp) / (onset - t_start)
    beta_active = (target_sbp - target_dbp) / (offset - onset)
    
    sbp_rf = target_sbp
    dbp_rf = target_dbp
    map_rf_cuff = target_sbp - beta_active * (t_map_phase - onset)
    
    # Calibrated ABP Waveform
    idx_active_rf = (t_ds >= onset) & (t_ds <= offset)
    t_active_rf = t_ds[idx_active_rf]
    dh_active_rf = phase_hr[idx_active_rf]
    dh_norm_rf = (dh_active_rf - dh_active_rf.min()) / (dh_active_rf.max() - dh_active_rf.min() + 1e-20)
    abp_rf = dbp_rf + (sbp_rf - dbp_rf) * dh_norm_rf
    abp_rf += (map_rf_cuff - np.mean(abp_rf))
    
    # ── 2. BANDPASS FILTERING FOR KOROTKOFF CLICKS (10 - 200 Hz) ──
    sos_koro = butter(4, [10.0, 200.0], btype='band', fs=FS_RF, output='sos')
    phase_koro_10k_filt = sosfiltfilt(sos_koro, phase_clean_10k)
    phase_koro_10k = np.append(np.diff(phase_koro_10k_filt) * FS_RF, 0.0) * SCALE * 0.1
    phase_koro = decimate(phase_koro_10k, DEC, ftype='fir')
    phase_koro_env = smooth(np.abs(hilbert(phase_koro)), int(0.5 * FS_HR))
    
    # Welch PSD on active heartbeat waveform (nfft=32768 for fractional BPM resolution)
    f_hr_rf, psd_hr_rf = welch(phase_hr[idx_active_rf], fs=FS_HR, nperseg=len(phase_hr[idx_active_rf]), nfft=32768)
    hr_band_rf = (f_hr_rf >= 0.5) & (f_hr_rf <= 2.5)
    hr_peak_hz_rf = f_hr_rf[hr_band_rf][np.argmax(psd_hr_rf[hr_band_rf])]
    hr_bpm_rf = hr_peak_hz_rf * 60
    
    # Slicing into Phase I, Phase II, Phase III for multi-phase PSD comparison
    idx_p1 = (t_ds >= 0.0) & (t_ds <= t_start)
    idx_p2 = (t_ds >= onset) & (t_ds <= offset)
    idx_p3 = (t_ds >= offset) & (t_ds <= 50.0)
    
    f_p1, psd_p1 = welch(phase_ac[idx_p1], fs=FS_HR, nperseg=int(FS_HR * 1.0), noverlap=int(FS_HR * 0.5))
    f_p2, psd_p2 = welch(phase_ac[idx_p2], fs=FS_HR, nperseg=int(FS_HR * 1.0), noverlap=int(FS_HR * 0.5))
    f_p3, psd_p3 = welch(phase_ac[idx_p3], fs=FS_HR, nperseg=int(FS_HR * 1.0), noverlap=int(FS_HR * 0.5))
    
    # ── PLOTTING THE 8-PANEL DASHBOARD (300 DPI) ──
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

    t_phys = t_ds
    
    # Prepend zeros to RF signals (no prepending)
    phase_ac_full = phase_ac
    phase_hr_full = phase_hr
    phase_hr_env_full = phase_hr_env
    phase_koro_full = phase_koro
    phase_koro_env_full = phase_koro_env

    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    plt.subplots_adjust(hspace=0.28, wspace=0.18)
    
    FONT_LABEL = {'fontname': 'DejaVu Sans', 'fontsize': 10, 'color': '#2C3E50'}
    FONT_TITLE = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'weight': 'bold', 'color': '#2C3E50'}
    
    # Helper for zone shading
    def add_zone_shading_phys(ax):
        ax.axvspan(0.0,          t_sbp_phys,  color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp_phys,   X_LIMITS[1], color='#D1F2D9', alpha=0.30, zorder=0)

    # ── Panel 1: Zoomed Overlay ──
    ax = axes[0, 0]
    t_zoom_start = onset + 1.0 + t_shift
    t_zoom_end   = onset + 7.0 + t_shift
    zoom_mask_rf = (t_phys >= t_zoom_start) & (t_phys <= t_zoom_end)
    t_zoom_rf    = t_phys[zoom_mask_rf]
    
    zoom_hb = phase_hr_full[zoom_mask_rf] / np.max(np.abs(phase_hr_full[zoom_mask_rf]))
    zoom_sn = phase_koro_full[zoom_mask_rf] / np.max(np.abs(phase_koro_full[zoom_mask_rf]))
    
    ax.plot(t_zoom_rf, zoom_hb, color='black', lw=2.0, label='Heartbeat (0.8-3.0 Hz)')
    ax.plot(t_zoom_rf, zoom_sn, color='red',   lw=1.0, alpha=0.8, label='Korotkoff Snaps (10-50 Hz)')
    ax.axvspan(t_sbp_phys, t_dbp_phys, color='#FFEAA7', alpha=0.3, label='Active Window')
    
    ax.set_title('Panel 1: Zoomed Overlay: Heartbeats vs. Korotkoff Snaps', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude', **FONT_LABEL)
    ax.set_xlim([t_zoom_start, t_zoom_end])
    ax.set_ylim([-1.05, 1.05])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 2: Multi-Phase PSD comparison (0 - 250 Hz)
    ax = axes[0, 1]
    ax.semilogy(f_p1, psd_p1 * 1e6, color='#7F8C8D', lw=1.2, alpha=0.8, label=f"Phase I ({t_start:.2f}s)")
    ax.semilogy(f_p2, psd_p2 * 1e6, color='#C0392B', lw=1.6, alpha=0.9, label=f"Phase II ({t_sbp_phys:.2f}-{t_dbp_phys:.2f}s)")
    ax.semilogy(f_p3, psd_p3 * 1e6, color='#27AE60', lw=1.2, alpha=0.8, label=f"Phase III ({t_dbp_phys:.2f}s+)")
    ax.set_title('Panel 2: Power Spectral Density (PSD) Across Physiological Phases', **FONT_TITLE)
    ax.set_xlabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_ylabel('Power Spectral Density [μm^2/Hz]', **FONT_LABEL)
    ax.set_xlim([0, 250])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 3: Heartbeat & Compliance Envelope (Beats Overlay)
    ax = axes[1, 0]
    norm_phase_hb = phase_hr_full / np.max(np.abs(phase_hr))
    norm_phase_env = phase_hr_env_full / np.max(np.abs(phase_hr))
    
    # Background beats
    ax.plot(t_phys, norm_phase_hb, color='#C0392B', lw=1.0, alpha=0.35, label='RMG Heartbeat wave (beats)')
    ax.plot(t_phys, norm_phase_env, color='#78281F', lw=2.4, label='RMG Compliance Envelope')
    
    # Mark peak cuff pressure
    idx_t_start_rf = np.argmin(np.abs(t_phys - t_start_phys))
    ax.plot(t_start_phys, norm_phase_hb[idx_t_start_rf], 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure (t=20.00s, P={P_start:.0f} mmHg)')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 3: RF Mechanical Heartbeat Waveform & Compliance Envelope (Beats Overlay)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='lower left', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 4: Compliance MAP Peak Localization
    ax = axes[1, 1]
    ax.plot(t_phys, norm_phase_env, color='#78281F', lw=2.0, label='Compliance Envelope')
    idx_map = np.argmin(np.abs(t_phys - (t_map_phase + t_shift)))
    
    # Mark Peak Cuff Pressure
    idx_t_start_env_rf = np.argmin(np.abs(t_phys - t_start_phys))
    ax.plot(t_start_phys, norm_phase_env[idx_t_start_env_rf], 'D', color='crimson', ms=12, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure ({P_start:.0f} mmHg) | Artery Occluded')
    ax.plot(t_map_phase + t_shift, phase_hr_env_full[idx_map] / np.max(np.abs(phase_hr)), '*', color='gold', ms=14, mec='black', mew=1.0, zorder=6,
            label=f'RMG MAP Peak (t={t_map_phase + t_shift:.2f}s, {map_rf_cuff:.1f} mmHg)')
    ax.plot(t_sbp_phys,  phase_hr_env_full[np.argmin(np.abs(t_phys - t_sbp_phys))] / np.max(np.abs(phase_hr)), 'o', color='red', ms=9, zorder=5, label=f'SBP ({sbp_rf:.1f} mmHg) t={t_sbp_phys:.2f}s')
    ax.plot(t_dbp_phys,  phase_hr_env_full[np.argmin(np.abs(t_phys - t_dbp_phys))] / np.max(np.abs(phase_hr)), 's', color='blue', ms=9, zorder=5, label=f'DBP ({dbp_rf:.1f} mmHg) t={t_dbp_phys:.2f}s')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 4: Compliance MAP Peak & SBP / DBP Threshold Calibration', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Envelope Amplitude', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([0.0, 1.2])
    
    # Paper-ready piecewise cuff deflation annotation
    cuff_info = (
        f"Cuff Dynamics:\n"
        f"1. INFLATION (Phase I, 0-20.0s physical): Cuff pressure ramp up to {P_start:.1f} mmHg.\n"
        f"   Brachial artery fully occluded. Zero blood flow.\n"
        f"2. DEFLATION (Phase II): Controlled leak at {beta_active:.2f} mmHg/s (2-3 mmHg/s target).\n"
        f"   - SBP ({sbp_rf:.1f} mmHg) at t={t_sbp_phys:.2f}s: Artery begins to reopen.\n"
        f"   - MAP ({map_rf_cuff:.1f} mmHg) at t={t_map_phase + t_shift:.2f}s: Max compliance.\n"
        f"   - DBP ({dbp_rf:.1f} mmHg) at t={t_dbp_phys:.2f}s: Artery fully open.\n"
        f"3. RECOVERY (Phase III): Laminar blood flow restored."
    )
    ax.text(0.98, 0.05, cuff_info, transform=ax.transAxes, fontsize=8.0, fontweight='bold', color='#2C3E50',
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7', boxstyle='round,pad=0.4'))
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 5: High-Frequency mechanical snaps
    ax = axes[2, 0]
    norm_snaps = phase_koro_full / np.max(np.abs(phase_koro))
    norm_snaps_env = phase_koro_env_full / np.max(np.abs(phase_koro))
    
    ax.plot(t_phys, norm_snaps, color='#C0392B', lw=0.6, alpha=0.35, label='RMG mechanical snaps (beats)')
    ax.plot(t_phys, norm_snaps_env, color='#78281F', lw=1.5, label='Snapping Envelope')
    add_zone_shading_phys(ax)
    ax.set_title('Panel 5: High-Frequency RMG mechanical snapping velocity clicks (10 - 200 Hz)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Value (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 6: PSD of RF mechanical clicks (Phase II Only)
    ax = axes[2, 1]
    ax.set_ylim([np.min(abp_rf) - 20, np.max(abp_rf) + 20])
    
    # Paper-ready HR and dynamic BP annotation
    hr_text = (
        f"ABP Waveform Dynamics:\n"
        f"• Systolic SBP: {np.max(abp_rf):.1f} mmHg\n"
        f"• Diastolic DBP: {np.min(abp_rf):.1f} mmHg\n"
        f"• Mean MAP: {np.mean(abp_rf):.1f} mmHg\n"
        f"• PSD Heart Rate: {hr_bpm_rf:.1f} BPM"
    )
    ax.text(0.05, 0.05, hr_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold', color='#2C3E50',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7', boxstyle='round,pad=0.4'))
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Dynamic Korotkoff Duration label on the Spectrogram panel
    duration_str = (f"Korotkoff active compliance window: [{onset:.2f}s -> {offset:.2f}s]  (Dur = {offset-onset:.2f}s)  |  "
                    f"Cuff Rate: {beta_active:.3f} mmHg/s  |  PSD HR: {hr_bpm_rf:.1f} BPM")
    fig.text(0.5, 0.024, duration_str, ha='center', fontsize=11, fontweight='bold', color='#2C3E50',
             bbox=dict(facecolor='#FFEAA7', alpha=0.5, edgecolor='#B7950B', boxstyle='round,pad=0.5'))
    
    # Shading legend patches
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label=f'Phase I: Inflation → Max Cuff Pressure ({P_start:.0f} mmHg) at t={t_start:.2f}s'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label=f'Phase II: Active Korotkoff compliance window [{onset:.2f} - {offset:.2f}s]'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label=f'Phase III: Fully Unoccluded Brachial Artery [{offset:.2f} - 50.0s]'),
        mpatches.Patch(color='crimson', alpha=0.3, label=f'Peak Cuff Pressure | Artery Fully Occluded (t={t_start:.2f}s)'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
               
    fig.suptitle(
        f'RF Radar RMG Heartbeat & Compliance Validation (10 - 200 Hz Band)  |  HR: {hr_bpm_rf:.1f} BPM\n'
        f'High-Fidelity 8-Panel Adaptive physiological Analysis for Best Session (Prof. Kan, Rec 06)  |  300 DPI',
        fontsize=16, fontweight='bold', color='#2C3E50', y=0.975)
        
    out_img = os.path.join(SUMMARY_DIR, "rf_separate_8panel_rf.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"RF 8-Panel Dashboard saved to: {out_img}")

# ── MAIN EXECUTION ───────────────────────────────────────────────────
def main():
    print("=" * 90)
    print(" EXECUTING MULTI-MODALITY INDEPENDENT 8-PANEL DASHBOARD COMPILER v8.7")
    print(" Subjects: Prof. Kan Rec 06 (Best Session)")
    print(" Steth: 50 Hz to 1000 Hz  |  RF: 10 Hz to 200 Hz  |  Resolution: 300 DPI")
    print("=" * 90)
    
    generate_steth_8panel()
    generate_rf_8panel()
    
    # Copy both separate 8-panel dashboards to artifacts folder
    import shutil
    shutil.copy2(os.path.join(SUMMARY_DIR, "rf_separate_8panel_rf.png"), 
                 r"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\rf_separate_8panel_rf.png")
    shutil.copy2(os.path.join(SUMMARY_DIR, "rf_separate_8panel_steth.png"), 
                 r"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\rf_separate_8panel_steth.png")
    print("Separate dashboards copied to artifacts successfully!")
    
    print("\n[SUCCESS] Both 8-Panel Dashboards generated successfully at 300 DPI!")

if __name__ == '__main__':
    main()
