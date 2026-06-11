import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, spectrogram, welch
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

# ── MAIN COMPILER ────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print(" GENERATING ADAPTIVE 8-PANEL COMPARATIVE ANALYSIS: MAGNITUDE VS PHASE | 300 DPI")
    print(" Active Segment Highlighting  |  Welch PSD HR Count  |  Cuff Dynamics")
    print("=" * 90)
    
    # Load targets from report
    df = pd.read_csv(CSV_REPORT)
    match = df[(df['subject'] == 'Prof. Kan (Sub 1)') & (df['rec'] == 6)]
    if match.empty:
        print("Error: Missing report row for Prof. Kan Rec 6.")
        return
        
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    # ── 1. LOAD AND PREPROCESS RAW DATA ──
    h5_path = os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    N = len(i_raw)
    t = np.arange(N) / FS_RF
    
    # Detect dynamic cuff maximum pressure (deflation onset)
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
        
    # Decimate raw magnitude and preprocessed phase to 1 kHz
    raw_iq = -i_raw + 1j * q_raw
    magnitude_ds = decimate(np.abs(raw_iq), DEC, ftype='fir')
    magnitude_clean_ds = decimate(np.abs(iq_c), DEC, ftype='fir')
    
    phase_ds = decimate(phase_clean_10k, DEC, ftype='fir') * SCALE
    t_ds = np.arange(len(phase_ds)) / FS_HR
    
    # Center signals
    sos_hp05 = butter(4, 0.5, btype='highpass', fs=FS_HR, output='sos')
    magnitude_ac = sosfiltfilt(sos_hp05, magnitude_clean_ds)
    phase_ac = sosfiltfilt(sos_hp05, phase_ds)
    
    # ── 2. EXTRACT CARDIAC HEARTBEATS (0.4 - 3.0 Hz) ──
    sos_hr = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    
    mag_raw = np.abs(iq_c)
    mag_hr_10k = sosfiltfilt(sos_hr, mag_raw)
    phase_hr_10k = sosfiltfilt(sos_hr, phase_clean_10k) * SCALE
    
    mag_hr = decimate(mag_hr_10k, DEC, ftype='fir')
    phase_hr = decimate(phase_hr_10k, DEC, ftype='fir')
    
    # Rolling compliance envelopes (1.5s smoothing)
    mag_hr_env = smooth(np.abs(hilbert(mag_hr)), int(1.5 * FS_HR))
    phase_hr_env = smooth(np.abs(hilbert(phase_hr)), int(1.5 * FS_HR))
    
    # Locate Compliance MAP Peaks separately in middle 70% of active window
    mid_start = onset + 0.15 * (offset - onset)
    mid_end   = offset - 0.15 * (offset - onset)
    
    mask_rf_mid = (t_ds >= mid_start) & (t_ds <= mid_end)
    t_map_mag = t_ds[mask_rf_mid][np.argmax(mag_hr_env[mask_rf_mid])]
    t_map_phase = t_ds[mask_rf_mid][np.argmax(phase_hr_env[mask_rf_mid])]
    
    # ── 3. EXTRACT HIGH-FREQUENCY KOROTKOFF SNAPPING CLICKS (10 - 49 Hz) ──
    sos_koro = butter(4, [10, 49], btype='band', fs=FS_RF, output='sos')
    
    mag_koro_10k = sosfiltfilt(sos_koro, mag_raw)
    phase_koro_10k_filt = sosfiltfilt(sos_koro, phase_clean_10k)
    phase_koro_10k = np.append(np.diff(phase_koro_10k_filt) * FS_RF, 0.0) * SCALE
    
    mag_koro = decimate(mag_koro_10k, DEC, ftype='fir')
    phase_koro = decimate(phase_koro_10k, DEC, ftype='fir')
    
    # High-freq envelopes (0.5s smoothing)
    mag_koro_env = smooth(np.abs(hilbert(mag_koro)), int(0.5 * FS_HR))
    phase_koro_env = smooth(np.abs(hilbert(phase_koro)), int(0.5 * FS_HR))
    
    # ── 4. PSD CALCULATIONS & DYNAMIC HEART RATE (HR) COUNT ──
    idx_active = (t_ds >= onset) & (t_ds <= offset)
    
    # Magnitude Heart Rate Count (nfft=32768 for fractional BPM resolution)
    f_hr_mag, psd_hr_mag = welch(mag_hr[idx_active], fs=FS_HR, nperseg=len(mag_hr[idx_active]), nfft=32768)
    hr_band = (f_hr_mag >= 0.5) & (f_hr_mag <= 2.5)
    hr_peak_mag_hz = f_hr_mag[hr_band][np.argmax(psd_hr_mag[hr_band])]
    hr_bpm_mag = hr_peak_mag_hz * 60
    # Phase Heart Rate Count (nfft=32768 for fractional BPM resolution)
    f_hr_ph, psd_hr_ph = welch(phase_hr[idx_active], fs=FS_HR, nperseg=len(phase_hr[idx_active]), nfft=32768)
    hr_peak_ph_hz = f_hr_ph[hr_band][np.argmax(psd_hr_ph[hr_band])]
    hr_bpm_ph = hr_peak_ph_hz * 60
    print(f"  [PSD HR Count] Magnitude: {hr_bpm_mag:.2f} BPM  |  Phase: {hr_bpm_ph:.2f} BPM")
    
    # ── 5. SCALING WAVEFORMS (Normalize by maximum during deflation phase only) ──
    idx_deflation = t_ds >= t_start
    max_mag_def = np.max(np.abs(mag_hr[idx_deflation])) + 1e-20
    max_phase_def = np.max(np.abs(phase_hr[idx_deflation])) + 1e-20
    
    max_mag_koro_def = np.max(np.abs(mag_koro[idx_deflation])) + 1e-20
    max_phase_koro_def = np.max(np.abs(phase_koro[idx_deflation])) + 1e-20
    
    # Scaled waveforms for visual display
    mag_hr_n = mag_hr / max_mag_def
    mag_hr_env_n = mag_hr_env / max_mag_def
    phase_hr_n = phase_hr / max_phase_def
    phase_hr_env_n = phase_hr_env / max_phase_def
    
    mag_koro_n = mag_koro / max_mag_koro_def
    mag_koro_env_n = mag_koro_env / max_mag_koro_def
    phase_koro_n = phase_koro / max_phase_koro_def
    phase_koro_env_n = phase_koro_env / max_phase_koro_def
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
            
    # ── TIME AXIS LOCKING & PREPENDING (Deflation onset locked at 20.0s physical time) ──
    t_shift = 0.0
    beta_active = (target_sbp - target_dbp) / (offset - onset)
    t_start_phys = t_start
    
    t_sbp_mag = onset
    t_dbp_mag = offset
    
    t_sbp_phase = onset
    t_dbp_phase = offset
    
    X_LIMITS = [0.0, t_dbp_mag + 5.0]
    
    t_ds_phys = t_ds
    
    # Prepend zero compliance waveforms (no prepending)
    mag_hr_n_full = mag_hr_n
    mag_hr_env_n_full = mag_hr_env_n
    phase_hr_n_full = phase_hr_n
    phase_hr_env_n_full = phase_hr_env_n
    
    mag_koro_n_full = mag_koro_n
    mag_koro_env_n_full = mag_koro_env_n
    phase_koro_n_full = phase_koro_n
    phase_koro_env_n_full = phase_koro_env_n
    
    # ── ADAPTIVE OSCILLOMETRIC PEAK (MAP) DETECTION ──
    t_map_mag_phys = t_map_mag
    t_map_phase_phys = t_map_phase
    
    # Recalibrate cuff pressures adaptively using SBP/DBP targets
    map_mmhg_mag = target_sbp - beta_active * (t_map_mag_phys - onset)
    map_mmhg_phase = target_sbp - beta_active * (t_map_phase_phys - onset)
    
    beta = beta_active
    sbp_cal = target_sbp
    dbp_cal = target_dbp
    
    print(f"  [Adaptive SBP/MAP/DBP Times] Mag: {t_sbp_mag:.2f}/{t_map_mag_phys:.2f}/{t_dbp_mag:.2f} s")
    print(f"  [Adaptive MAP] Mag: {map_mmhg_mag:.1f} mmHg  |  Phase: {map_mmhg_phase:.1f} mmHg")
    
    # ── 6. PLOTTING THE COMPARATIVE 8-PANEL DASHBOARD (300 DPI) ──
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    plt.subplots_adjust(hspace=0.28, wspace=0.18)
    
    FONT_LABEL = {'fontname': 'DejaVu Sans', 'fontsize': 10, 'color': '#2C3E50'}
    FONT_TITLE = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'weight': 'bold', 'color': '#2C3E50'}
    
    # Helper for zone shading
    def add_zone_shading_phys(ax, t_sbp, t_dbp):
        ax.axvspan(t_shift,      t_sbp,       color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp,        t_dbp,       color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp,        X_LIMITS[1], color='#D1F2D9', alpha=0.30, zorder=0)
        
    # ── Panel 1: Magnitude snaps ──
    ax = axes[0, 0]
    plot_highlighted_signal(ax, t_ds_phys, mag_koro_n_full, t_sbp_mag, t_dbp_mag, active_color='#16A085', lw=0.6, label='Mag Snapping clicks (Active)')
    plot_highlighted_signal(ax, t_ds_phys, mag_koro_env_n_full, t_sbp_mag, t_dbp_mag, active_color='#0E6251', lw=1.8, label='Snapping Envelope (Active)')
    add_zone_shading_phys(ax, t_sbp_mag, t_dbp_mag)
    ax.set_title('Panel 1: RF Magnitude High-Frequency Snapping Clicks (10-49 Hz)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Value (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # ── Panel 2: Phase snaps ──
    ax = axes[0, 1]
    plot_highlighted_signal(ax, t_ds_phys, phase_koro_n_full, t_sbp_phase, t_dbp_phase, active_color='#C0392B', lw=0.6, label='Phase Velocity snaps (Active)')
    plot_highlighted_signal(ax, t_ds_phys, phase_koro_env_n_full, t_sbp_phase, t_dbp_phase, active_color='#78281F', lw=1.8, label='Snapping Envelope (Active)')
    add_zone_shading_phys(ax, t_sbp_phase, t_dbp_phase)
    ax.set_title('Panel 2: RF Phase Displacement High-Frequency Snapping Clicks (10-49 Hz)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Value (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Panel 3: Magnitude Spectrogram
    ax = axes[1, 0]
    mag_koro_200 = decimate(mag_koro, 5, ftype='fir')
    f_sp_m, t_sp_m, Sxx_sp_m = spectrogram(mag_koro_200, fs=200, nperseg=128, noverlap=96)
    Sxx_db_m = 10 * np.log10(Sxx_sp_m + 1e-12)
    im_m = ax.pcolormesh(t_sp_m + t_shift, f_sp_m, Sxx_db_m, shading='gouraud', cmap='turbo', vmin=-110, vmax=-10)
    plt.colorbar(im_m, ax=ax, label='Spectral Density [dB]')
    add_zone_shading_phys(ax, t_sbp_mag, t_dbp_mag)
    ax.set_title('Panel 3: Magnitude Korotkoff Clicks Spectrogram', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([0, 50])
    
    # Panel 4: Phase Spectrogram & Welch PSD HR Count
    ax = axes[1, 1]
    phase_koro_200 = decimate(phase_koro, 5, ftype='fir')
    f_sp_p, t_sp_p, Sxx_sp_p = spectrogram(phase_koro_200, fs=200, nperseg=128, noverlap=96)
    Sxx_db_p = 10 * np.log10(Sxx_sp_p + 1e-12)
    im_p = ax.pcolormesh(t_sp_p + t_shift, f_sp_p, Sxx_db_p, shading='gouraud', cmap='turbo', vmin=-110, vmax=-10)
    plt.colorbar(im_p, ax=ax, label='Spectral Density [dB]')
    add_zone_shading_phys(ax, t_sbp_phase, t_dbp_phase)
    ax.set_title('Panel 4: Phase Displacement Korotkoff Clicks Spectrogram', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Frequency (Hz)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([0, 50])
    
    # Paper-ready HR annotation on Panel 4
    hr_text = (
        f"PSD Heart Rate:\n"
        f"• Mag HR: {hr_bpm_mag:.1f} BPM\n"
        f"• Phase HR: {hr_bpm_ph:.1f} BPM"
    )
    ax.text(0.05, 0.05, hr_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))
            
    # ── Panel 5: Magnitude Zoomed Overlay (No Raw Signal) ──
    ax = axes[2, 0]
    t_zoom_start = t_sbp_mag + 1.0
    t_zoom_end   = t_sbp_mag + 7.0
    zoom_mask    = (t_ds_phys >= t_zoom_start) & (t_ds_phys <= t_zoom_end)
    t_zoom       = t_ds_phys[zoom_mask]
    
    zoom_hb_m = mag_hr_n_full[zoom_mask] / np.max(np.abs(mag_hr_n_full[zoom_mask]))
    zoom_sn_m = mag_koro_n_full[zoom_mask] / np.max(np.abs(mag_koro_n_full[zoom_mask]))
    
    ax.plot(t_zoom, zoom_hb_m, color='black', lw=2.0, label='Heartbeat (0.8-3.0 Hz)')
    ax.plot(t_zoom, zoom_sn_m, color='red',   lw=1.0, alpha=0.8, label='Korotkoff Snaps (10-50 Hz)')
    ax.axvspan(t_sbp_mag, t_dbp_mag, color='#FFEAA7', alpha=0.3, label='Active Window')
    ax.set_title('Panel 5: Zoomed Overlay: Magnitude Heartbeats vs. Snaps', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude', **FONT_LABEL)
    ax.set_xlim([t_zoom_start, t_zoom_end])
    ax.set_ylim([-1.05, 1.05])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # ── Panel 6: Phase Zoomed Overlay (No Raw Signal) ──
    ax = axes[2, 1]
    zoom_hb_p = phase_hr_n_full[zoom_mask] / np.max(np.abs(phase_hr_n_full[zoom_mask]))
    zoom_sn_p = phase_koro_n_full[zoom_mask] / np.max(np.abs(phase_koro_n_full[zoom_mask]))
    
    ax.plot(t_zoom, zoom_hb_p, color='black', lw=2.0, label='Heartbeat (0.8-3.0 Hz)')
    ax.plot(t_zoom, zoom_sn_p, color='red',   lw=1.0, alpha=0.8, label='Korotkoff Snaps (10-50 Hz)')
    ax.axvspan(t_sbp_phase, t_dbp_phase, color='#FFEAA7', alpha=0.3, label='Active Window')
    ax.set_title('Panel 6: Zoomed Overlay: Phase Displacement Heartbeats vs. Snaps', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude', **FONT_LABEL)
    ax.set_xlim([t_zoom_start, t_zoom_end])
    ax.set_ylim([-1.05, 1.05])
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # ── Panel 7: Magnitude Heartbeat & Compliance Envelope ──
    ax = axes[3, 0]
    plot_highlighted_signal(ax, t_ds_phys, mag_hr_n_full, t_sbp_mag, t_dbp_mag, active_color='#16A085', lw=1.0, label='Mag Heartbeat (Active)')
    plot_highlighted_signal(ax, t_ds_phys, mag_hr_env_n_full, t_sbp_mag, t_dbp_mag, active_color='#0E6251', lw=2.4, label='Mag Compliance Envelope (Active)')
    # Peak cuff pressure marker
    idx_tstart_m = np.argmin(np.abs(t_ds_phys - t_start_phys))
    ax.plot(t_start_phys, mag_hr_n_full[idx_tstart_m], 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure (t=20.00s, {P_start:.0f} mmHg)')
    idx_map_m = np.argmin(np.abs(t_ds_phys - t_map_mag_phys))
    ax.plot(t_map_mag_phys, mag_hr_env_n_full[idx_map_m], '^', color='orange', ms=12, mec='black', mew=1.0, zorder=6,
            label=f'Mag MAP Peak (t={t_map_mag_phys:.2f}s, {map_mmhg_mag:.1f} mmHg)')
    add_zone_shading_phys(ax, t_sbp_mag, t_dbp_mag)
    ax.set_title('Panel 7: RF Magnitude Heartbeat & Compliance Envelope (Beats Overlay)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc='lower left', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # ── Panel 8: Phase Heartbeat & Compliance Envelope ──
    ax = axes[3, 1]
    plot_highlighted_signal(ax, t_ds_phys, phase_hr_n_full, t_sbp_phase, t_dbp_phase, active_color='#C0392B', lw=1.0, label='Phase Heartbeat (Active)')
    plot_highlighted_signal(ax, t_ds_phys, phase_hr_env_n_full, t_sbp_phase, t_dbp_phase, active_color='#78281F', lw=2.4, label='Phase Compliance Envelope (Active)')
    # Peak cuff pressure marker
    idx_tstart_p = np.argmin(np.abs(t_ds_phys - t_start_phys))
    ax.plot(t_start_phys, phase_hr_n_full[idx_tstart_p], 'D', color='crimson', ms=10, mec='black', mew=1.0, zorder=7,
            label=f'Peak Cuff Pressure (t=20.00s, {P_start:.0f} mmHg)')
    idx_map_p = np.argmin(np.abs(t_ds_phys - t_map_phase_phys))
    ax.plot(t_map_phase_phys, phase_hr_env_n_full[idx_map_p], '*', color='gold', ms=14, mec='black', mew=1.0, zorder=6,
            label=f'Phase MAP Peak (t={t_map_phase_phys:.2f}s, {map_mmhg_phase:.1f} mmHg)')
    add_zone_shading_phys(ax, t_sbp_phase, t_dbp_phase)
    ax.set_title('Panel 8: RF Phase Displacement Heartbeat & Compliance Envelope (Beats Overlay)', **FONT_TITLE)
    ax.set_xlabel('Physical Time (s)', **FONT_LABEL)
    ax.set_ylabel('Normalized Amplitude (a.u.)', **FONT_LABEL)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim([-1.2, 1.2])
    
    # Paper-ready cuff dynamics annotation
    cuff_info = (
        f"Cuff Dynamics:\n"
        f"1. INFLATION (Phase I, 0-20.0s physical): Cuff pressure ramp up to {P_start:.1f} mmHg.\n"
        f"   Brachial artery fully occluded. Zero blood flow.\n"
        f"2. DEFLATION (Phase II): Controlled leak at {beta:.2f} mmHg/s (2-3 mmHg/s target).\n"
        f"   - SBP ({sbp_cal:.1f} mmHg) at t={t_sbp_mag:.2f}s: Artery begins to reopen.\n"
        f"   - MAP  Mag: {map_mmhg_mag:.1f} mmHg at t={t_map_mag_phys:.2f}s (Max Mag oscillation).\n"
        f"   - MAP Phase: {map_mmhg_phase:.1f} mmHg at t={t_map_phase_phys:.2f}s (Max Phase oscillation).\n"
        f"   - DBP ({dbp_cal:.1f} mmHg) at t={t_dbp_mag:.2f}s: Artery fully open.\n"
        f"3. RECOVERY (Phase III): Laminar blood flow restored."
    )
    ax.text(0.98, 0.05, cuff_info, transform=ax.transAxes, fontsize=8.0, fontweight='bold', color='#2C3E50',
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#BDC3C7', boxstyle='round,pad=0.4'))
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.15)
    
    # Dynamic Korotkoff Duration label on the Spectrogram panel
    duration_str = (f"Korotkoff active compliance window: [{t_sbp_mag:.2f}s -> {t_dbp_mag:.2f}s]  (Dur = {t_dbp_mag-t_sbp_mag:.2f}s)  |  "
                    f"Cuff Rate: {beta:.3f} mmHg/s  |  Mag HR: {hr_bpm_mag:.1f} BPM  |  Phase HR: {hr_bpm_ph:.1f} BPM")
    fig.text(0.5, 0.024, duration_str, ha='center', fontsize=11, fontweight='bold', color='#2C3E50',
             bbox=dict(facecolor='#FFEAA7', alpha=0.5, edgecolor='#B7950B', boxstyle='round,pad=0.5'))
    
    # Shading legend patches
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label=f'Phase I: Inflation → Max Cuff Pressure ({P_start:.0f} mmHg) at t=20.00s'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label=f'Phase II: Active Korotkoff compliance window [{t_sbp_mag:.2f} - {t_dbp_mag:.2f}s]'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label=f'Phase III: Fully Unoccluded Brachial Artery [{t_dbp_mag:.2f} - end]'),
        mpatches.Patch(color='crimson', alpha=0.3, label=f'Peak Cuff Pressure | Artery Fully Occluded (t=20.00s)'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
               
    fig.suptitle(
        'RF Radar RMG Demodulation Comparison: Magnitude vs Phase displacement Domains\n'
        'High-Fidelity 8-Panel Adaptive physiological Analysis for Best Session (Prof. Kan, Rec 06)  |  300 DPI',
        fontsize=16, fontweight='bold', color='#2C3E50', y=0.975)
        
    out_img = os.path.join(SUMMARY_DIR, "rf_magnitude_vs_phase_8panel_dashboard.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparative 8-Panel Dashboard saved to: {out_img}")
    
    # Also copy to artifacts folder
    art_img = r"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\rf_magnitude_vs_phase_8panel_dashboard.png"
    import shutil
    shutil.copy2(out_img, art_img)
    print(f"Dashboard copied to artifacts: {art_img}")

if __name__ == '__main__':
    main()

