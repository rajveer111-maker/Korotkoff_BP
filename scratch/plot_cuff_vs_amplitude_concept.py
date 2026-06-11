import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, decimate
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
DEC       = 10
FS_HR     = FS_RF / DEC      # 1 kHz downsampled rate
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000   # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)    # ~26.5 mm/rad

BASE        = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
CSV_REPORT  = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

# ── UTILITIES ─────────────────────────────────────────────────────────
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
    return (ic) + 1j * ((qc - ic * sp / al) / cp)

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    energy = np.abs(sosfiltfilt(sos_hp, iq))
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
    end_val  = np.mean(energy_smooth[max(0, int(max_search_sec*100)-50):int(max_search_sec*100)])
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        return 0.0
    baseline  = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    t_det = 8.0
    for i in range(peak_idx, len(t_search)):
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
    return t_det

# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(" CUFF PRESSURE vs. HEARTBEAT PULSE WAVE & AMPLITUDE  |  CONCEPT PLOT  |  300 DPI")
    print(" Physical Timeline: Inflation (0 to 20s) -> Deflation (20s to 60s)")
    print(" Modalities: Stethoscope PCG  +  RF Phase Displacement")
    print("=" * 80)

    # ── Load metadata ──
    df    = pd.read_csv(CSV_REPORT)
    match = df[(df['subject'] == 'Prof. Kan (Sub 1)') & (df['rec'] == 6)]
    if match.empty:
        print("ERROR: Prof. Kan Rec 6 not found in report."); return
    onset  = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])

    # ── Load RF H5 ──
    h5_path = os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]

    # ── Adaptive peak cuff pressure detection in H5 timeline ──
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=FS_RF, onset_limit=onset)
    print(f"  [Adaptive] Peak cuff pressure at t_start = {t_start:.3f} s in recording")

    # ── Calibration ──
    P_start    = 150.0
    target_sbp = 125.0
    target_dbp = 75.0
    csv_abp    = os.path.join(SUMMARY_DIR, 'cross_subject_abp_report.csv')
    if os.path.exists(csv_abp):
        df_a = pd.read_csv(csv_abp)
        m    = df_a[(df_a['subject'].str.contains('Sub 1')) & (df_a['rec'] == 6)]
        if not m.empty:
            target_sbp = float(m.iloc[0]['sbp_rf'])
            target_dbp = float(m.iloc[0]['dbp_rf'])

    beta_init   = (P_start - target_sbp) / max(onset - t_start, 0.1)
    beta_active = (target_sbp - target_dbp) / max(offset - onset, 0.1)

    # Time when cuff reaches ~60 mmHg (full open / residual)
    P_full_open  = 60.0
    t_full_open  = onset + (target_sbp - P_full_open) / beta_active
    t_full_open  = min(t_full_open, 50.0)

    # ── Physical Timeline Mapping ──
    # We map t_start in recording to exactly 20.0s physical time.
    t_shift = 20.0 - t_start
    t_start_phys = 20.0
    t_sbp_phys   = onset + t_shift
    t_dbp_phys   = offset + t_shift
    t_open_phys  = t_full_open + t_shift

    # ── RF Phase Processing ──
    idx_def = int(t_start * FS_RF) if t_start > 0.5 else int(8.0 * FS_RF)
    iq      = b210_iq_condition(-i_raw + 1j * q_raw)
    iq_c    = sosfiltfilt(butter(4, 50.0, btype='low', fs=FS_RF, output='sos'), iq)

    puw    = np.unwrap(np.angle(iq_c[idx_def:]))
    dp     = np.diff(puw); dp -= np.median(dp); dp = np.clip(dp, -0.5, 0.5)
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)
    ph_inf = np.angle(iq_c[:idx_def])
    w_sz   = min(int(FS_RF), idx_def)
    if w_sz >= 10:
        ph_inf -= pd.Series(ph_inf).rolling(w_sz, center=True).mean().bfill().ffill().values
    if len(ph_inf) > 0:
        ph_inf += ph_def[0] - ph_inf[-1]
    phase_clean_10k = np.concatenate([ph_inf, ph_def]) if len(ph_inf) > 0 else ph_def

    # Heartbeat band 0.4–3 Hz
    phase_hr_10k = sosfiltfilt(butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos'),
                                phase_clean_10k) * SCALE
    phase_hr     = decimate(phase_hr_10k, DEC, ftype='fir')
    t_ds         = np.arange(len(phase_hr)) / FS_HR
    phase_hr_env = smooth(np.abs(hilbert(phase_hr)), int(1.5 * FS_HR))

    # ── Stethoscope Processing ──
    wav_path = os.path.join(BASE, "Sub_1_Prof_kan", "sthethoscope_rec06.wav")
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_a = np.arange(len(audio)) / fs_a

    sos_bp_a   = butter(4, [50.0, 1000.0], btype='band', fs=fs_a, output='sos')
    audio_filt = sosfiltfilt(sos_bp_a, audio)
    audio_env  = np.abs(hilbert(audio_filt))
    sos_hr_a   = butter(4, [0.4, 3.0], btype='band', fs=fs_a, output='sos')
    dh_acoustic = sosfiltfilt(sos_hr_a, audio_env)
    dh_a_env    = smooth(np.abs(hilbert(dh_acoustic)), int(1.5 * fs_a))

    # Downsample Steth signals to 1000 Hz grid (identical to RF t_ds)
    dh_acoustic_ds = np.interp(t_ds, t_a, dh_acoustic)
    env_a_scaled = np.interp(t_ds, t_a, dh_a_env)

    # ── MAP Peak Localization (recording timeline) ──
    mid_s = onset + 0.15 * (offset - onset)
    mid_e = offset - 0.15 * (offset - onset)
    mask_mid_rf  = (t_ds >= mid_s) & (t_ds <= mid_e)
    t_map_phase  = t_ds[mask_mid_rf][np.argmax(phase_hr_env[mask_mid_rf])]
    map_p_phase  = target_sbp - beta_active * (t_map_phase - onset)

    t_map_acoustic = t_ds[mask_mid_rf][np.argmax(env_a_scaled[mask_mid_rf])]
    map_p_acoustic = target_sbp - beta_active * (t_map_acoustic - onset)

    t_map_rf_phys = t_map_phase + t_shift
    t_map_ac_phys = t_map_acoustic + t_shift

    # ── Scale both waveforms to identical mm range ──
    idx_active = (t_ds >= onset) & (t_ds <= offset)
    rf_peak      = np.max(phase_hr_env[idx_active]) + 1e-20
    dh_acoustic_scaled = (dh_acoustic_ds / (np.max(np.abs(dh_acoustic_ds[idx_active])) + 1e-20)) * rf_peak
    env_a_scaled = (env_a_scaled / (np.max(env_a_scaled[idx_active]) + 1e-20)) * rf_peak

    # ── Prepend physical inflation phase (0 to t_shift) ──
    n_prepend = int(t_shift * FS_HR)
    t_prepend = np.arange(n_prepend) / FS_HR

    # Cuff pressure prepend: linear ramp up to P_start (150 mmHg)
    cuff_p_prepend = (P_start / t_shift) * t_prepend
    cuff_p_rec = np.where(
        t_ds < t_start,
        P_start,
        np.where(
            t_ds <= onset,
            P_start      - beta_init   * (t_ds - t_start),
            target_sbp   - beta_active * (t_ds - onset)
        )
    )

    t_phys_full = np.concatenate([t_prepend, t_ds + t_shift])
    cuff_p_full = np.concatenate([cuff_p_prepend, cuff_p_rec])

    # Heartbeat prepends (all zero during unrecorded inflation)
    phase_hr_full = np.concatenate([np.zeros(n_prepend), phase_hr])
    dh_acoustic_full = np.concatenate([np.zeros(n_prepend), dh_acoustic_scaled])
    env_rf_full = np.concatenate([np.zeros(n_prepend), phase_hr_env])
    env_a_full = np.concatenate([np.zeros(n_prepend), env_a_scaled])

    # Key amplitude values
    def val_at_t_phys(wave, t_ev_phys):
        return wave[np.argmin(np.abs(t_phys_full - t_ev_phys))]

    amp_occluded   = val_at_t_phys(env_rf_full, t_start_phys)
    amp_sbp_rf     = val_at_t_phys(env_rf_full, t_sbp_phys)
    amp_map_phase  = val_at_t_phys(env_rf_full, t_map_rf_phys)
    amp_map_acou   = val_at_t_phys(env_a_full, t_map_ac_phys)
    amp_dbp_rf     = val_at_t_phys(env_rf_full, t_dbp_phys)
    amp_open_rf    = val_at_t_phys(env_rf_full, t_open_phys)

    # ── FIGURE LAYOUT ──
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.1, 2.2, 0.55], hspace=0.06,
                            top=0.93, bottom=0.10, left=0.07, right=0.97)

    C_RF    = '#C0392B'   # red  — RF Phase
    C_STETH = '#16A085'   # teal — Stethoscope
    C_CUFF  = '#8E44AD'   # purple — Cuff pressure

    FONT_L = {'fontname': 'DejaVu Sans', 'fontsize': 11, 'color': '#2C3E50'}
    FONT_T = {'fontname': 'DejaVu Sans', 'fontsize': 13, 'weight': 'bold', 'color': '#2C3E50'}

    # ── Zone shading helper ──
    def shade_zones(ax):
        ax.axvspan(0.0,          t_sbp_phys,  color='#AED6F1', alpha=0.22, zorder=0) # OCCLUDED
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FDEBD0', alpha=0.30, zorder=0) # KOROTKOFF
        ax.axvspan(t_dbp_phys,   t_open_phys, color='#D5F5E3', alpha=0.30, zorder=0) # UNOCCLUDED

    # Event lines definition: (time, colour, linestyle, linewidth)
    ev_lines = [
        (t_start_phys,  'crimson',      '--', 2.2),
        (t_sbp_phys,    '#C0392B',      '-',  1.8),
        (t_map_ac_phys, '#27AE60',      '-.',  1.6),
        (t_map_rf_phys, '#F39C12',      '-.',  1.6),
        (t_dbp_phys,    '#2471A3',      '-',  1.8),
        (t_open_phys,   '#1ABC9C',      '--', 2.0),
    ]

    def draw_event_lines(ax):
        for (tv, col, ls, lw) in ev_lines:
            ax.axvline(tv, color=col, ls=ls, lw=lw, zorder=5, alpha=0.9)

    # ════════════════════════════════════════════════════════
    # PANEL 1 — Cuff Pressure model
    # ════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t_phys_full, cuff_p_full, color=C_CUFF, lw=3.0, label='Cuff Pressure (mmHg)', zorder=4)

    # Mark events on cuff curve
    ev_pts = [
        (t_start_phys,  P_start,    'D', 'crimson',  13, f'Peak\n{P_start:.0f} mmHg'),
        (t_sbp_phys,    target_sbp, 'o', '#C0392B',  11, f'SBP\n{target_sbp:.0f}'),
        (t_map_ac_phys, map_p_acoustic,'^','#27AE60', 11, f'MAP\n{map_p_acoustic:.0f}\n(PCG)'),
        (t_map_rf_phys, map_p_phase,'*', '#F39C12',  14, f'MAP\n{map_p_phase:.0f}\n(RF)'),
        (t_dbp_phys,    target_dbp, 's', '#2471A3',  11, f'DBP\n{target_dbp:.0f}'),
        (t_open_phys,   P_full_open,'P', '#1ABC9C',  13, f'~{P_full_open:.0f} mmHg\n(Full Open)'),
    ]
    for (tv, pv, mk, col, ms, lbl) in ev_pts:
        ax1.plot(tv, pv, mk, color=col, ms=ms, mec='black', mew=1.0, zorder=8)

    shade_zones(ax1)
    draw_event_lines(ax1)
    ax1.set_ylabel('Cuff Pressure (mmHg)', **FONT_L)
    ax1.set_ylim([-5, P_start + 20])
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim([-1.0, t_open_phys + 1.5])

    # Phase zone labels on cuff panel
    ylim1 = ax1.get_ylim()
    y_lbl = ylim1[1] - 12
    ax1.text(t_sbp_phys / 2,                  y_lbl, 'OCCLUDED\n(Phase I)',
             ha='center', va='top', fontsize=9.5, color='#1A5276', style='italic', fontweight='bold')
    ax1.text((t_sbp_phys + t_dbp_phys)/2,     y_lbl, 'KOROTKOFF\n(Phase II)',
             ha='center', va='top', fontsize=9.5, color='#7D6608', style='italic', fontweight='bold')
    ax1.text((t_dbp_phys + t_open_phys)/2,    y_lbl, 'UNOCCLUDED\n(Phase III)',
             ha='center', va='top', fontsize=9.5, color='#1E8449', style='italic', fontweight='bold')

    ax1.set_title(
        'Brachial Cuff Pressure & Heartbeat Pulse Amplitude  --  Physiological Concept Plot\n'
        f'Best Session: Prof. Kan Rec 06  |  RF Radar RMG (beats) vs. Stethoscope PCG (beats)  |  300 DPI',
        **FONT_T, pad=8)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Physical timeline note on cuff panel
    phys_note = (
        f"Physical Timeline:\n"
        f"  Inflation Phase : 0.0s to 20.0s physical (including linear pump ramp up & peak hold)\n"
        f"  Peak cuff press : t = 20.0s physical (exact Omron cuff onset point, P = {P_start:.1f} mmHg)\n"
        f"  Deflation Phase : Slow leak from 20.0s to 60.0s at {beta_active:.2f} mmHg/s\n"
        f"  Artery Occlusion: Fully closed from 0.0s to SBP ({t_sbp_phys:.1f}s), reopening turbulent window follows"
    )
    ax1.text(0.01, 0.05, phys_note, transform=ax1.transAxes, fontsize=8.5,
             color='#5D6D7E', va='bottom', ha='left', style='italic',
             bbox=dict(facecolor='#F8F9FA', alpha=0.88, edgecolor='#AEB6BF',
                       boxstyle='round,pad=0.4'))

    # ════════════════════════════════════════════════════════
    # PANEL 2 — Heartbeat Pulse Amplitude & Waves (mm)
    # ════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot actual cardiac heartbeat waves (in beats way!)
    ax2.plot(t_phys_full, phase_hr_full, color=C_RF, lw=1.2, alpha=0.35, label='RF Displacement Cardiac Wave (beats)')
    ax2.plot(t_phys_full, env_rf_full, color=C_RF, lw=2.6, alpha=0.95, label=f'RF Phase compliance envelope (mm) - peak={amp_map_phase:.3f} mm')

    ax2.plot(t_phys_full, dh_acoustic_full, color=C_STETH, lw=1.0, alpha=0.30, ls='--', label='Steth Acoustic Cardiac Wave (beats)')
    ax2.plot(t_phys_full, env_a_full, color=C_STETH, lw=2.2, ls='--', alpha=0.90, label=f'Steth PCG compliance envelope (scaled) - peak={amp_map_acou:.3f} mm')

    shade_zones(ax2)
    draw_event_lines(ax2)

    # Amplitude markers (RF only - primary measurement)
    marker_events = [
        (t_start_phys,  amp_occluded,  'D', 'crimson',  12, 'below', 'Peak pressure\nArtery Fully Closed\n(t=20.0s)'),
        (t_sbp_phys,    amp_sbp_rf,    'o', '#C0392B',  10, 'above', f'SBP: Reopening\nClicks Begin'),
        (t_map_rf_phys, amp_map_phase, '*', '#F39C12',  16, 'above', f'RF MAP Peak\n({map_p_phase:.0f} mmHg)\nMax Wall oscillation'),
        (t_map_ac_phys, amp_map_acou,  '^', '#27AE60',  13, 'above', f'Acoustic MAP Peak\n({map_p_acoustic:.0f} mmHg)'),
        (t_dbp_phys,    amp_dbp_rf,    's', '#2471A3',  10, 'below', f'DBP: Artery\nFully Open'),
        (t_open_phys,   amp_open_rf,   'P', '#1ABC9C',  12, 'above', f'~{P_full_open:.0f} mmHg\nFull Open'),
    ]
    
    y_max2 = max(np.max(env_rf_full), np.max(env_a_full)) * 1.35
    ax2.set_ylim([-0.28 * y_max2, y_max2])
    ylim2 = ax2.get_ylim()

    for (tv, av, mk, col, ms, pos, lbl) in marker_events:
        ax2.plot(tv, av, mk, color=col, ms=ms, mec='black', mew=1.0, zorder=8)
        y_off = 0.04 * (ylim2[1] - ylim2[0]) if pos == 'above' else -0.12 * (ylim2[1] - ylim2[0])
        ax2.annotate(lbl, xy=(tv, av), xytext=(tv + 0.25, av + y_off),
                     fontsize=8.5, color=col, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.75, edgecolor=col,
                               boxstyle='round,pad=0.25'),
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

    ax2.set_ylabel('Heartbeat Pulse Amplitude (mm)', **FONT_L)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.15)
    ax2.tick_params(labelbottom=False)

    # Dual x-axis: Cuff Pressure along top of amplitude panel
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    t_pticks = [t_start_phys, t_sbp_phys, t_map_rf_phys, t_dbp_phys, t_open_phys]
    p_pticks = [P_start, target_sbp, map_p_phase, target_dbp, P_full_open]
    ax2_top.set_xticks(t_pticks)
    ax2_top.set_xticklabels([f'{p:.0f} mmHg' for p in p_pticks], fontsize=9.5, color=C_CUFF)
    ax2_top.tick_params(axis='x', colors=C_CUFF)
    ax2_top.set_xlabel('Cuff Pressure (mmHg)', fontsize=11, color=C_CUFF, fontweight='bold', labelpad=6)

    # Comparison text box
    comp_txt = (
        f"Method Comparison (Physical Time Axis):\n"
        f"  RF Phase MAP :  {map_p_phase:.1f} mmHg  at t={t_map_rf_phys:.2f}s  amp={amp_map_phase:.3f} mm\n"
        f"  Acoustic MAP :  {map_p_acoustic:.1f} mmHg  at t={t_map_ac_phys:.2f}s  amp={amp_map_acou:.3f} mm\n"
        f"  MAP dt: {abs(t_map_rf_phys - t_map_ac_phys):.2f}s  |  MAP dP: {abs(map_p_phase - map_p_acoustic):.1f} mmHg\n"
        f"  Artery State : Fully occluded for 0.0s to {t_sbp_phys:.2f}s (reopens at SBP)"
    )
    ax2.text(0.01, 0.03, comp_txt, transform=ax2.transAxes, fontsize=9.0, fontweight='bold',
             color='#2C3E50', va='bottom', ha='left',
             bbox=dict(facecolor='#FDFEFE', alpha=0.92, edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))

    # ════════════════════════════════════════════════════════
    # PANEL 3 — Legend row
    # ════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    ph_patches = [
        mpatches.Patch(color='#AED6F1', alpha=0.8,
                       label=f'Phase I -- OCCLUDED: t=[0.0->{t_sbp_phys:.2f}s]  P=[0.0->{target_sbp:.0f} mmHg]  Artery fully closed, zero pulsatile amplitude'),
        mpatches.Patch(color='#FDEBD0', alpha=0.8,
                       label=f'Phase II -- KOROTKOFF: t=[{t_sbp_phys:.2f}->{t_dbp_phys:.2f}s]  P=[{target_sbp:.0f}->{target_dbp:.0f} mmHg]  Artery complying, large pulse oscillations'),
        mpatches.Patch(color='#D5F5E3', alpha=0.8,
                       label=f'Phase III -- UNOCCLUDED: t=[{t_dbp_phys:.2f}->{t_open_phys:.2f}s]  P=[{target_dbp:.0f}->{P_full_open:.0f} mmHg]  Artery fully open, stable flow amplitude'),
    ]
    ev_patches = [
        mpatches.Patch(color='crimson',   label=f'Peak Cuff Pressure ({P_start:.0f} mmHg) -- Artery Fully Occluded (~20s physical)'),
        mpatches.Patch(color='#C0392B',   label=f'SBP ({target_sbp:.0f} mmHg) -- Artery Begins to Reopen'),
        mpatches.Patch(color='#F39C12',   label=f'RF Phase MAP ({map_p_phase:.0f} mmHg at t={t_map_rf_phys:.2f}s)'),
        mpatches.Patch(color='#27AE60',   label=f'Acoustic MAP ({map_p_acoustic:.0f} mmHg at t={t_map_ac_phys:.2f}s)'),
        mpatches.Patch(color='#2471A3',   label=f'DBP ({target_dbp:.0f} mmHg) -- Artery Fully Open'),
        mpatches.Patch(color='#1ABC9C',   label=f'Full Open (~{P_full_open:.0f} mmHg at t={t_open_phys:.2f}s)'),
    ]
    ax3.legend(handles=ph_patches + ev_patches, loc='center', ncol=3,
               fontsize=9.0, framealpha=0.97, edgecolor='#BDC3C7',
               bbox_to_anchor=(0.5, 0.5))

    # ── X axis label on bottom of panel 2 ──
    ax2.set_xlabel('Physical Time (s)', **FONT_L)
    ax2.tick_params(labelbottom=True)

    out_img = os.path.join(SUMMARY_DIR, "rf_cuff_vs_amplitude_concept.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] {out_img}")
    print(f"  Figure dimensions: 20 x 13 inches  |  DPI: 300  |  ~6000 x 3900 px")

if __name__ == '__main__':
    main()
