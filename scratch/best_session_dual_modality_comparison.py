"""
Premium Dual-Modality Best-Session Comparison Figure
======================================================
Produces a 3-panel clinical publication-grade comparison figure at 300 DPI
with a clean white background.

All physiological waveforms are normalized to [0, 1] as requested to enable
clear visual comparison of shapes, timings, and physiological transitions.
"""

import h5py
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'best_session_dual_modality_comparison.png')

SUB1_RF   = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_2.h5')
SUB1_WAV  = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec02.wav')
SUB2_RF   = os.path.join(BASE, 'Sub_2_Rajveer',  'Rec_2.h5')
SUB2_WAV  = os.path.join(BASE, 'Sub_2_Rajveer',  'sthethoscope_rec02.wav')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Enforce post-inflation Korotkoff windows (starts after 20s)
TARGET_DUR_S = 17.5
STETH_OFFSET = 3.5

# Premium Color Palette for white background
C_SUB1      = '#005F73'  # Deep Blue-Green
C_SUB2      = '#CA6702'  # Rust Orange
C_STETH     = '#0A9396'  # Teal
C_RF        = '#AE2012'  # Crimson
C_GRID      = '#E5E5E5'  # Very light gray for grid lines
C_TEXT      = '#222222'  # Dark charcoal text
C_HIGHLIGHT = '#E9D8A6'  # Muted gold shading for Korotkoff windows

# ── PROCESSING HELPERS ─────────────────────────────────────────────
def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    R = np.sqrt(xc**2 + yc**2 - c)
    return xc, yc, R

def iq_condition_circle(i_raw, q_raw):
    xc, yc, R = fit_circle(i_raw, q_raw)
    return i_raw - xc, q_raw - yc, xc, yc, R

def robust_phase(i_c, q_c):
    iq = i_c + 1j * q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3 * iqr, 0.01)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return signal.detrend(phase, type='linear')

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, win):
    return np.sqrt(np.maximum(smooth(x**2, win), 1e-20))

def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin + 1e-20)

def detect_deflation_onset_rf(vk, t, lo=18.0, hi=35.0, fb=20.0):
    sl, sh = int(lo*FS_RF), int(min(hi*FS_RF, len(vk)))
    if sh <= sl+FS_RF: return fb
    tr = smooth(np.abs(vk), int(FS_RF*2))
    dt = np.diff(tr[sl:sh])
    dts = smooth(np.abs(dt), max(1, int(FS_RF*0.5)))
    if dts.max() < 1e-12: return fb
    td = t[sl + np.argmax(dts)]
    return float(td) if lo<=td<=hi else fb

def detect_deflation_onset_st(mag, t, fs_aud, lo=18.0, hi=35.0, fb=20.0):
    sl = int(lo * fs_aud)
    sh = int(min(hi * fs_aud, len(mag)))
    if sh <= sl + int(fs_aud): return fb
    trend = smooth(np.abs(mag), int(fs_aud * 2.0))
    dt = np.diff(trend[sl:sh])
    dts = smooth(np.abs(dt), max(1, int(fs_aud * 0.5)))
    if dts.max() < 1e-12: return fb
    td = t[sl + np.argmax(dts)]
    return float(td) if lo<=td<=hi else fb

# ── LOAD & PROCESS MODALITIES ──────────────────────────────────────
def process_subject(rf_path, wav_path, subj_name):
    print(f"Processing {subj_name}...")
    
    # 1. Process RF
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
    N_rf = len(i_raw)
    t_rf = np.arange(N_rf) / FS_RF
    
    # Circle fit and center
    i_c, q_c, xc_fit, yc_fit, R_fit = iq_condition_circle(i_raw, q_raw)
    phi = robust_phase(i_c, q_c)
    
    # Korotkoff Velocity (10-50 Hz)
    sos_vk = butter(4, [10, 50], btype='band', fs=FS_RF, output='sos')
    pk = sosfiltfilt(sos_vk, phi)
    vk = np.append(np.diff(pk) * FS_RF, 0) * SCALE  # mm/s
    
    # Heartbeat Displacement (0.4-3 Hz)
    sos_dh = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh = sosfiltfilt(sos_dh, phi) * SCALE  # mm
    
    # Adaptive RF Window (Starts post-inflation >20s)
    defl_rf = detect_deflation_onset_rf(vk, t_rf)
    koro_on_rf = max(defl_rf + STETH_OFFSET, 20.0)
    koro_off_rf = min(koro_on_rf + TARGET_DUR_S, t_rf[-1] - 2.0)
    
    # Active vs Baseline RF PSD
    mask_k_rf = (t_rf >= koro_on_rf) & (t_rf <= koro_off_rf)
    mask_b_rf = (t_rf >= t_rf[-1] - 7.0) & (t_rf <= t_rf[-1] - 2.0)
    
    f_psd_rf, p_psd_rf = welch(vk[mask_k_rf], fs=FS_RF, nperseg=min(len(vk[mask_k_rf]), int(FS_RF*2)))
    _, p_base_rf = welch(vk[mask_b_rf], fs=FS_RF, nperseg=min(len(vk[mask_b_rf]), int(FS_RF*2)))
    
    # RF Envelopes
    env_rf = sliding_rms(vk, int(FS_RF*0.5))
    
    # 2. Process Stethoscope
    fs_aud, audio_stereo = wav.read(wav_path)
    audio = audio_stereo[:, 0].astype(np.float32) # Take channel 1
    # Downsample audio to speed up and match time resolution
    ds_factor = 4
    audio_ds = signal.decimate(audio, ds_factor)
    fs_aud_ds = fs_aud // ds_factor
    N_aud = len(audio_ds)
    t_aud = np.arange(N_aud) / fs_aud_ds
    
    # Bandpass filter Stethoscope (50 - 1000 Hz)
    sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
    ka = sosfiltfilt(sos_aud, audio_ds)
    
    # Adaptive Stethoscope Window
    defl_st = detect_deflation_onset_st(ka, t_aud, fs_aud_ds)
    koro_on_st = max(defl_st + STETH_OFFSET, 20.0)
    koro_off_st = min(koro_on_st + TARGET_DUR_S, t_aud[-1] - 2.0)
    
    # Stethoscope Envelopes
    env_st = sliding_rms(ka, int(fs_aud_ds*0.5))
    
    # Active vs Baseline Stethoscope PSD
    mask_k_st = (t_aud >= koro_on_st) & (t_aud <= koro_off_st)
    mask_b_st = (t_aud >= t_aud[-1] - 7.0) & (t_aud <= t_aud[-1] - 2.0)
    f_psd_st, p_psd_st = welch(ka[mask_k_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_k_st]), int(fs_aud_ds*2)))
    _, p_base_st = welch(ka[mask_b_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_b_st]), int(fs_aud_ds*2)))
    
    # 3. Aligned Modality Cross-Correlation
    # Resample envelopes to 100 Hz for lag computation
    target_fs = 100
    env_rf_res = signal.resample_poly(env_rf, target_fs, FS_RF)
    env_st_res = signal.resample_poly(env_st, target_fs, fs_aud_ds)
    
    # Trim to common length
    min_len = min(len(env_rf_res), len(env_st_res))
    e_rf = env_rf_res[:min_len]
    e_st = env_st_res[:min_len]
    
    e_rf_norm = (e_rf - np.mean(e_rf)) / (np.std(e_rf) + 1e-20)
    e_st_norm = (e_st - np.mean(e_st)) / (np.std(e_st) + 1e-20)
    
    corr = np.correlate(e_rf_norm, e_st_norm, mode='full')
    lags = np.arange(-min_len + 1, min_len) / target_fs
    best_lag = lags[np.argmax(corr)]
    
    t_aud_aligned = t_aud + best_lag
    
    return {
        't_rf': t_rf, 'phi': phi, 'vk': vk, 'dh': dh,
        'k_on_rf': koro_on_rf, 'k_off_rf': koro_off_rf, 'defl_rf': defl_rf,
        'f_psd_rf': f_psd_rf, 'p_psd_rf': p_psd_rf, 'p_base_rf': p_base_rf,
        'env_rf': env_rf,
        't_aud': t_aud, 't_aud_aligned': t_aud_aligned, 'ka': ka,
        'k_on_st': koro_on_st, 'k_off_st': koro_off_st, 'defl_st': defl_st,
        'f_psd_st': f_psd_st, 'p_psd_st': p_psd_st, 'p_base_st': p_base_st,
        'env_st': env_st,
        'lags': lags, 'corr': corr / np.max(corr), 'best_lag': best_lag
    }

sub1 = process_subject(SUB1_RF, SUB1_WAV, "Sub_1 (Prof. Kan)")
sub2 = process_subject(SUB2_RF, SUB2_WAV, "Sub_2 (Rajveer)")

# ── PLOT DUAL-MODALITY PUBLICATION FIGURE ──────────────────────────
print("Generating premium 3-panel clinical comparison figure...")
fig = plt.figure(figsize=(18, 22), dpi=300)
fig.patch.set_facecolor('#ffffff')

# GridSpec with 3 rows corresponding to the 3 panels
gs = gridspec.GridSpec(3, 3, figure=fig,
                       height_ratios=[1.0, 1.0, 1.0],
                       hspace=0.38, wspace=0.28,
                       left=0.06, right=0.96,
                       top=0.94, bottom=0.05)

# Styling Helper for white publication theme
def style_ax_pub(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color=C_TEXT, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=10, labelpad=4)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=10, labelpad=4)
    ax.tick_params(colors=C_TEXT, labelsize=9, length=4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(0.8)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9, ls='-')
    return ax

TBOX = dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.9, lw=0.8)

# ===================================================================
# PANEL 1: RF RADAR RADIOMYOGRAPHY (RMG) ANALYSIS (ROW 0)
# ===================================================================
print("  Plotting Panel 1: RF Radar RMG Analysis (Normalized)...")

# Subplot 1A: Sub 1 RF Continuous Unwrapped Phase (Normalized)
ax1A = fig.add_subplot(gs[0, 0])
style_ax_pub(ax1A, "Sub 1: Continuous Unwrapped Phase", "Time (s)", "Normalized Phase Amplitude (a.u.)")
ds1 = max(1, len(sub1['t_rf'])//4000)
t_p1 = sub1['t_rf'][::ds1]
phi_n1 = normalize(sub1['phi'])[::ds1]
ax1A.plot(t_p1, phi_n1, color=C_SUB1, lw=1.0, label=r'Sub 1 Unwrapped Phase $\phi(t)$')
ax1A.axvspan(sub1['k_on_rf'], sub1['k_off_rf'], color=C_HIGHLIGHT, alpha=0.25, 
             label=f"Korotkoff Window\n({sub1['k_on_rf']:.1f}-{sub1['k_off_rf']:.1f} s)")
ax1A.axvline(sub1['defl_rf'], color='#E63946', ls=':', lw=1.5, label='Deflation Onset')
ax1A.set_xlim([10, 52])
ax1A.set_ylim([-0.05, 1.05])
ax1A.legend(fontsize=8, framealpha=0.9, loc='upper right')

# Subplot 1B: Sub 2 RF Continuous Unwrapped Phase (Normalized)
ax1B = fig.add_subplot(gs[0, 1])
style_ax_pub(ax1B, "Sub 2: Continuous Unwrapped Phase", "Time (s)", "Normalized Phase Amplitude (a.u.)")
ds2 = max(1, len(sub2['t_rf'])//4000)
t_p2 = sub2['t_rf'][::ds2]
phi_n2 = normalize(sub2['phi'])[::ds2]
ax1B.plot(t_p2, phi_n2, color=C_SUB2, lw=1.0, label=r'Sub 2 Unwrapped Phase $\phi(t)$')
ax1B.axvspan(sub2['k_on_rf'], sub2['k_off_rf'], color=C_HIGHLIGHT, alpha=0.25, 
             label=f"Korotkoff Window\n({sub2['k_on_rf']:.1f}-{sub2['k_off_rf']:.1f} s)")
ax1B.axvline(sub2['defl_rf'], color='#E63946', ls=':', lw=1.5, label='Deflation Onset')
ax1B.set_xlim([10, 52])
ax1B.set_ylim([-0.05, 1.05])
ax1B.legend(fontsize=8, framealpha=0.9, loc='upper right')

# Subplot 1C: RF Micro-Vibrations & Heartbeat Waves (Normalized)
ax1C = fig.add_subplot(gs[0, 2])
style_ax_pub(ax1C, "RF Physiological Waveforms (Normalized)", "Time (s)", "Normalized Amplitude (a.u.)")
# Sub 1 Normalized Waves
vk_n1 = normalize(sub1['vk'])[::ds1]
dh_n1 = normalize(sub1['dh'])[::ds1]
ax1C.plot(t_p1, vk_n1, color=C_SUB1, lw=0.5, alpha=0.85, label='Sub 1 $v_k(t)$ (Micro-velocity)')
ax1C.plot(t_p1, dh_n1 - 0.25, color='#83C5BE', lw=0.8, label='Sub 1 $d(t)$ (Heartbeat Displacement)')

# Sub 2 Normalized Waves
vk_n2 = normalize(sub2['vk'])[::ds2]
dh_n2 = normalize(sub2['dh'])[::ds2]
ax1C.plot(t_p2, vk_n2 + 0.3, color=C_SUB2, lw=0.5, alpha=0.85, label='Sub 2 $v_k(t)$ (Micro-velocity)')
ax1C.plot(t_p2, dh_n2 + 0.05, color='#E9D8A6', lw=0.8, label='Sub 2 $d(t)$ (Heartbeat Displacement)')

ax1C.set_xlim([15, 50])
ax1C.set_ylim([-0.3, 1.45])
ax1C.legend(fontsize=7, framealpha=0.9, loc='upper right')

# Add text labels for Panel 1
fig.text(0.02, 0.95, "PANEL 1: RF Radar Radiomyography (RMG) Phase & Extracted Vitals (Normalized)", 
         color=C_TEXT, fontsize=14, fontweight='bold')


# ===================================================================
# PANEL 2: STETHOSCOPE ACOUSTIC PRESSURE ANALYSIS (ROW 1)
# ===================================================================
print("  Plotting Panel 2: Stethoscope Acoustic Analysis (Normalized)...")

# Subplot 2A: Sub 1 Stethoscope filtered audio & RMS envelope (Normalized)
ax2A = fig.add_subplot(gs[1, 0])
style_ax_pub(ax2A, "Sub 1: Filtered Acoustic Audio (50-1000 Hz)", "Time (s)", "Normalized Acoustic Pressure (a.u.)")
ds_aud1 = max(1, len(sub1['t_aud'])//6000)
t_aud1 = sub1['t_aud'][::ds_aud1]
ka_n1 = normalize(sub1['ka'])[::ds_aud1]
env_n1 = normalize(sub1['env_st'])[::ds_aud1]
ax2A.plot(t_aud1, ka_n1, color='#A2D2FF', lw=0.4, alpha=0.7, label='Filtered Audio')
ax2A.plot(t_aud1, env_n1, color=C_SUB1, lw=1.2, label='RMS Envelope')
ax2A.axvspan(sub1['k_on_st'], sub1['k_off_st'], color=C_HIGHLIGHT, alpha=0.25, 
             label=f"Korotkoff Window\n({sub1['k_on_st']:.1f}-{sub1['k_off_st']:.1f} s)")
ax2A.axvline(sub1['defl_st'], color='#E63946', ls=':', lw=1.5, label='Deflation Onset')
ax2A.set_xlim([10, 52])
ax2A.set_ylim([-0.05, 1.05])
ax2A.legend(fontsize=8, framealpha=0.9, loc='upper right')

# Subplot 2B: Sub 2 Stethoscope filtered audio & RMS envelope (Normalized)
ax2B = fig.add_subplot(gs[1, 1])
style_ax_pub(ax2B, "Sub 2: Filtered Acoustic Audio (50-1000 Hz)", "Time (s)", "Normalized Acoustic Pressure (a.u.)")
ds_aud2 = max(1, len(sub2['t_aud'])//6000)
t_aud2 = sub2['t_aud'][::ds_aud2]
ka_n2 = normalize(sub2['ka'])[::ds_aud2]
env_n2 = normalize(sub2['env_st'])[::ds_aud2]
ax2B.plot(t_aud2, ka_n2, color='#FAD2E1', lw=0.4, alpha=0.7, label='Filtered Audio')
ax2B.plot(t_aud2, env_n2, color=C_SUB2, lw=1.2, label='RMS Envelope')
ax2B.axvspan(sub2['k_on_st'], sub2['k_off_st'], color=C_HIGHLIGHT, alpha=0.25, 
             label=f"Korotkoff Window\n({sub2['k_on_st']:.1f}-{sub2['k_off_st']:.1f} s)")
ax2B.axvline(sub2['defl_st'], color='#E63946', ls=':', lw=1.5, label='Deflation Onset')
ax2B.set_xlim([10, 52])
ax2B.set_ylim([-0.05, 1.05])
ax2B.legend(fontsize=8, framealpha=0.9, loc='upper right')

# Subplot 2C: Stethoscope PSD Curves (Both Subjects - Normalized)
ax2C = fig.add_subplot(gs[1, 2])
style_ax_pub(ax2C, "Acoustic Power Spectral Density (Normalized)", "Frequency (Hz)", "Normalized PSD (a.u.)")
# Sub 1 Normalized PSD
fm1 = sub1['f_psd_st'] <= 600
p_psd_st_norm1 = normalize(10*np.log10(sub1['p_psd_st'] + 1e-20))
p_base_st_norm1 = normalize(10*np.log10(sub1['p_base_st'] + 1e-20))
ax2C.plot(sub1['f_psd_st'][fm1], p_psd_st_norm1[fm1], 
          color=C_SUB1, lw=1.5, label='Sub 1 (Korotkoff Window)')
ax2C.plot(sub1['f_psd_st'][fm1], p_base_st_norm1[fm1], 
          color=C_SUB1, ls='--', lw=1.0, alpha=0.6, label='Sub 1 (Quiet Baseline)')
# Sub 2 Normalized PSD
fm2 = sub2['f_psd_st'] <= 600
p_psd_st_norm2 = normalize(10*np.log10(sub2['p_psd_st'] + 1e-20))
p_base_st_norm2 = normalize(10*np.log10(sub2['p_base_st'] + 1e-20))
ax2C.plot(sub2['f_psd_st'][fm2], p_psd_st_norm2[fm2], 
          color=C_SUB2, lw=1.5, label='Sub 2 (Korotkoff Window)')
ax2C.plot(sub2['f_psd_st'][fm2], p_base_st_norm2[fm2], 
          color=C_SUB2, ls='--', lw=1.0, alpha=0.6, label='Sub 2 (Quiet Baseline)')

ax2C.set_xlim([0, 500])
ax2C.set_ylim([-0.05, 1.05])
ax2C.legend(fontsize=7, framealpha=0.9, loc='upper right')

# Add text labels for Panel 2
fig.text(0.02, 0.64, "PANEL 2: Gold-Standard Acoustic Stethoscope Validation & Spectral Analysis (Normalized)", 
         color=C_TEXT, fontsize=14, fontweight='bold')


# ===================================================================
# PANEL 3: CROSS-MODALITY COMPARISON & CONSENSUS VALIDATION (ROW 2)
# ===================================================================
print("  Plotting Panel 3: Cross-Modality Alignment (Normalized)...")

# Subplot 3A: Sub 1 Overlaid aligned RF vs Stethoscope envelopes (Normalized)
ax3A = fig.add_subplot(gs[2, 0])
style_ax_pub(ax3A, "Sub 1: Aligned Modalities (Consensus)", "Time (s)", "Normalized Envelope Amplitude (a.u.)")
env_rf_n1 = normalize(sub1['env_rf'])
env_st_n1 = normalize(sub1['env_st'])

ds_rf1 = max(1, len(sub1['t_rf'])//4000)
ds_st1 = max(1, len(sub1['t_aud'])//4000)

ax3A.plot(sub1['t_rf'][::ds_rf1], env_rf_n1[::ds_rf1], color=C_RF, lw=1.2, label='RF RMG Envelope')
ax3A.plot(sub1['t_aud_aligned'][::ds_st1], env_st_n1[::ds_st1], color=C_STETH, lw=1.0, alpha=0.85, label='Acoustic Envelope')
ax3A.axvspan(sub1['k_on_rf'], sub1['k_off_rf'], color=C_HIGHLIGHT, alpha=0.25, label='Consensus Window')
ax3A.set_xlim([15, 52])
ax3A.set_ylim([-0.05, 1.05])
ax3A.legend(fontsize=8, framealpha=0.9, loc='upper right')
ax3A.text(0.02, 0.95, f"Lag = {sub1['best_lag']:.2f} s\nAligned Overlap", transform=ax3A.transAxes, 
          fontsize=8.5, va='top', bbox=TBOX)

# Subplot 3B: Sub 2 Overlaid aligned RF vs Stethoscope envelopes (Normalized)
ax3B = fig.add_subplot(gs[2, 1])
style_ax_pub(ax3B, "Sub 2: Aligned Modalities (Consensus)", "Time (s)", "Normalized Envelope Amplitude (a.u.)")
env_rf_n2 = normalize(sub2['env_rf'])
env_st_n2 = normalize(sub2['env_st'])

ds_rf2 = max(1, len(sub2['t_rf'])//4000)
ds_st2 = max(1, len(sub2['t_aud'])//4000)

ax3B.plot(sub2['t_rf'][::ds_rf2], env_rf_n2[::ds_rf2], color=C_RF, lw=1.2, label='RF RMG Envelope')
ax3B.plot(sub2['t_aud_aligned'][::ds_st2], env_st_n2[::ds_st2], color=C_STETH, lw=1.0, alpha=0.85, label='Acoustic Envelope')
ax3B.axvspan(sub2['k_on_rf'], sub2['k_off_rf'], color=C_HIGHLIGHT, alpha=0.25, label='Consensus Window')
ax3B.set_xlim([15, 52])
ax3B.set_ylim([-0.05, 1.05])
ax3B.legend(fontsize=8, framealpha=0.9, loc='upper right')
ax3B.text(0.02, 0.95, f"Lag = {sub2['best_lag']:.2f} s\nAligned Overlap", transform=ax3B.transAxes, 
          fontsize=8.5, va='top', bbox=TBOX)

# Subplot 3C: Cross-Correlation Lag Curve (Both Subjects)
ax3C = fig.add_subplot(gs[2, 2])
style_ax_pub(ax3C, "Envelope Cross-Correlation Lag Curve", "Lag Time (s)", "Cross-Correlation R (Normalised)")
ax3C.plot(sub1['lags'], sub1['corr'], color=C_SUB1, lw=1.8, label=f"Sub 1 (Peak Lag = {sub1['best_lag']:.2f} s)")
ax3C.plot(sub2['lags'], sub2['corr'], color=C_SUB2, lw=1.8, label=f"Sub 2 (Peak Lag = {sub2['best_lag']:.2f} s)")
ax3C.axvline(0, color='#666666', ls='--', lw=0.8)
ax3C.axvline(sub1['best_lag'], color=C_SUB1, ls=':', lw=1.2)
ax3C.axvline(sub2['best_lag'], color=C_SUB2, ls=':', lw=1.2)
ax3C.set_xlim([-10, 10])
ax3C.set_ylim([-0.05, 1.05])
ax3C.legend(fontsize=8, framealpha=0.9, loc='upper right')

# Add text labels for Panel 3
fig.text(0.02, 0.33, "PANEL 3: Cross-Modality Temporal Alignment & Physiological Consensus Validation", 
         color=C_TEXT, fontsize=14, fontweight='bold')

# Figure Title
fig.suptitle("Clinical RMG Radar vs. Acoustic Stethoscope: Best-Session Comparative Validation\n"
             "USRP B210 @ 0.9 GHz Carrier  |  Clinical 10–50 Hz RMG Band  |  100% Normalized Amplitude y-Axes  |  White Background",
             color=C_TEXT, fontsize=15, fontweight='bold', y=0.985)

# Save
plt.savefig(OUT, dpi=300, facecolor='#ffffff', bbox_inches='tight')
print(f"Premium Comparison Dashboard successfully saved to: {OUT}")
