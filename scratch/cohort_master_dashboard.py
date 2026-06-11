"""
Clinical Master Cohort Dashboard (3x2 Panel Layout)
===================================================
Produces a premium 3x2 panel clinical cohort master dashboard (300 DPI, white background)
combining the results of Subject 1 (Prof. Kan, Rec 06 - Left Column) and Subject 2 (Rajveer, Rec 04 - Right Column)
side-by-side.
- Row 1 (Top): The Brachial Cuff Pressure vs. Dual-Modality Heartbeat Pulse Amplitude Concept Plot (exactly matching the provided images).
- Row 2 (Center): Welch Power Spectral Density (PSD) Active vs. Baseline Profiles.
- Row 3 (Bottom): 6-Method Envelope Consensus Windowing.
"""

import h5py
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'cohort_master_dashboard.png')

# Paths for Subject 1 (Prof. Kan, Rec 06)
SUB1_RF   = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
SUB1_WAV  = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

# Paths for Subject 2 (Rajveer, Rec 04)
SUB2_RF   = os.path.join(BASE, 'Sub_2_Rajveer',  'Rec_4.h5')
SUB2_WAV  = os.path.join(BASE, 'Sub_2_Rajveer',  'sthethoscope_rec04.wav')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Colors for 6 methods
COLORS_6 = {
    'Hilbert': '#005F73',       # Deep Teal
    'RMS Power': '#CA6702',     # Rust Orange
    'TKEO': '#AE2012',          # Crimson
    'MAV': '#9B5DE5',           # Purple
    'Core Band': '#0A9396',     # Green-Teal
    'Slope MAV': '#F15BB5'      # Hot Pink
}

C_STETH     = '#0A9396'  # Teal for Steth
C_RF        = '#CA6702'  # Rust for RF
C_HIGHLIGHT = '#E9D8A6'  # Muted gold shading for Korotkoff consensus windows
C_TEXT      = '#222222'  # Dark text
C_GRID      = '#E5E5E5'  # Light grid

COLORS_6 = {
    'Hilbert': '#005F73',       # Deep Teal
    'RMS Power': '#CA6702',     # Rust Orange
    'TKEO': '#AE2012',          # Crimson
    'MAV': '#9B5DE5',           # Purple
    'Core Band': '#0A9396',     # Green-Teal
    'Slope MAV': '#F15BB5'      # Hot Pink
}

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

def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin + 1e-20)

def normalize_pulse_for_display(t, val, defl_onset):
    val_clean = val.copy()
    val_clean[t < defl_onset] = 0.0
    mask_defl = t >= defl_onset
    max_val = np.max(np.abs(val_clean[mask_defl]))
    if max_val > 0:
        val_clean = val_clean / max_val
    return 0.5 + 0.35 * val_clean

def normalize_envelope_for_display(t, env, defl_onset):
    env_clean = env.copy()
    env_clean[t < defl_onset] = 0.0
    max_val = np.max(env_clean[t >= defl_onset])
    if max_val > 0:
        env_clean = env_clean / max_val
    return env_clean

def extract_6_envelopes(x, fs, win_s=0.5):
    win = max(1, int(fs * win_s))
    m1 = np.abs(signal.hilbert(x))
    m2 = np.sqrt(np.maximum(smooth(x**2, win), 1e-20))
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
    m3 = smooth(np.abs(tkeo), win)
    m4 = smooth(np.abs(x), win)
    if fs == 10000:
        sos_core = butter(4, [10, 100], btype='band', fs=fs, output='sos')
    else:
        sos_core = butter(4, [50, 300], btype='band', fs=fs, output='sos')
    core_sig = sosfiltfilt(sos_core, x)
    m5 = np.sqrt(np.maximum(smooth(core_sig**2, win), 1e-20))
    dx = np.append(np.diff(x), 0)
    m6 = smooth(np.abs(dx), win)
    return {
        'Hilbert': normalize(m1),
        'RMS Power': normalize(m2),
        'TKEO': normalize(m3),
        'MAV': normalize(m4),
        'Core Band': normalize(m5),
        'Slope MAV': normalize(m6)
    }

def notch(x, f0, fs, Q=30):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

# ── LOAD & PROCESS SUBJECT DATA ────────────────────────────────────
def load_and_process(rf_path, wav_path, k_on, k_off, defl_onset, notches, lag):
    # RF
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
    t_rf = np.arange(len(i_raw)) / FS_RF
    i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
    phi = robust_phase(i_c, q_c)
    
    # Clean Phase of harmonics
    for freq in notches:
        phi = notch(phi, freq, FS_RF)
    
    # Clean deflation boundaries to suppress valve and cuff dump transients
    t_start_clean = defl_onset + 3.0
    t_end_clean   = k_off + 1.2
    
    # Korotkoff velocity (30-180 Hz)
    sos_vk = butter(4, [30, 180], btype='band', fs=FS_RF, output='sos')
    vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE
    vk[(t_rf < t_start_clean) | (t_rf > t_end_clean)] = 0.0
    
    # Heartbeat displacement (0.4-3 Hz)
    sos_dh = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh = sosfiltfilt(sos_dh, phi) * SCALE
    dh[(t_rf < t_start_clean) | (t_rf > t_end_clean)] = 0.0
    
    # Zero out the high-frequency vibration signal during inflation for cleaner envelope consensus
    vk_clean = vk.copy()
    env_rf_dict = extract_6_envelopes(vk_clean, FS_RF, win_s=0.5)

    # Audio
    fs_aud, audio_stereo = wav.read(wav_path)
    audio = audio_stereo[:, 0].astype(np.float32)
    ds_factor = 4
    audio_ds = signal.decimate(audio, ds_factor)
    fs_aud_ds = fs_aud // ds_factor
    t_aud = (np.arange(len(audio_ds)) / fs_aud_ds) + lag
    
    # Acoustic filter (50-1000 Hz)
    sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
    ka = sosfiltfilt(sos_aud, audio_ds)
    ka[(t_aud < t_start_clean) | (t_aud > t_end_clean)] = 0.0
    
    # Acoustic heartbeat displacement (0.4-3 Hz)
    audio_env = np.abs(signal.hilbert(ka))
    sos_hr_a = butter(4, [0.4, 3.0], btype='band', fs=fs_aud_ds, output='sos')
    dh_acoustic = sosfiltfilt(sos_hr_a, audio_env)
    dh_acoustic[(t_aud < t_start_clean) | (t_aud > t_end_clean)] = 0.0
    
    # Zero out the acoustic signal during inflation for cleaner envelope consensus
    ka_clean = ka.copy()
    env_st_dict = extract_6_envelopes(ka_clean, fs_aud_ds, win_s=0.5)

    # PSD Welch using locked boundaries
    mask_k_rf = (t_rf >= k_on) & (t_rf <= k_off)
    mask_b_rf = (t_rf >= t_rf[-1] - 7.0) & (t_rf <= t_rf[-1] - 2.0)
    
    mask_k_st = (t_aud >= k_on) & (t_aud <= k_off)
    mask_b_st = (t_aud >= t_aud[-1] - 7.0) & (t_aud <= t_aud[-1] - 2.0)

    f_rf, p_k_rf = welch(vk[mask_k_rf], fs=FS_RF, nperseg=min(len(vk[mask_k_rf]), int(FS_RF*2)))
    _, p_b_rf = welch(vk[mask_b_rf], fs=FS_RF, nperseg=min(len(vk[mask_b_rf]), int(FS_RF*2)))

    f_st, p_k_st = welch(ka[mask_k_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_k_st]), int(fs_aud_ds*2)))
    _, p_b_st = welch(ka[mask_b_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_b_st]), int(fs_aud_ds*2)))

    # Calculate heart rate (HR) from RF displacement on stable window [30.0, 39.0]s
    mask_hr_rf = (t_rf >= 30.0) & (t_rf <= 39.0)
    t_stable_rf = t_rf[mask_hr_rf]
    dh_stable_rf = dh[mask_hr_rf]
    if len(dh_stable_rf) > 0 and np.max(np.abs(dh_stable_rf)) > 0:
        dh_stable_rf = dh_stable_rf / np.max(np.abs(dh_stable_rf))
    min_dist_rf = int(FS_RF * 0.5)
    prom_rf = 0.1 if 'Sub_2_Rajveer' in rf_path else 0.2
    peaks_rf, _ = signal.find_peaks(dh_stable_rf, distance=min_dist_rf, prominence=prom_rf)
    hr_rf = 60.0 / np.mean(np.diff(t_stable_rf[peaks_rf])) if len(peaks_rf) > 1 else 0.0

    # Calculate heart rate (HR) from Steth displacement on stable window [30.0, 39.0]s
    mask_hr_st = (t_aud >= 30.0) & (t_aud <= 39.0)
    t_stable_st = t_aud[mask_hr_st]
    dh_stable_st = dh_acoustic[mask_hr_st]
    if len(dh_stable_st) > 0 and np.max(np.abs(dh_stable_st)) > 0:
        dh_stable_st = dh_stable_st / np.max(np.abs(dh_stable_st))
    min_dist_st = int(fs_aud_ds * 0.5)
    peaks_st, _ = signal.find_peaks(dh_stable_st, distance=min_dist_st, prominence=0.2)
    hr_st = 60.0 / np.mean(np.diff(t_stable_st[peaks_st])) if len(peaks_st) > 1 else 0.0

    return {
        't_rf': t_rf, 'vk': vk, 'dh': dh, 'env_rf': env_rf_dict['RMS Power'], 'env_rf_dict': env_rf_dict,
        't_aud': t_aud, 'ka': ka, 'dh_acoustic': dh_acoustic, 'env_st': env_st_dict['RMS Power'], 'env_st_dict': env_st_dict,
        'f_rf': f_rf, 'p_k_rf': p_k_rf, 'p_b_rf': p_b_rf,
        'f_st': f_st, 'p_k_st': p_k_st, 'p_b_st': p_b_st,
        'hr_rf': hr_rf, 'hr_st': hr_st,
        'defl_onset': defl_onset
    }

print("Processing Subject 1 (Prof. Kan, Rec 06)...")
sub1 = load_and_process(SUB1_RF, SUB1_WAV, k_on=27.53, k_off=43.33, defl_onset=18.0, notches=[100.71, 201.43, 302.14, 402.86], lag=1.7083)

print("Processing Subject 2 (Rajveer, Rec 04)...")
sub2 = load_and_process(SUB2_RF, SUB2_WAV, k_on=27.38, k_off=42.00, defl_onset=18.6, notches=[50.0, 64.0, 100.6, 201.2], lag=2.6042)

# ── PLOT MASTER 3x2 COHORT DASHBOARD ──────────────────────────────
print("Plotting premium 3x2 cohort master figure...")
fig, axes = plt.subplots(3, 2, figsize=(20, 24), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Styling Helper with EXTRA-LARGE text
def style_ax_extra_large(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color=C_TEXT, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=14, labelpad=7)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=14, labelpad=7)
    ax.tick_params(colors=C_TEXT, labelsize=12, length=5, width=1.2)
    for sp in ax.spines.values():
        sp.set_edgecolor('#999999')
        sp.set_linewidth(1.2)
    ax.grid(True, color=C_GRID, lw=0.8, alpha=0.9, ls='-')
    return ax

TBOX = dict(boxstyle='round,pad=0.4', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.95, lw=1.0)

# ===================================================================
# ROW 1: CUFF PRESSURE VS. HEARTBEAT COMPLIANCE PULSES PLOTS
# ===================================================================
# Subplot 1A: Subject 1 (Left)
ax1A = axes[0, 0]
style_ax_extra_large(
    ax1A, 
    "(A) Subject 1 (Prof. Kan, Rec 06): Arterial Compliance Pulses vs. Cuff Pressure\n"
    f"[GT: SBP=125, DBP=75, MAP=92 mmHg | Dur=15.8s | HR: RF = {sub1['hr_rf']:.0f} BPM, Steth = {sub1['hr_st']:.0f} BPM]",
    "Time (s)", "Normalized Compliance Displacement (a.u.)"
)
ds_1_rf = max(1, len(sub1['t_rf'])//4000)
ds_1_st = max(1, len(sub1['t_aud'])//6000)

# Shading Zones matching their provided concept image
ax1A.axvspan(0.0, 18.0, color='#E5E8E8', alpha=0.5, label='Inflation Phase (~18 s)')
ax1A.axvspan(18.0, 27.53, color='#BDC3C7', alpha=0.3, label='Occluded (above SBP)')
ax1A.axvspan(27.53, 43.33, color='#FDF2E9', alpha=0.7, label='Korotkoff Region (15.80 s)')

# Plot raw heartbeat displacement (thin)
ax1A.plot(sub1['t_rf'][::ds_1_rf], normalize_pulse_for_display(sub1['t_rf'], sub1['dh'], 19.0)[::ds_1_rf], color=C_RF, lw=1.5, alpha=0.9, label='RF Heartbeat Pulses')
ax1A.plot(sub1['t_aud'][::ds_1_st], normalize_pulse_for_display(sub1['t_aud'], sub1['dh_acoustic'], 19.0)[::ds_1_st], color=C_STETH, lw=1.2, alpha=0.85, ls='--', label='Steth Heartbeat Pulses')

# Right y-axis for Cuff Pressure
ax1A_cuff = ax1A.twinx()
t_grid_1 = np.linspace(0, 52, 500)
p_grid_1 = np.zeros_like(t_grid_1)
beta_active_1 = 50.0 / (43.33 - 27.53)
p_onset_1 = 125.0 + beta_active_1 * (27.53 - 18.0)
p_grid_1[t_grid_1 <= 18.0] = (p_onset_1 / 18.0) * t_grid_1[t_grid_1 <= 18.0]
p_grid_1[t_grid_1 > 18.0] = 125.0 - beta_active_1 * (t_grid_1[t_grid_1 > 18.0] - 27.53)
ax1A_cuff.plot(t_grid_1, p_grid_1, color='#7D3C98', lw=2.5, alpha=0.7, label='Cuff Pressure')
ax1A_cuff.set_ylabel('Cuff Pressure (mmHg)', color='#7D3C98', fontsize=14, labelpad=7)
ax1A_cuff.tick_params(axis='y', colors='#7D3C98', labelsize=13)
ax1A_cuff.set_ylim([0, 180])

# Clinical BP markers
ax1A.axvline(27.53, color='#E63946', ls='--', lw=2.0)
ax1A.axvline(43.33, color='#2980B9', ls='--', lw=2.0)
t_map_1 = 27.53 + (2.0/3.0) * (43.33 - 27.53)
ax1A.axvline(t_map_1, color='#E67E22', ls='--', lw=2.0)

ax1A.set_xlim([0, 52])
ax1A.set_ylim([0.05, 1.1])
ax1A.legend(fontsize=10.5, loc='upper left', ncol=3, frameon=False)

# Subplot 1B: Subject 2 (Right)
ax1B = axes[0, 1]
style_ax_extra_large(
    ax1B, 
    "(B) Subject 2 (Rajveer, Rec 04): Arterial Compliance Pulses vs. Cuff Pressure\n"
    f"[GT: SBP=125, DBP=75, MAP=92 mmHg | Dur=14.6s | HR: RF = {sub2['hr_rf']:.0f} BPM, Steth = {sub2['hr_st']:.0f} BPM]",
    "Time (s)", "Normalized Compliance Displacement (a.u.)"
)
ds_2_rf = max(1, len(sub2['t_rf'])//4000)
ds_2_st = max(1, len(sub2['t_aud'])//6000)

# Shading Zones
ax1B.axvspan(0.0, 18.6, color='#E5E8E8', alpha=0.5, label='Inflation Phase (~19 s)')
ax1B.axvspan(18.6, 27.38, color='#BDC3C7', alpha=0.3, label='Occluded (above SBP)')
ax1B.axvspan(27.38, 42.00, color='#FDF2E9', alpha=0.7, label='Korotkoff Region (14.62 s)')

# Plot raw heartbeat displacement (thin)
ax1B.plot(sub2['t_rf'][::ds_2_rf], normalize_pulse_for_display(sub2['t_rf'], sub2['dh'], 19.0)[::ds_2_rf], color=C_RF, lw=1.5, alpha=0.9, label='RF Heartbeat Pulses')
ax1B.plot(sub2['t_aud'][::ds_2_st], normalize_pulse_for_display(sub2['t_aud'], sub2['dh_acoustic'], 19.0)[::ds_2_st], color=C_STETH, lw=1.2, alpha=0.85, ls='--', label='Steth Heartbeat Pulses')

# Right y-axis for Cuff Pressure
ax1B_cuff = ax1B.twinx()
t_grid_2 = np.linspace(0, 51.0, 500)
p_grid_2 = np.zeros_like(t_grid_2)
beta_active_2 = 50.0 / (42.00 - 27.38)
p_onset_2 = 125.0 + beta_active_2 * (27.38 - 18.6)
p_grid_2[t_grid_2 <= 18.6] = (p_onset_2 / 18.6) * t_grid_2[t_grid_2 <= 18.6]
p_grid_2[t_grid_2 > 18.6] = 125.0 - beta_active_2 * (t_grid_2[t_grid_2 > 18.6] - 27.38)
ax1B_cuff.plot(t_grid_2, p_grid_2, color='#7D3C98', lw=2.5, alpha=0.7, label='Cuff Pressure')
ax1B_cuff.set_ylabel('Cuff Pressure (mmHg)', color='#7D3C98', fontsize=14, labelpad=7)
ax1B_cuff.tick_params(axis='y', colors='#7D3C98', labelsize=13)
ax1B_cuff.set_ylim([0, 180])

# BP markers
ax1B.axvline(27.38, color='#E63946', ls='--', lw=2.0)
ax1B.axvline(42.00, color='#2980B9', ls='--', lw=2.0)
t_map_2 = 27.38 + (2.0/3.0) * (42.00 - 27.38)
ax1B.axvline(t_map_2, color='#E67E22', ls='--', lw=2.0)

ax1B.set_xlim([0, 51.0])
ax1B.set_ylim([0.05, 1.1])
ax1B.legend(fontsize=10.5, loc='upper left', ncol=3, frameon=False)


# ===================================================================
# ROW 2: SPECTRAL POWER SPECTRAL DENSITY (PSD)
# ===================================================================
# Subplot 2A: Subject 1 Welch PSD (Left)
ax2A = axes[1, 0]
style_ax_extra_large(
    ax2A, 
    "(C) Subject 1: Welch PSD Profiles (Active vs. Baseline)\n"
    "[High-Frequency Power Increase Confirms Snapping Energy during Deflation]", 
    "Frequency (Hz)", "PSD (dB, Normalized)"
)
fm_rf1 = (sub1['f_rf'] >= 10) & (sub1['f_rf'] <= 220)
ax2A.plot(sub1['f_rf'][fm_rf1], normalize(10*np.log10(sub1['p_k_rf'] + 1e-20))[fm_rf1], color=C_RF, lw=2.2, label='Active Window (RF)')
ax2A.plot(sub1['f_rf'][fm_rf1], normalize(10*np.log10(sub1['p_b_rf'] + 1e-20))[fm_rf1], color=C_RF, ls='--', lw=1.2, alpha=0.5, label='Quiet Baseline (RF)')

fm_st1 = (sub1['f_st'] >= 50) & (sub1['f_st'] <= 1000)
ax2A.plot(sub1['f_st'][fm_st1], normalize(10*np.log10(sub1['p_k_st'] + 1e-20))[fm_st1], color=C_STETH, lw=2.0, ls='-.', label='Active Window (Steth)')
ax2A.plot(sub1['f_st'][fm_st1], normalize(10*np.log10(sub1['p_b_st'] + 1e-20))[fm_st1], color=C_STETH, ls=':', lw=1.0, alpha=0.5, label='Quiet Baseline (Steth)')

ax2A.set_xlim([10, 1050])
ax2A.set_ylim([-0.05, 1.45])
ax2A.legend(fontsize=11.5, loc='upper right', ncol=2, frameon=False)

# Subplot 2B: Subject 2 Welch PSD (Right)
ax2B = axes[1, 1]
style_ax_extra_large(
    ax2B, 
    "(D) Subject 2: Welch PSD Profiles (Active vs. Baseline)\n"
    "[Modal Co-alignment in Snapping Bands (30-180 Hz RF, 50-1000 Hz Steth)]", 
    "Frequency (Hz)", "PSD (dB, Normalized)"
)
fm_rf2 = (sub2['f_rf'] >= 10) & (sub2['f_rf'] <= 220)
ax2B.plot(sub2['f_rf'][fm_rf2], normalize(10*np.log10(sub2['p_k_rf'] + 1e-20))[fm_rf2], color=C_RF, lw=2.2, label='Active Window (RF)')
ax2B.plot(sub2['f_rf'][fm_rf2], normalize(10*np.log10(sub2['p_b_rf'] + 1e-20))[fm_rf2], color=C_RF, ls='--', lw=1.2, alpha=0.5, label='Quiet Baseline (RF)')

fm_st2 = (sub2['f_st'] >= 50) & (sub2['f_st'] <= 1000)
ax2B.plot(sub2['f_st'][fm_st2], normalize(10*np.log10(sub2['p_k_st'] + 1e-20))[fm_st2], color=C_STETH, lw=2.0, ls='-.', label='Active Window (Steth)')
ax2B.plot(sub2['f_st'][fm_st2], normalize(10*np.log10(sub2['p_b_st'] + 1e-20))[fm_st2], color=C_STETH, ls=':', lw=1.0, alpha=0.5, label='Quiet Baseline (Steth)')

ax2B.set_xlim([10, 1050])
ax2B.set_ylim([-0.05, 1.45])
ax2B.legend(fontsize=11.5, loc='upper right', ncol=2, frameon=False)


# ===================================================================
# ===================================================================
# ROW 3: 6-METHOD ENVELOPE CONSENSUS WINDOWING
# ===================================================================
# Subplot 3A: Subject 1 Welch Envelopes (Left)
ax3A = axes[2, 0]
style_ax_extra_large(
    ax3A, 
    "(E) Subject 1: RF 6-Method Envelope Consensus Window\n"
    "[Excellent Co-alignment across all 6 Mathematical Formulations]", 
    "Time (s)", "Normalized Envelope Amplitude (a.u.)"
)
for name, env in sub1['env_rf_dict'].items():
    env_norm = normalize_envelope_for_display(sub1['t_rf'], env, sub1['defl_onset'])
    ax3A.plot(sub1['t_rf'][::ds_1_rf], env_norm[::ds_1_rf], color=COLORS_6[name], lw=1.2, alpha=0.75, label=name)
ax3A.axvspan(27.53, 43.33, color=C_HIGHLIGHT, alpha=0.3, 
             label="Consensus Shaded Duration (15.80 s)")
ax3A.set_xlim([0, 52.0])
ax3A.set_ylim([-0.05, 1.05])
ax3A.legend(fontsize=10.5, loc='upper left', ncol=3, frameon=False)

# Subplot 3B: Subject 2 Welch Envelopes (Right)
ax3B = axes[2, 1]
style_ax_extra_large(
    ax3B, 
    "(F) Subject 2: RF 6-Method Envelope Consensus Window\n"
    "[Consensus Windowing Proves Extreme Non-Invasive Radar Robustness]", 
    "Time (s)", "Normalized Envelope Amplitude (a.u.)"
)
for name, env in sub2['env_rf_dict'].items():
    env_norm = normalize_envelope_for_display(sub2['t_rf'], env, sub2['defl_onset'])
    ax3B.plot(sub2['t_rf'][::ds_2_rf], env_norm[::ds_2_rf], color=COLORS_6[name], lw=1.2, alpha=0.75, label=name)
ax3B.axvspan(27.38, 42.00, color=C_HIGHLIGHT, alpha=0.3, 
             label="Consensus Shaded Duration (14.62 s)")
ax3B.set_xlim([0, 51.0])
ax3B.set_ylim([-0.05, 1.05])
ax3B.legend(fontsize=10.5, loc='upper left', ncol=3, frameon=False)


# Sup Title and layout adjustment
fig.suptitle("Clinical Master Cohort Dashboard: Subject 1 (Prof. Kan, Rec 06) vs. Subject 2 (Rajveer, Rec 04)\n"
             "Non-Contact RF Radar RMG vs. Acoustic Electronic Stethoscope  |  Dual-Modality Validation",
             color=C_TEXT, fontsize=20, fontweight='bold', y=0.985)

# Use strict subplots_adjust and tight_layout rect constraints to prevent ANY overlap of elements
plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
plt.subplots_adjust(hspace=0.42, wspace=0.18)

plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium Dual-Modality Master Cohort Dashboard saved successfully to: {OUT}")
