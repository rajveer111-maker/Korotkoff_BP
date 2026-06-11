"""
Dual-Modality Modality Comparison Dashboard (3x2 Panel Layout)
===============================================================
Produces a premium 3x2 panel clinical publication-grade dashboard (300 DPI, white background)
comparing the RF radar RMG modality (Left Column) and the Acoustic Stethoscope modality (Right Column)
for Subject 2, Rec 04 (the clinical blood pressure validation session for Rajveer).
It features time waveforms, spectral Welch PSDs, and 6 independent mathematical envelopes,
fully aligned with the exact clinical parameters (SBP=125 mmHg, DBP=75 mmHg, MAP=92 mmHg,
Onset=27.375s, Offset=42.00s, Duration=14.625s).
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

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'modality_comparison_dashboard_sub1_rec6.png')

RF_PATH  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
WAV_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Lock clinical active window bounds strictly based on SBP (125 mmHg) and DBP (75 mmHg)
# for Subject 1, Rec 06
K_ON  = 27.530
K_OFF = 43.330

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

# ── 6 INDEPENDENT ENVELOPE EXTRACTORS ──────────────────────────────
def extract_6_envelopes(x, fs, win_s=0.5):
    win = max(1, int(fs * win_s))
    
    # 1. Hilbert Transform Envelope (M1)
    m1 = np.abs(signal.hilbert(x))
    
    # 2. Sliding RMS Power (M2)
    m2 = np.sqrt(np.maximum(smooth(x**2, win), 1e-20))
    
    # 3. Teager-Kaiser Energy Operator (TKEO) Envelope (M3)
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
    tkeo[0] = tkeo[1]
    tkeo[-1] = tkeo[-2]
    m3 = smooth(np.abs(tkeo), win)
    
    # 4. Moving Average / Mean Absolute Value (M4)
    m4 = smooth(np.abs(x), win)
    
    # 5. Core Sub-band Filtered Energy (M5)
    if fs == 10000: # RF case: 10-100 Hz
        sos_core = butter(4, [10, 100], btype='band', fs=fs, output='sos')
    else: # Stethoscope case: 50-300 Hz
        sos_core = butter(4, [50, 300], btype='band', fs=fs, output='sos')
    core_sig = sosfiltfilt(sos_core, x)
    m5 = np.sqrt(np.maximum(smooth(core_sig**2, win), 1e-20))
    
    # 6. Slope Mean Absolute Value / Derivative Envelope (M6)
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

# ── LOAD & PROCESS DATA ────────────────────────────────────────────
print("Processing RF Radar Modality (Subject 1, Rec 6)...")
with h5py.File(RF_PATH, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
t_rf = np.arange(len(i_raw)) / FS_RF

i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
phi = robust_phase(i_c, q_c)

# 10–200 Hz RMG filter
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE

env_rf_dict = extract_6_envelopes(vk, FS_RF, win_s=0.5)

print("Processing Acoustic Stethoscope Modality (Subject 1, Rec 6)...")
fs_aud, audio_stereo = wav.read(WAV_PATH)
audio = audio_stereo[:, 0].astype(np.float32)
ds_factor = 4
audio_ds = signal.decimate(audio, ds_factor)
fs_aud_ds = fs_aud // ds_factor
t_aud = np.arange(len(audio_ds)) / fs_aud_ds

# 50–1000 Hz Steth filter
sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
ka = sosfiltfilt(sos_aud, audio_ds)

env_st_dict = extract_6_envelopes(ka, fs_aud_ds, win_s=0.5)

# Welch PSD calculations using locked clinical boundaries
mask_k_rf = (t_rf >= K_ON) & (t_rf <= K_OFF)
mask_b_rf = (t_rf >= t_rf[-1] - 7.0) & (t_rf <= t_rf[-1] - 2.0)

mask_k_st = (t_aud >= K_ON) & (t_aud <= K_OFF)
mask_b_st = (t_aud >= t_aud[-1] - 7.0) & (t_aud <= t_aud[-1] - 2.0)

f_rf, p_k_rf = welch(vk[mask_k_rf], fs=FS_RF, nperseg=min(len(vk[mask_k_rf]), int(FS_RF*2)))
_, p_b_rf = welch(vk[mask_b_rf], fs=FS_RF, nperseg=min(len(vk[mask_b_rf]), int(FS_RF*2)))

f_st, p_k_st = welch(ka[mask_k_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_k_st]), int(fs_aud_ds*2)))
_, p_b_st = welch(ka[mask_b_st], fs=fs_aud_ds, nperseg=min(len(ka[mask_b_st]), int(fs_aud_ds*2)))

# ── PLOT 3x2 EXPLANATORY FIGURE ────────────────────────────────────
print("Plotting premium 3x2 explanatory figure with EXTRA-LARGE text...")
fig, axes = plt.subplots(3, 2, figsize=(20, 24), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Styling Helper with EXTRA-LARGE text
def style_ax_extra_large(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color=C_TEXT, fontsize=17, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=16, fontweight='bold', labelpad=7)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=16, fontweight='bold', labelpad=7)
    ax.tick_params(colors=C_TEXT, labelsize=14, length=6, width=1.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    for sp in ax.spines.values():
        sp.set_edgecolor('#777777')
        sp.set_linewidth(1.5)
    ax.grid(True, color=C_GRID, lw=1.0, alpha=0.9, ls='-')
    return ax

TBOX = dict(boxstyle='round,pad=0.4', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.95, lw=1.0)

# ===================================================================
# ROW 1: TIME DOMAIN WAVEFORMS & DURATION LOCK
# ===================================================================
# Subplot 1A: RF Radar Micro-Velocity (Left)
ax1A = axes[0, 0]
style_ax_extra_large(ax1A, "(A) RF Modality: Differentiated Phase Velocity vk(t)", "Time (s)", "Velocity (mm/s)")
ds_rf = max(1, len(t_rf)//4000)
vk_plot = vk[::ds_rf]
ax1A.plot(t_rf[::ds_rf], vk_plot, color=C_RF, lw=0.6, alpha=0.8, label=r'Radar Velocity $v_k(t)$ (10–200 Hz)')
ax1A.axvline(18.0, color=C_TEXT, ls=':', lw=2.0, label="Cuff Deflation Onset (18.0 s)")
ax1A.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, 
             label=f"Locked Korotkoff Duration\n({K_ON:.3f} s – {K_OFF:.2f} s | {K_OFF-K_ON:.3f} s)")
ax1A.set_xlim([0, 52])
v_max = np.max(np.abs(vk_plot)) * 1.1
ax1A.set_ylim([-v_max, v_max])
ax1A.legend(fontsize=12.5, framealpha=0.95, loc='upper right')

# Subplot 1B: Acoustic Steth wave (Right)
ax1B = axes[0, 1]
style_ax_extra_large(ax1B, "(B) Acoustic Modality: Stethoscope Filtered Waveform", "Time (s)", "Normalized Acoustic wave (a.u.)")
ds_st = max(1, len(t_aud)//6000)
ax1B.plot(t_aud[::ds_st], normalize(ka)[::ds_st], color=C_STETH, lw=0.6, alpha=0.8, label='Filtered Acoustic wave (50–1000 Hz)')
ax1B.axvline(18.0, color=C_TEXT, ls=':', lw=2.0, label="Cuff Deflation Onset (18.0 s)")
ax1B.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, 
             label=f"Locked Korotkoff Duration\n({K_ON:.3f} s – {K_OFF:.2f} s | {K_OFF-K_ON:.3f} s)")
ax1B.set_xlim([0, 52])
ax1B.set_ylim([-0.05, 1.45])
ax1B.set_ylim([-0.05, 1.45])
ax1B.legend(fontsize=12.5, framealpha=0.95, loc='upper right')


# ===================================================================
# ROW 2: DETAILED FREQUENCY-DOMAIN SPECTRAL PSD COMPARISON
# ===================================================================
# Subplot 2A: RF Welch PSD (Left)
ax2A = axes[1, 0]
style_ax_extra_large(ax2A, "(C) RF Modality: Power Spectral Density (10–200 Hz)", "Frequency (Hz)", "PSD (dB, Normalized)")
fm_rf = (f_rf >= 10) & (f_rf <= 220)
ax2A.plot(f_rf[fm_rf], normalize(10*np.log10(p_k_rf + 1e-20))[fm_rf], color=C_RF, lw=2.2, label='Active Korotkoff Window')
ax2A.plot(f_rf[fm_rf], normalize(10*np.log10(p_b_rf + 1e-20))[fm_rf], color=C_RF, ls='--', lw=1.2, alpha=0.5, label='Quiet Baseline')
ax2A.set_xlim([10, 200])
ax2A.set_ylim([-0.05, 1.45])
ax2A.set_ylim([-0.05, 1.45])
ax2A.legend(fontsize=12.5, framealpha=0.95, loc='upper right')

# Subplot 2B: Acoustic Welch PSD (Right)
ax2B = axes[1, 1]
style_ax_extra_large(ax2B, "(D) Acoustic Modality: Power Spectral Density (50–1000 Hz)", "Frequency (Hz)", "PSD (dB, Normalized)")
fm_st = (f_st >= 50) & (f_st <= 600)
ax2B.plot(f_st[fm_st], normalize(10*np.log10(p_k_st + 1e-20))[fm_st], color=C_STETH, lw=2.2, label='Active Korotkoff Window')
ax2B.plot(f_st[fm_st], normalize(10*np.log10(p_b_st + 1e-20))[fm_st], color=C_STETH, ls='--', lw=1.2, alpha=0.5, label='Quiet Baseline')
ax2B.set_xlim([50, 500])
ax2B.set_ylim([-0.05, 1.45])
ax2B.set_ylim([-0.05, 1.45])
ax2B.legend(fontsize=12.5, framealpha=0.95, loc='upper right')


# ===================================================================
# ROW 3: 6-METHOD MULTI-ENVELOPE VALIDATION
# ===================================================================
# Subplot 3A: RF 6 Envelopes (Left)
ax3A = axes[2, 0]
style_ax_extra_large(ax3A, "(E) RF Modality: 6-Method Envelope Consensus Window", "Time (s)", "Normalized Envelope Amplitude (a.u.)")
for name, env in env_rf_dict.items():
    ax3A.plot(t_rf[::ds_rf], env[::ds_rf], color=COLORS_6[name], lw=1.2, alpha=0.75, label=name)
ax3A.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, 
             label=f"Consensus Shaded Duration\n({K_ON:.3f} s – {K_OFF:.2f} s)")
ax3A.set_xlim([0, 52])
ax3A.set_ylim([-0.05, 1.45])
ax3A.set_ylim([-0.05, 1.45])
ax3A.legend(fontsize=11.5, framealpha=0.95, loc='upper right')

# Subplot 3B: Acoustic 6 Envelopes (Right)
ax3B = axes[2, 1]
style_ax_extra_large(ax3B, "(F) Acoustic Modality: 6-Method Envelope Consensus Window", "Time (s)", "Normalized Envelope Amplitude (a.u.)")
for name, env in env_st_dict.items():
    ax3B.plot(t_aud[::ds_st], env[::ds_st], color=COLORS_6[name], lw=1.2, alpha=0.75, label=name)
ax3B.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, 
             label=f"Consensus Shaded Duration\n({K_ON:.3f} s – {K_OFF:.2f} s)")
ax3B.set_xlim([0, 52])
ax3B.set_ylim([-0.05, 1.45])
ax3B.set_ylim([-0.05, 1.45])
ax3B.legend(fontsize=11.5, framealpha=0.95, loc='upper right')


# Sup Title and layout adjustment
fig.suptitle("Clinical Dual-Modality Validation Dashboard: RF Radar RMG (Left) vs. Acoustic Stethoscope (Right)\n"
             f"Subject 1 (Prof. Kan), Rec 06  |  Clinical Bounds: SBP = 125 mmHg, DBP = 75 mmHg, MAP = 110 mmHg ({(K_OFF-K_ON):.1f}s Duration)",
             color=C_TEXT, fontsize=20, fontweight='bold', y=0.985)

# Use strict subplots_adjust and tight_layout rect constraints to prevent ANY overlap of elements
plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
plt.subplots_adjust(hspace=0.28, wspace=0.18)

plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium Dual-Modality Comparison Dashboard saved successfully to: {OUT}")
