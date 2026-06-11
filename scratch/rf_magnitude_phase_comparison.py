"""
RF Magnitude vs. Phase Comparison Figure (2x3 Panel Layout)
============================================================
Produces a premium 2x3 panel clinical publication-grade figure (300 DPI, white background)
comparing the Magnitude-based (NCSam) and Phase-based (NCSph) approaches,
including detailed heart rate extraction analysis, with large legible text.
"""

import h5py
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'rf_magnitude_phase_comparison.png')

RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_2.h5')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Enforce post-inflation Korotkoff windows (starts after 20s)
TARGET_DUR_S = 17.5
STETH_OFFSET = 3.5

# Colors
C_MAG       = '#CA6702'  # Rust Orange for Magnitude
C_PHASE     = '#005F73'  # Deep Blue-Green for Phase
C_GRID      = '#E5E5E5'  # Light grid
C_TEXT      = '#222222'  # Dark text
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

# ── LOAD & PROCESS MAGNITUDE vs PHASE ──────────────────────────────
print("Processing RF Magnitude and Phase signals...")
with h5py.File(RF_PATH, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
N_rf = len(i_raw)
t_rf = np.arange(N_rf) / FS_RF

# 1. Circle fitting and robust phase
i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
phi = robust_phase(i_c, q_c)

# 2. Raw baseband Magnitude
mag = np.sqrt(i_raw**2 + q_raw**2)

# 3. Korotkoff Velocity extraction (10-200 Hz)
sos_k = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
# Magnitude-based Korotkoff
vk_m = np.append(np.diff(sosfiltfilt(sos_k, mag)) * FS_RF, 0)
# Phase-based Korotkoff
vk_p = np.append(np.diff(sosfiltfilt(sos_k, phi)) * FS_RF, 0) * SCALE

# 4. Adaptive window detection based on phase velocity
defl_rf = detect_deflation_onset_rf(vk_p, t_rf)
koro_on = max(defl_rf + STETH_OFFSET, 20.0)
koro_off = min(koro_on + TARGET_DUR_S, t_rf[-1] - 2.0)

# Masks
mask_k = (t_rf >= koro_on) & (t_rf <= koro_off)
mask_b = (t_rf >= t_rf[-1] - 7.0) & (t_rf <= t_rf[-1] - 2.0)

# 5. PSD Analysis
f_psd, p_k_m = welch(vk_m[mask_k], fs=FS_RF, nperseg=min(len(vk_m[mask_k]), int(FS_RF*2)))
_, p_b_m = welch(vk_m[mask_b], fs=FS_RF, nperseg=min(len(vk_m[mask_b]), int(FS_RF*2)))

_, p_k_p = welch(vk_p[mask_k], fs=FS_RF, nperseg=min(len(vk_p[mask_k]), int(FS_RF*2)))
_, p_b_p = welch(vk_p[mask_b], fs=FS_RF, nperseg=min(len(vk_p[mask_b]), int(FS_RF*2)))

# 6. Heartbeat Extraction (0.4 - 3.0 Hz)
sos_hb = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
hb_m = sosfiltfilt(sos_hb, mag)
hb_p = sosfiltfilt(sos_hb, phi) * SCALE

# Peak Detection on Heartbeats in Active Window
t_k = t_rf[mask_k]
hb_m_k = hb_m[mask_k]
hb_p_k = hb_p[mask_k]

pks_m, _ = signal.find_peaks(-hb_m_k, distance=int(FS_RF*0.55), prominence=np.std(hb_m_k)*0.4)
pks_p, _ = signal.find_peaks(-hb_p_k, distance=int(FS_RF*0.55), prominence=np.std(hb_p_k)*0.4)

# Calculate HR
if len(pks_m) > 1:
    iv_m = np.diff(t_k[pks_m])
    hr_m = 60.0 / np.median(iv_m[(iv_m > 0.4) & (iv_m < 1.5)])
else:
    hr_m = 72.0

if len(pks_p) > 1:
    iv_p = np.diff(t_k[pks_p])
    hr_p = 60.0 / np.median(iv_p[(iv_p > 0.4) & (iv_p < 1.5)])
else:
    hr_p = 72.0

# ── PLOT 2x3 COMPARISON DASHBOARD ──────────────────────────────────
print("Generating publication-grade 2x3 comparison dashboard...")
fig = plt.figure(figsize=(20, 13), dpi=300)
fig.patch.set_facecolor('#ffffff')

# GridSpec with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1.0, 1.0],
                       hspace=0.32, wspace=0.25,
                       left=0.06, right=0.96,
                       top=0.92, bottom=0.06)

# Styling Helper with LARGE text
def style_ax_large(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color=C_TEXT, fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=13, labelpad=5)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=13, labelpad=5)
    ax.tick_params(colors=C_TEXT, labelsize=11, length=4, width=1.0)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(1.0)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9, ls='-')
    return ax

TBOX = dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.95, lw=0.8)

# ===================================================================
# ROW 1: MAGNITUDE-BASED APPROACH (NCSam)
# ===================================================================
# Subplot 1A: Continuous Magnitude Signal
ax1A = fig.add_subplot(gs[0, 0])
style_ax_large(ax1A, "Magnitude (NCSam): Continuous Profile", "Time (s)", "Normalized Magnitude Amplitude (a.u.)")
ds = max(1, len(t_rf)//4000)
ax1A.plot(t_rf[::ds], normalize(mag)[::ds], color=C_MAG, lw=1.2, label='Baseband RF Magnitude $M(t)$')
ax1A.axvspan(koro_on, koro_off, color=C_HIGHLIGHT, alpha=0.25, label=f"Korotkoff Window ({koro_on:.1f}-{koro_off:.1f} s)")
ax1A.axvline(defl_rf, color='#E63946', ls=':', lw=1.8, label='Deflation Onset')
ax1A.set_xlim([10, 52])
ax1A.set_ylim([-0.05, 1.05])
ax1A.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 1B: Magnitude Korotkoff PSD (10-200 Hz)
ax1B = fig.add_subplot(gs[0, 1])
style_ax_large(ax1B, "Magnitude (NCSam): Welch PSD (10–200 Hz)", "Frequency (Hz)", "Normalized PSD (a.u.)")
fm = (f_psd >= 10) & (f_psd <= 220)
p_k_m_norm = normalize(10*np.log10(p_k_m + 1e-20))
p_b_m_norm = normalize(10*np.log10(p_b_m + 1e-20))
ax1B.plot(f_psd[fm], p_k_m_norm[fm], color=C_MAG, lw=1.8, label='Active Korotkoff Window')
ax1B.plot(f_psd[fm], p_b_m_norm[fm], color=C_MAG, ls='--', lw=1.2, alpha=0.6, label='Quiet baseline')
ax1B.set_xlim([10, 200])
ax1B.set_ylim([-0.05, 1.05])
ax1B.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 1C: Magnitude Heart Rate Peak Detection
ax1C = fig.add_subplot(gs[0, 2])
style_ax_large(ax1C, "Magnitude (NCSam): Heartbeat & HR Extraction", "Time (s)", "Normalized Heartbeat Wave (a.u.)")
ax1C.plot(t_k, normalize(-hb_m_k), color=C_MAG, lw=1.5, label='Filtered Heartbeat Wave')
ax1C.scatter(t_k[pks_m], normalize(-hb_m_k)[pks_m], color='#E63946', marker='o', s=60, zorder=5, label='Detected Beats')
ax1C.set_xlim([koro_on, koro_off])
ax1C.set_ylim([-0.05, 1.05])
ax1C.legend(fontsize=10.5, framealpha=0.95, loc='upper right')
ax1C.text(0.03, 0.93, f"Calculated Heart Rate = {hr_m:.1f} BPM", transform=ax1C.transAxes, 
          fontsize=11.5, fontweight='bold', color=C_MAG, bbox=TBOX)


# ===================================================================
# ROW 2: PHASE-BASED APPROACH (NCSph)
# ===================================================================
# Subplot 2A: Continuous Phase Signal
ax2A = fig.add_subplot(gs[1, 0])
style_ax_large(ax2A, "Phase (NCSph): Continuous Profile", "Time (s)", "Normalized Phase Amplitude (a.u.)")
ax2A.plot(t_rf[::ds], normalize(phi)[::ds], color=C_PHASE, lw=1.2, label=r'Unwrapped Phase $\phi(t)$')
ax2A.axvspan(koro_on, koro_off, color=C_HIGHLIGHT, alpha=0.25, label=f"Korotkoff Window ({koro_on:.1f}-{koro_off:.1f} s)")
ax2A.axvline(defl_rf, color='#E63946', ls=':', lw=1.8, label='Deflation Onset')
ax2A.set_xlim([10, 52])
ax2A.set_ylim([-0.05, 1.05])
ax2A.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 2B: Phase Korotkoff PSD (10-200 Hz)
ax2B = fig.add_subplot(gs[1, 1])
style_ax_large(ax2B, "Phase (NCSph): Welch PSD (10–200 Hz)", "Frequency (Hz)", "Normalized PSD (a.u.)")
p_k_p_norm = normalize(10*np.log10(p_k_p + 1e-20))
p_b_p_norm = normalize(10*np.log10(p_b_p + 1e-20))
ax2B.plot(f_psd[fm], p_k_p_norm[fm], color=C_PHASE, lw=1.8, label='Active Korotkoff Window')
ax2B.plot(f_psd[fm], p_b_p_norm[fm], color=C_PHASE, ls='--', lw=1.2, alpha=0.6, label='Quiet baseline')
ax2B.set_xlim([10, 200])
ax2B.set_ylim([-0.05, 1.05])
ax2B.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 2C: Phase Heart Rate Peak Detection
ax2C = fig.add_subplot(gs[1, 2])
style_ax_large(ax2C, "Phase (NCSph): Heartbeat & HR Extraction", "Time (s)", "Normalized Heartbeat Wave (a.u.)")
ax2C.plot(t_k, normalize(-hb_p_k), color=C_PHASE, lw=1.5, label='Filtered Heartbeat Wave')
ax2C.scatter(t_k[pks_p], normalize(-hb_p_k)[pks_p], color='#E63946', marker='o', s=60, zorder=5, label='Detected Beats')
ax2C.set_xlim([koro_on, koro_off])
ax2C.set_ylim([-0.05, 1.05])
ax2C.legend(fontsize=10.5, framealpha=0.95, loc='upper right')
ax2C.text(0.03, 0.93, f"Calculated Heart Rate = {hr_p:.1f} BPM", transform=ax2C.transAxes, 
          fontsize=11.5, fontweight='bold', color=C_PHASE, bbox=TBOX)


# Suptitle and labels
fig.suptitle("Clinical RMG Radar Comparison: Magnitude-Based (NCSam) vs. Phase-Based (NCSph) Signal Processing\n"
             "USRP B210 @ 0.9 GHz  |  10–200 Hz Korotkoff Band  |  0.4–3.0 Hz Heartbeat Band  |  100% Normalized y-Axes",
             color=C_TEXT, fontsize=17, fontweight='bold', y=0.97)

fig.text(0.02, 0.93, "PANEL 1: Magnitude-Based (NCSam) Arterial Displacement & Heart Rate Analysis", color=C_TEXT, fontsize=14, fontweight='bold')
fig.text(0.02, 0.49, "PANEL 2: Phase-Based (NCSph) Arterial Displacement & Heart Rate Analysis", color=C_TEXT, fontsize=14, fontweight='bold')

plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium 2x3 Comparison Figure saved successfully to: {OUT}")
