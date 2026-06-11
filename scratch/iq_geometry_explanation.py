"""
IQ Geometry & Signal Extraction Principles (3x2 Panel Layout)
===============================================================
Produces a premium 3x2 panel publication-grade clinical figure (300 DPI, white background)
explaining the mathematical relationship between the raw IQ arc, baseband magnitude,
centered phase, and extracted physiological signals, with large legible text.
"""

import h5py
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'iq_geometry_explanation.png')

RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_2.h5')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Enforce post-inflation Korotkoff windows (starts after 20s)
TARGET_DUR_S = 17.5
STETH_OFFSET = 3.5

# Premium publication colors
C_RAW       = '#94D2BD'  # Muted green for raw data
C_MAG       = '#CA6702'  # Rust Orange for Magnitude
C_PHASE     = '#005F73'  # Deep Blue-Green for Phase
C_GRID      = '#E5E5E5'  # Light grid
C_TEXT      = '#222222'  # Dark text
C_HIGHLIGHT = '#AE2012'  # Crimson highlight for segments
C_FIT       = '#9B5DE5'  # Muted Purple for circle fits

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

# ── LOAD & PROCESS DATA ────────────────────────────────────────────
print("Loading and processing baseband IQ geometry...")
with h5py.File(RF_PATH, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
N_rf = len(i_raw)
t_rf = np.arange(N_rf) / FS_RF

# Fit raw circle and center
xc, yc, R_fit = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
phi = robust_phase(i_c, q_c)

# Calculate baseband magnitude
mag = np.sqrt(i_raw**2 + q_raw**2)

# Select a 5-second active window segment for visual highlight (e.g. t = 25s to 30s)
t_start, t_end = 25.0, 30.0
mask_seg = (t_rf >= t_start) & (t_rf <= t_end)
i_seg, q_seg = i_raw[mask_seg], q_raw[mask_seg]
i_c_seg, q_c_seg = i_c[mask_seg], q_c[mask_seg]

# Filters
sos_k = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
sos_hb = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')

vk_m = np.append(np.diff(sosfiltfilt(sos_k, mag)) * FS_RF, 0)
vk_p = np.append(np.diff(sosfiltfilt(sos_k, phi)) * FS_RF, 0) * SCALE

hb_m = sosfiltfilt(sos_hb, mag)
hb_p = sosfiltfilt(sos_hb, phi) * SCALE

# ── PLOT 3x2 EXPLANATORY FIGURE ────────────────────────────────────
print("Plotting premium 3x2 explanatory dashboard...")
fig, axes = plt.subplots(3, 2, figsize=(18, 22), dpi=300)
fig.patch.set_facecolor('#ffffff')

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
# ROW 1: MAGNITUDE & RAW IQ ARC RELATION
# ===================================================================
# Subplot 1A: Raw IQ Constellation (Top Left)
ax1A = axes[0, 0]
style_ax_large(ax1A, "Raw IQ Constellation & Circular Arc trajectory", "In-Phase I(t)", "Quadrature Q(t)")
ds = max(1, len(i_raw)//10000)
ax1A.scatter(i_raw[::ds], q_raw[::ds], color=C_RAW, s=2, alpha=0.3, label='Full Session raw IQ')
ax1A.plot(i_seg[::2], q_seg[::2], color=C_HIGHLIGHT, lw=2.0, label=f"Selected segment ({t_start:.0f}–{t_end:.0f}s)")

# Draw circle fit
theta_c = np.linspace(0, 2*np.pi, 200)
ax1A.plot(xc + R_fit*np.cos(theta_c), yc + R_fit*np.sin(theta_c), color=C_FIT, ls='--', lw=1.5, label='Circle Fit')
ax1A.scatter([xc], [yc], color=C_FIT, marker='+', s=100, zorder=5, label='Center $(x_c, y_c)$')

# Draw Magnitude vectors from origin (0,0) to arc
ax1A.arrow(0, 0, i_seg[0], q_seg[0], color=C_MAG, head_width=0.01, length_includes_head=True, zorder=6, label=r'Magnitude vector $M(t)$')
ax1A.arrow(0, 0, i_seg[-1], q_seg[-1], color=C_MAG, head_width=0.01, length_includes_head=True, zorder=6)

ax1A.axis('equal')
ax1A.legend(fontsize=10, framealpha=0.95, loc='lower right')
ax1A.text(0.03, 0.93, "The raw baseband IQ signal traces a circular arc\nwhose distance from origin represents Magnitude M(t).", 
          transform=ax1A.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# Subplot 1B: Raw Magnitude vs. Time (Top Right)
ax1B = axes[0, 1]
style_ax_large(ax1B, "Baseband Magnitude M(t) over Time", "Time (s)", "Raw Magnitude (a.u.)")
ds_t = max(1, len(t_rf)//4000)
ax1B.plot(t_rf[::ds_t], mag[::ds_t], color=C_MAG, lw=1.2, label='Magnitude $M(t)$')
ax1B.plot(t_rf[mask_seg][::2], mag[mask_seg][::2], color=C_HIGHLIGHT, lw=2.0, label='Highlighted Segment')
ax1B.set_xlim([10, 52])
ax1B.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax1B.text(0.03, 0.93, "Magnitude profile follows the raw respiratory\ntrajecotry shape, matching the raw IQ arc behavior.", 
          transform=ax1B.transAxes, fontsize=10.5, va='top', bbox=TBOX)


# ===================================================================
# ROW 2: CENTERED PHASE & CENTERED IQ ARC RELATION
# ===================================================================
# Subplot 2A: Centered IQ Constellation (Center Left)
ax2A = axes[1, 0]
style_ax_large(ax2A, "Centered IQ Constellation & Angular Phase", "Centered I_c(t)", "Centered Q_c(t)")
ax2A.scatter(i_c[::ds], q_c[::ds], color=C_RAW, s=2, alpha=0.3, label='Full Session Centered IQ')
ax2A.plot(i_c_seg[::2], q_c_seg[::2], color=C_HIGHLIGHT, lw=2.0, label=f"Selected segment ({t_start:.0f}–{t_end:.0f}s)")

# Draw sector representing Phase
ax2A.plot([0, i_c_seg[0]], [0, q_c_seg[0]], color=C_PHASE, ls=':', lw=1.5, label=r'Phase Angle vectors $\phi(t)$')
ax2A.plot([0, i_c_seg[-1]], [0, q_c_seg[-1]], color=C_PHASE, ls=':', lw=1.5)
ax2A.scatter([0], [0], color=C_PHASE, marker='+', s=100, zorder=5, label='New Origin (0,0)')

ax2A.axis('equal')
ax2A.legend(fontsize=10, framealpha=0.95, loc='lower right')
ax2A.text(0.03, 0.93, "Subtracting circle center centers the arc at (0,0).\nRadar displacement now corresponds strictly\nto angular sector phase variations.", 
          transform=ax2A.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# Subplot 2B: Centered Unwrapped Phase vs. Time (Center Right)
ax2B = axes[1, 1]
style_ax_large(ax2B, r"Unwrapped Centered Phase $\phi(t)$ over Time", "Time (s)", "Centered Phase (rad)")
ax2B.plot(t_rf[::ds_t], phi[::ds_t], color=C_PHASE, lw=1.2, label=r'Unwrapped Phase $\phi(t)$')
ax2B.plot(t_rf[mask_seg][::2], phi[mask_seg][::2], color=C_HIGHLIGHT, lw=2.0, label='Highlighted Segment')
ax2B.set_xlim([10, 52])
ax2B.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax2B.text(0.03, 0.93, "Centered phase unwrapping removes multi-period\nambiguity, capturing high-fidelity chest movements.", 
          transform=ax2B.transAxes, fontsize=10.5, va='top', bbox=TBOX)


# ===================================================================
# ROW 3: FILTERED PHYSIOLOGICAL COMPONENTS COMPARISON
# ===================================================================
# Subplot 3A: Magnitude-Based Physiological Components (Bottom Left)
ax3A = axes[2, 0]
style_ax_large(ax3A, "Magnitude-Based Physiological Extraction (NCSam)", "Time (s)", "Normalized Amplitude (a.u.)")
t_seg = t_rf[mask_seg][::2]
hb_m_seg = normalize(hb_m[mask_seg])[::2]
vk_m_seg = normalize(vk_m[mask_seg])[::2]
ax3A.plot(t_seg, hb_m_seg, color=C_MAG, lw=1.5, label='Heartbeat Wave (0.4-3.0 Hz)')
ax3A.plot(t_seg, vk_m_seg - 0.2, color='#83C5BE', lw=0.6, alpha=0.9, label='Korotkoff Wave (10-200 Hz)')
ax3A.set_xlim([t_start, t_end])
ax3A.set_ylim([-0.3, 1.3])
ax3A.legend(fontsize=10, framealpha=0.95, loc='upper right')

# Subplot 3B: Phase-Based Physiological Components (Bottom Right)
ax3B = axes[2, 1]
style_ax_large(ax3B, "Phase-Based Physiological Extraction (NCSph)", "Time (s)", "Normalized Amplitude (a.u.)")
hb_p_seg = normalize(hb_p[mask_seg])[::2]
vk_p_seg = normalize(vk_p[mask_seg])[::2]
ax3B.plot(t_seg, hb_p_seg, color=C_PHASE, lw=1.5, label='Heartbeat Wave (0.4-3.0 Hz)')
ax3B.plot(t_seg, vk_p_seg - 0.2, color='#E9D8A6', lw=0.6, alpha=0.9, label='Korotkoff Wave (10-200 Hz)')
ax3B.set_xlim([t_start, t_end])
ax3B.set_ylim([-0.3, 1.3])
ax3B.legend(fontsize=10, framealpha=0.95, loc='upper right')


# Sup Title and layout adjustment
fig.suptitle("Mathematical Extraction Principles of RMG Radar: Raw IQ Geometry, Magnitude & Centered Phase\n"
             "Comparing Magnitude-Based (NCSam) and Phase-Based (NCSph) Signal Representations  |  LARGE TEXT FOR PUBLICATION",
             color=C_TEXT, fontsize=17, fontweight='bold', y=0.985)

fig.text(0.02, 0.945, "PANEL 1: Baseband Magnitude (NCSam) & Circular Arc Geometrical Link", color=C_TEXT, fontsize=13.5, fontweight='bold')
fig.text(0.02, 0.645, "PANEL 2: Centered Angular Phase (NCSph) Geometrical Translation", color=C_TEXT, fontsize=13.5, fontweight='bold')
fig.text(0.02, 0.345, "PANEL 3: Extracted Heartbeat and Korotkoff Physiological Waveforms Comparison", color=C_TEXT, fontsize=13.5, fontweight='bold')

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium 3x2 IQ Geometry Figure saved successfully to: {OUT}")
