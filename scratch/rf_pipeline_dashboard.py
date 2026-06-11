"""
RF Demodulation & Korotkoff Velocity Pipeline Dashboard (2x2 Panel Layout)
========================================================================
Produces a premium 2x2 panel clinical publication-grade dashboard (300 DPI, white background)
explaining the step-by-step RF radar signal processing pipeline from raw baseband IQ data
to the final physical Korotkoff velocity vk(t) for Subject 2, Rec 04.
- Panel A (Top Left): Raw Baseband IQ Constellation Circle Fitting & Clutter DC Centering.
- Panel B (Top Right): Demodulated Raw Magnitude A(t) and Raw Phase over the full recording.
- Panel C (Bottom Left): Centered Unwrapped Phase and Heartbeat Displacement Waveform (0.4–3 Hz).
- Panel D (Bottom Right): Differentiated Korotkoff Velocity vk(t) (10–200 Hz) & Locked Active Window.
"""

import h5py
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'rf_pipeline_dashboard.png')

RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Lock clinical active window bounds strictly based on SBP (125 mmHg) and DBP (75 mmHg)
# for Subject 2, Rec 04
K_ON  = 27.375
K_OFF = 42.00
DEFL  = 18.6

C_RF        = '#CA6702'  # Rust for RF
C_STETH     = '#0A9396'  # Teal for Steth
C_HIGHLIGHT = '#E9D8A6'  # Muted gold shading
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

def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin + 1e-20)

# ── LOAD & PROCESS DATA ────────────────────────────────────────────
print("Loading raw RF data for Subject 2, Rec 4...")
with h5py.File(RF_PATH, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
t_rf = np.arange(len(i_raw)) / FS_RF

# 1. Circle Fitting & Centering
i_c, q_c, xc_fit, yc_fit, R_fit = iq_condition_circle(i_raw, q_raw)

# 2. Raw Magnitude and Phase (full recording, downsampled for clean plot)
ds = 20
i_raw_ds = signal.decimate(i_raw, ds)
q_raw_ds = signal.decimate(q_raw, ds)
t_ds = t_rf[::ds]
mag_raw = np.sqrt(i_raw_ds**2 + q_raw_ds**2)
phase_raw = np.unwrap(np.angle(i_raw_ds + 1j * q_raw_ds))

# 3. Robust Centered Unwrapped Phase
phi = robust_phase(i_c, q_c)

# 4. Heartbeat Displacement (0.4-3 Hz)
sos_dh = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
dh = sosfiltfilt(sos_dh, phi) * SCALE  # mm

# 5. Korotkoff velocity (10-200 Hz)
sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE  # mm/s

# ── PLOT DUAL-MODALITY PIPELINE ────────────────────────────────────
print("Plotting premium 2x2 pipeline figure...")
fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Styling Helper
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color=C_TEXT, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=11, labelpad=4)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=11, labelpad=4)
    ax.tick_params(colors=C_TEXT, labelsize=10, length=4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(1.0)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9, ls='-')
    return ax

TBOX = dict(boxstyle='round,pad=0.35', facecolor='#ffffff', edgecolor='#dddddd', alpha=0.95, lw=0.8)

# ===================================================================
# PANEL A: IQ CONSTELLATION & CIRCLE FITTING
# ===================================================================
axA = axes[0, 0]
style_ax(axA, "(A) Baseband IQ Constellation & Clutter DC Centering", "In-phase I (a.u.)", "Quadrature Q (a.u.)")
# Plot downsampled trajectory for speed and clarity
axA.scatter(i_raw[::100], q_raw[::100], color='#BDC3C7', s=1.0, alpha=0.5, label='Raw Baseband IQ Arc')
axA.plot(xc_fit, yc_fit, 'X', color='#E63946', ms=10, mew=1.0, mec='black', label=f'Clutter Center ({xc_fit:.3f}, {yc_fit:.3f})')

# Plot fitted circle arc
theta_circle = np.linspace(0, 2*np.pi, 200)
circle_i = xc_fit + R_fit * np.cos(theta_circle)
circle_q = yc_fit + R_fit * np.sin(theta_circle)
axA.plot(circle_i, circle_q, '--', color='#1D3557', lw=1.5, label='Least-Squares Circle Fit')
axA.scatter(i_c[::100], q_c[::100], color=C_RF, s=1.0, alpha=0.5, label='Centered IQ Arc (Radius centered at 0,0)')

axA.legend(fontsize=9.5, framealpha=0.95, loc='upper right')
axA.set_aspect('equal', 'box')
axA.text(0.03, 0.97, "Strong wall/clutter reflection shifts the raw IQ arc\naway from (0,0). Circle-fitting models the clutter DC\noffset, enabling artifact-free phase unwrapping.", 
         transform=axA.transAxes, fontsize=10, va='top', bbox=TBOX)

# ===================================================================
# PANEL B: DEMODULATED RAW MAGNITUDE & PHASE
# ===================================================================
axB = axes[0, 1]
style_ax(axB, "(B) Demodulated Raw Amplitude & Unwrapped Phase", "Time (s)", "Raw Amplitude A(t) (a.u.)")
axB.plot(t_ds, mag_raw, color='#457B9D', lw=1.2, label=r'Raw Amplitude $A(t) = \sqrt{I(t)^2 + Q(t)^2}$')
axB.axvline(DEFL, color=C_TEXT, ls=':', lw=1.5, label=f"Deflation Onset ({DEFL:.1f} s)")

# Twin axis for phase
axB_ph = axB.twinx()
axB_ph.plot(t_ds, phase_raw, color='#8338EC', lw=1.2, alpha=0.85, label=r'Raw Unwrapped Phase $\phi_{raw}(t)$')
axB_ph.set_ylabel(r'Raw Phase (rad)', color='#8338EC', fontsize=11, labelpad=4)
axB_ph.tick_params(axis='y', colors='#8338EC', labelsize=10)

# Combine legends
lines1, labels1 = axB.get_legend_handles_labels()
lines2, labels2 = axB_ph.get_legend_handles_labels()
axB.legend(lines1 + lines2, labels1 + labels2, fontsize=9.5, framealpha=0.95, loc='lower right')
axB.set_xlim([0, 52])
axB.text(0.03, 0.97, r"Magnitude $A(t)$ shows the Omron cuff inflation pump" + "\n" +
                     r"ramp-up (0-18.6s) and linear pressure decay." + "\n" +
                     r"Raw phase $\phi_{raw}(t)$ drifts due to body movement.", 
         transform=axB.transAxes, fontsize=10, va='top', bbox=TBOX)

# ===================================================================
# PANEL C: CENTERED UNWRAPPED PHASE & HEARTBEAT DISPLACEMENT
# ===================================================================
axC = axes[1, 0]
style_ax(axC, "(C) Centered Unwrapped Phase & Heartbeat Displacement (0.4–3 Hz)", "Time (s)", "Arterial Heartbeat Displacement (mm)")
# Decimate for plotting speed
ds_c = 10
axC.plot(t_rf[::ds_c], (phi * SCALE)[::ds_c], color='#BDC3C7', lw=0.6, alpha=0.6, label='Centered Unwrapped Phase (mm)')
axC.plot(t_rf[::ds_c], dh[::ds_c], color=C_RF, lw=1.5, alpha=0.9, label='Arterial Heartbeat Pulse Displacement (0.4–3 Hz)')
axC.axvline(DEFL, color=C_TEXT, ls=':', lw=1.5)
axC.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, label=f"Active Korotkoff Window ({K_ON:.3f} s – {K_OFF:.2f} s)")

axC.legend(fontsize=9.5, framealpha=0.95, loc='upper right')
axC.set_xlim([0, 52])
axC.set_ylim([-1.2, 1.8])
axC.text(0.03, 0.97, "Bandpass filtering centered phase in 0.4–3 Hz\nisolates the sub-millimeter arterial wall displacement,\nrevealing highly defined cardiac heartbeat waveforms.", 
         transform=axC.transAxes, fontsize=10, va='top', bbox=TBOX)

# ===================================================================
# PANEL D: KOROTKOFF VELOCITY & ACTIVE WINDOW LOCK
# ===================================================================
axD = axes[1, 1]
style_ax(axD, "(D) Differentiated Korotkoff Micro-Velocity vk(t) (10–200 Hz)", "Time (s)", "Micro-Velocity vk(t) (mm/s)")
axD.plot(t_rf[::ds_c], vk[::ds_c], color='#1D3557', lw=0.5, alpha=0.85, label=r'Isolated Arterial Snaps $v_k(t)$ (10–200 Hz)')
axD.axvline(DEFL, color=C_TEXT, ls=':', lw=1.5, label=f"Deflation Onset ({DEFL:.1f} s)")
axD.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.3, label=f"Locked Korotkoff Region\n({K_ON:.3f} s – {K_OFF:.2f} s | {K_OFF-K_ON:.3f} s)")

axD.legend(fontsize=9.5, framealpha=0.95, loc='upper right')
axD.set_xlim([0, 52])
axD.set_ylim([-0.12, 0.18])
axD.text(0.03, 0.97, "Differentiating the phase and bandpass filtering (10-200 Hz)\nisolates high-frequency micro-velocity transients (arterial snaps)\nwhich occur strictly within the active Korotkoff region.", 
         transform=axD.transAxes, fontsize=10, va='top', bbox=TBOX)

# Sup Title and layout adjustment
fig.suptitle("USRP Radar Radiomyography (RMG) Signal Extraction & Demodulation Pipeline\n"
             "Subject 2 (Rajveer), Rec 04  |  Step-by-Step Raw Baseband to Differentiated Korotkoff Velocity  |  300 DPI",
             color=C_TEXT, fontsize=16, fontweight='bold', y=0.975)

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.94])
plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium RF Radar Pipeline Dashboard saved successfully to: {OUT}")
