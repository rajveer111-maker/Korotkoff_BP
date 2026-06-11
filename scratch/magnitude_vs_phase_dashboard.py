"""
RF Demodulation Comparison Dashboard: Magnitude vs. Phase Domains
==================================================================
Produces a premium 4x2 panel clinical publication-grade dashboard (300 DPI, white background)
comparing RF Magnitude-based analysis (Left Column) and RF Phase-based analysis (Right Column)
for Subject 2, Rec 04 (Rajveer's clinical recording).
- Adheres strictly to high-tier medical journal standards (Nature, IEEE, Springer) with desaturated,
  soft, and light clinical colors.
- Row 1: High-Frequency Snapping Clicks (10–200 Hz) & Envelopes.
- Row 2: Zoomed Cardiac compliance vs. snaps (K_ON to K_ON + 8s, locally normalized to [-1, 1] for 100% visibility).
- Row 3: Absolute Displacement vs. Time Analysis (0 to 47s) for Magnitude (Left) and Phase (Right).
- Row 4: Spectrograms / STFT plots (10–200 Hz) shifted strictly into the bottom row.
"""

import h5py
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, decimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'magnitude_vs_phase_dashboard.png')

RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')

FS_RF     = 10000
DEC       = 10
FS_HR     = FS_RF // DEC  # 1 kHz downsampled rate for stability
FC        = 0.9e9
LAMBDA_MM = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE     = LAMBDA_MM / (4 * np.pi)      # 26.51 mm/rad

# Lock active window bounds
K_ON  = 27.375
K_OFF = 42.00
DEFL  = 18.6
T_MAX = 47.0  # Max recording length

# ── JOURNAL STANDARD LIGHT CLINICAL COLORS ─────────────────────────
# Uses desaturated, light, and elegant color schemes matching Nature and IEEE standards
C_MAGNITUDE      = '#457B9D'  # Soft steel blue
C_MAGNITUDE_ENV  = '#1D3557'  # Deep navy steel blue
C_PHASE          = '#E07A5F'  # Soft terracotta rose
C_PHASE_ENV      = '#9A3B3B'  # Soft dark crimson
C_HEARTBEAT      = '#2B2D42'  # Soft dark charcoal black
C_RAW_DRIFT      = '#E0E0E0'  # Soft light desaturated grey
C_HIGHLIGHT      = '#F4F1DE'  # Soft light eggshell cream
C_TEXT           = '#333333'  # Soft dark text
C_GRID           = '#F1F1F1'  # Super thin grid line

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

# ── LOAD & PROCESS DATA ────────────────────────────────────────────
print("Loading raw RF data for Subject 2, Rec 4...")
with h5py.File(RF_PATH, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
N = len(i_raw)
t_rf = np.arange(N) / FS_RF

# 1. Centering and Phase Extraction
i_c, q_c, xc_fit, yc_fit, R_fit = iq_condition_circle(i_raw, q_raw)

# Raw lowpass filter set to 300 Hz to fully preserve clinical high-freq snapping clicking bands
sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
iq_c = sosfiltfilt(sos_lp, -i_raw + 1j * q_raw)
mag_clean = np.abs(iq_c)
phi = robust_phase(i_c, q_c)

# 2. Downsample Magnitude and Phase to 1 kHz
magnitude_ds = decimate(mag_clean, DEC, ftype='fir')
phase_ds = decimate(phi, DEC, ftype='fir') * SCALE  # mm
t_ds = np.arange(len(phase_ds)) / FS_HR

# Center magnitude and phase using highpass filter (0.5 Hz)
sos_hp05 = butter(4, 0.5, btype='highpass', fs=FS_HR, output='sos')
magnitude_ac = sosfiltfilt(sos_hp05, magnitude_ds)
phase_ac = sosfiltfilt(sos_hp05, phase_ds)

# 3. Extract Cardiac Heartbeats (0.4–3.0 Hz)
sos_hr = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
mag_hr_10k = sosfiltfilt(sos_hr, mag_clean)
phase_hr_10k = sosfiltfilt(sos_hr, phi) * SCALE

mag_hr = decimate(mag_hr_10k, DEC, ftype='fir')
phase_hr = decimate(phase_hr_10k, DEC, ftype='fir')

# Rolling compliance envelopes (1.5s smoothing)
mag_hr_env = smooth(np.abs(signal.hilbert(mag_hr)), int(1.5 * FS_HR))
phase_hr_env = smooth(np.abs(signal.hilbert(phase_hr)), int(1.5 * FS_HR))

# 4. Extract High-Frequency Korotkoff Clicks (10–200 Hz)
sos_koro = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
mag_koro_10k = sosfiltfilt(sos_koro, mag_clean)
phase_koro_10k_filt = sosfiltfilt(sos_koro, phi)
phase_koro_10k = np.append(np.diff(phase_koro_10k_filt) * FS_RF, 0.0) * SCALE  # mm/s

mag_koro = decimate(mag_koro_10k, DEC, ftype='fir')
phase_koro = decimate(phase_koro_10k, DEC, ftype='fir')

# High-freq envelopes (0.5s smoothing)
mag_koro_env = smooth(np.abs(signal.hilbert(mag_koro)), int(0.5 * FS_HR))
phase_koro_env = smooth(np.abs(signal.hilbert(phase_koro)), int(0.5 * FS_HR))

# 5. Heart Rate PSD Count
idx_active = (t_ds >= K_ON) & (t_ds <= K_OFF)
f_hr, psd_hr_mag = welch(mag_hr[idx_active], fs=FS_HR, nperseg=len(mag_hr[idx_active]), nfft=32768)
f_hr_p, psd_hr_ph = welch(phase_hr[idx_active], fs=FS_HR, nperseg=len(phase_hr[idx_active]), nfft=32768)
hr_band = (f_hr >= 0.5) & (f_hr <= 2.5)

hr_bpm_mag = f_hr[hr_band][np.argmax(psd_hr_mag[hr_band])] * 60
hr_bpm_ph = f_hr_p[hr_band][np.argmax(psd_hr_ph[hr_band])] * 60

# 6. Locate Compliance MAP Peaks inside middle 70% of active window
mid_s = K_ON + 0.15 * (K_OFF - K_ON)
mid_e = K_OFF - 0.15 * (K_OFF - K_ON)
mask_mid = (t_ds >= mid_s) & (t_ds <= mid_e)
t_map_mag = t_ds[mask_mid][np.argmax(mag_hr_env[mask_mid])]
t_map_phase = t_ds[mask_mid][np.argmax(phase_hr_env[mask_mid])]

# Calibrate cuff pressures
P_start = 155.0
target_sbp = 125.0
target_dbp = 75.0
beta_active = (target_sbp - target_dbp) / (K_OFF - K_ON)
map_mmhg_mag = target_sbp - beta_active * (t_map_mag - K_ON)
map_mmhg_phase = target_sbp - beta_active * (t_map_phase - K_ON)

# 7. Normalize waveforms by maximum in deflation phase for visual balance
max_mag = np.max(np.abs(mag_hr[t_ds >= DEFL])) + 1e-20
max_phase = np.max(np.abs(phase_hr[t_ds >= DEFL])) + 1e-20
max_mag_k = np.max(np.abs(mag_koro[t_ds >= DEFL])) + 1e-20
max_phase_k = np.max(np.abs(phase_koro[t_ds >= DEFL])) + 1e-20

mag_hr_n = mag_hr / max_mag
mag_hr_env_n = mag_hr_env / max_mag
phase_hr_n = phase_hr / max_phase
phase_hr_env_n = phase_hr_env / max_phase

mag_koro_n = mag_koro / max_mag_k
mag_koro_env_n = mag_koro_env / max_mag_k
phase_koro_n = phase_koro / max_phase_k
phase_koro_env_n = phase_koro_env / max_phase_k

# ── PLOT DUAL-MODALITY 4x2 DASHBOARD ──────────────────────────────
print("Plotting premium 4x2 magnitude vs phase dashboard...")
fig, axes = plt.subplots(4, 2, figsize=(20, 24), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Styling Helper
def style_ax_extra_large(ax, title, xlabel, ylabel):
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

# Shading Zones
def add_zone_shading(ax):
    ax.axvspan(0.0, DEFL, color='#F8F9F9', alpha=0.6, zorder=0)
    ax.axvspan(DEFL, K_ON, color='#EBEDEF', alpha=0.4, zorder=0)
    ax.axvspan(K_ON, K_OFF, color=C_HIGHLIGHT, alpha=0.5, zorder=0)

# ===================================================================
# ROW 1: HIGH-FREQUENCY KOROTKOFF CLICKS (10–200 Hz) & ENVELOPES
# ===================================================================
# Subplot 1A: Magnitude Clicks (Left)
ax1A = axes[0, 0]
style_ax_extra_large(ax1A, "(A) RF Magnitude: High-Frequency Snapping Clicks (10–200 Hz)", "Time (s)", "Normalized Value (a.u.)")
add_zone_shading(ax1A)
ax1A.plot(t_ds, mag_koro_n, color=C_MAGNITUDE, lw=0.5, alpha=0.75, label='Magnitude Snapping clicks')
ax1A.plot(t_ds, mag_koro_env_n, color=C_MAGNITUDE_ENV, lw=1.8, label='Snapping Envelope')
ax1A.axvline(DEFL, color=C_TEXT, ls=':', lw=1.5)
ax1A.set_xlim([0, T_MAX])
ax1A.set_ylim([-0.05, 1.45])
ax1A.legend(loc='upper right', fontsize=9.5)
ax1A.text(0.02, 0.95, "Magnitude high-frequency clicks derived from spatial\nmodulations of clutter reflection during cuff deflation.", 
          transform=ax1A.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# Subplot 1B: Phase Clicks / Velocity (Right)
ax1B = axes[0, 1]
style_ax_extra_large(ax1B, "(B) RF Phase Displacement: High-Frequency Snapping Velocity (10–200 Hz)", "Time (s)", "Normalized Value (a.u.)")
add_zone_shading(ax1B)
ax1B.plot(t_ds, phase_koro_n, color=C_PHASE, lw=0.5, alpha=0.75, label='Phase Velocity snaps')
ax1B.plot(t_ds, phase_koro_env_n, color=C_PHASE_ENV, lw=1.8, label='Snapping Envelope')
ax1B.axvline(DEFL, color=C_TEXT, ls=':', lw=1.5)
ax1B.set_xlim([0, T_MAX])
ax1B.set_ylim([-0.05, 1.45])
ax1B.legend(loc='upper right', fontsize=9.5)
ax1B.text(0.02, 0.95, "Phase velocity snaps represent true sub-millimeter\narterial physical wall micro-velocity transients.", 
          transform=ax1B.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# ===================================================================
# ROW 2: ZOOMED OVERLAYS (8-second highly-detailed local normalized zoom for absolute visibility)
# ===================================================================
# Subplot 2A: Magnitude Normalized Zoom (Left)
ax2A = axes[1, 0]
style_ax_extra_large(ax2A, "(C) Zoomed Magnitude compliance vs. Snaps (Normalized Zoom)", "Time (s)", "Normalized Amplitude (a.u.)")
t_zoom_start = K_ON  # Start exactly at SBP onset (27.375 s)
t_zoom_end   = K_ON + 8.0  # Show exactly 8 seconds of detailed pulses
zoom_mask    = (t_ds >= t_zoom_start) & (t_ds <= t_zoom_end)
t_zoom       = t_ds[zoom_mask]

# Locally normalize to fill the vertical axis beautifully, ensuring the heartbeat compliance is 100% visible
zoom_hb_m = mag_hr_n[zoom_mask] / np.max(np.abs(mag_hr_n[zoom_mask]))
zoom_sn_m = mag_koro_n[zoom_mask] / np.max(np.abs(mag_koro_n[zoom_mask]))

# Plot slow compliance heartbeat in solid charcoal black
ax2A.plot(t_zoom, zoom_hb_m, color=C_HEARTBEAT, lw=2.2, label='Heartbeat compliance (0.4-3 Hz)')
# Plot snaps in soft sage steel blue for maximum desaturated contrast
ax2A.plot(t_zoom, zoom_sn_m, color=C_MAGNITUDE, lw=1.0, alpha=0.75, label='Korotkoff Snaps (10-200 Hz)')
ax2A.set_xlim([t_zoom_start, t_zoom_end])
ax2A.set_ylim([-1.05, 1.05])
ax2A.legend(loc='upper right', fontsize=9.5)
ax2A.text(0.02, 0.95, "Detail view showing magnitude heartbeat compliance wave locally\nnormalized to swing dynamically between -1 and 1 on the y-axis,\noverlaid with high-frequency snaps.", 
          transform=ax2A.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# Subplot 2B: Phase Normalized Zoom (Right)
ax2B = axes[1, 1]
style_ax_extra_large(ax2B, "(D) Zoomed Phase displacement vs. Snaps (Normalized Zoom)", "Time (s)", "Normalized Amplitude (a.u.)")

# Locally normalize to fill the vertical axis beautifully, ensuring the heartbeat displacement is 100% visible
zoom_hb_p = phase_hr_n[zoom_mask] / np.max(np.abs(phase_hr_n[zoom_mask]))
zoom_sn_p = phase_koro_n[zoom_mask] / np.max(np.abs(phase_koro_n[zoom_mask]))

# Plot slow heartbeat displacement in solid charcoal black
ax2B.plot(t_zoom, zoom_hb_p, color=C_HEARTBEAT, lw=2.2, label='Heartbeat displacement (0.4-3 Hz)')
# Plot snaps in soft terracotta rose for maximum desaturated contrast
ax2B.plot(t_zoom, zoom_sn_p, color=C_PHASE, lw=1.0, alpha=0.75, label='Korotkoff Snaps (10-200 Hz)')
ax2B.set_xlim([t_zoom_start, t_zoom_end])
ax2B.set_ylim([-1.05, 1.05])
ax2B.legend(loc='upper right', fontsize=9.5)
ax2B.text(0.02, 0.95, "Detail view showing physical phase displacement wave locally\nnormalized to swing dynamically between -1 and 1 on the y-axis,\noverlaid with high-frequency micro-velocity transients.", 
          transform=ax2B.transAxes, fontsize=10.5, va='top', bbox=TBOX)


# ===================================================================
# ROW 3: PHYSICAL DISPLACEMENT VS. TIME ANALYSIS (0 to 47s)
# ===================================================================
# Subplot 3A: Magnitude AC Displacement Analysis (Left)
ax3A = axes[2, 0]
style_ax_extra_large(ax3A, "(E) Magnitude AC Displacement Analysis vs. Time", "Time (s)", "Magnitude AC Displacement (a.u.)")
add_zone_shading(ax3A)

# Plot raw centered Magnitude AC signal in soft light grey
ax3A.plot(t_ds, magnitude_ac, color=C_RAW_DRIFT, lw=1.0, alpha=0.55, label='Centered Raw Magnitude')
# Plot slow Magnitude heartbeat compliance pulse (0.4-3 Hz) in solid dark steel blue
ax3A.plot(t_ds, mag_hr, color=C_MAGNITUDE_ENV, lw=1.8, label='Arterial Heartbeat (0.4–3 Hz)')
# Plot envelope in soft steel blue
ax3A.plot(t_ds, mag_hr_env, color=C_MAGNITUDE, lw=2.2, ls='--', label='Compliance Envelope')

# MAP & Peak Cuff markers
ax3A.plot(DEFL, mag_hr[np.argmin(np.abs(t_ds - DEFL))], 'D', color='crimson', ms=8, mec='black', mew=0.8, zorder=7)
ax3A.plot(t_map_mag, mag_hr_env[np.argmin(np.abs(t_ds - t_map_mag))], '^', color='orange', ms=10, mec='black', mew=0.8, zorder=6)

ax3A.set_xlim([0, T_MAX])
ax3A.set_ylim([np.min(magnitude_ac) - 0.05, np.max(magnitude_ac) + 0.15])
ax3A.legend(loc='upper right', fontsize=9.5)
ax3A.text(0.02, 0.95, f"Magnitude compliance peak at t={t_map_mag:.2f} s\n"
                     f"Equivalent MAP = {map_mmhg_mag:.1f} mmHg\n"
                     f"Magnitude HR = {hr_bpm_mag:.1f} BPM", 
          transform=ax3A.transAxes, fontsize=10.5, va='top', bbox=TBOX)

# Subplot 3B: Phase Physical Displacement Analysis (Right)
ax3B = axes[2, 1]
style_ax_extra_large(ax3B, "(F) Physical Phase displacement Analysis vs. Time", "Time (s)", "Physical Displacement (mm)")
add_zone_shading(ax3B)

# Plot raw centered unwrapped Phase displacement in soft light grey
ax3B.plot(t_ds, phase_ac, color=C_RAW_DRIFT, lw=1.0, alpha=0.55, label='Centered Unwrapped Phase (mm)')
# Plot isolated physical Arterial Heartbeat Displacement (0.4-3 Hz) in soft dark crimson
ax3B.plot(t_ds, phase_hr, color=C_PHASE_ENV, lw=1.8, label='Arterial Heartbeat displacement (mm)')
# Plot compliance envelope in soft terracotta
ax3B.plot(t_ds, phase_hr_env, color=C_PHASE, lw=2.2, ls='--', label='Compliance Envelope (mm)')

# MAP & Peak Cuff markers in physical units
ax3B.plot(DEFL, phase_hr[np.argmin(np.abs(t_ds - DEFL))], 'D', color='crimson', ms=8, mec='black', mew=0.8, zorder=7)
ax3B.plot(t_map_phase, phase_hr_env[np.argmin(np.abs(t_ds - t_map_phase))], '*', color='gold', ms=12, mec='black', mew=0.8, zorder=6)

ax3B.set_xlim([0, T_MAX])
ax3B.set_ylim([np.min(phase_ac) - 0.2, np.max(phase_ac) + 0.6])
ax3B.legend(loc='upper right', fontsize=9.5)
ax3B.text(0.02, 0.95, f"Phase compliance peak at t={t_map_phase:.2f} s\n"
                     f"Equivalent MAP = {map_mmhg_phase:.1f} mmHg\n"
                     f"Phase HR = {hr_bpm_ph:.1f} BPM", 
          transform=ax3B.transAxes, fontsize=10.5, va='top', bbox=TBOX)


# ===================================================================
# ROW 4: SPECTROGRAM / STFT ANALYSIS (10–200 Hz)
# ===================================================================
# Subplot 4A: Magnitude Spectrogram (Left)
ax4A = axes[3, 0]
style_ax_extra_large(ax4A, "(G) Magnitude Korotkoff Clicks Spectrogram", "Time (s)", "Frequency (Hz)")
mag_koro_600 = signal.decimate(mag_koro_10k, 16, ftype='fir')  # downsample to ~625 Hz
f_sp_m, t_sp_m, Sxx_sp_m = spectrogram(mag_koro_600, fs=625, nperseg=128, noverlap=96)
Sxx_db_m = 10 * np.log10(Sxx_sp_m + 1e-12)
im_m = ax4A.pcolormesh(t_sp_m, f_sp_m, Sxx_db_m, shading='gouraud', cmap='viridis', vmin=-110, vmax=-10)
plt.colorbar(im_m, ax=ax4A, label='Spectral Density [dB]').ax.tick_params(labelsize=8)
add_zone_shading(ax4A)
ax4A.set_xlim([0, T_MAX])
ax4A.set_ylim([10, 200])

# Subplot 4B: Phase Spectrogram (Right)
ax4B = axes[3, 1]
style_ax_extra_large(ax4B, "(H) Phase Displacement Korotkoff Clicks Spectrogram", "Time (s)", "Frequency (Hz)")
phase_koro_600 = signal.decimate(phase_koro_10k_filt, 16, ftype='fir')
f_sp_p, t_sp_p, Sxx_sp_p = spectrogram(phase_koro_600, fs=625, nperseg=128, noverlap=96)
Sxx_db_p = 10 * np.log10(Sxx_sp_p + 1e-12)
im_p = ax4B.pcolormesh(t_sp_p, f_sp_p, Sxx_db_p, shading='gouraud', cmap='viridis', vmin=-110, vmax=-10)
plt.colorbar(im_p, ax=ax4B, label='Spectral Density [dB]').ax.tick_params(labelsize=8)
add_zone_shading(ax4B)
ax4B.set_xlim([0, T_MAX])
ax4B.set_ylim([10, 200])


# Shading legend patches at the bottom
ph_patches = [
    mpatches.Patch(color='#F8F9F9', alpha=0.8, label=f'Phase I: Inflation → Peak Cuff Pressure ({P_start:.0f} mmHg) at t={DEFL}s'),
    mpatches.Patch(color='#EBEDEF', alpha=0.8, label='Phase II-A: Pre-SBP Occluded deflation leak region'),
    mpatches.Patch(color=C_HIGHLIGHT, alpha=0.8, label=f'Phase II-B: Active Korotkoff compliance window [{K_ON:.3f} s – {K_OFF:.2f} s] (Dur = {K_OFF-K_ON:.3f} s)'),
]
fig.legend(handles=ph_patches, loc='lower center', ncol=3, fontsize=11,
           framealpha=0.97, edgecolor='#BDC3C7', bbox_to_anchor=(0.5, -0.012))

# Sup Title
fig.suptitle("RF Radar RMG Demodulation Comparison: Magnitude vs. Phase displacement Domains\n"
             "Subject 2 (Rajveer), Rec 04  |  High-Fidelity 4x2 Demodulation & Absolute displacement Analysis Dashboard  |  300 DPI",
             color=C_TEXT, fontsize=18, fontweight='bold', y=0.985)

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
plt.subplots_adjust(hspace=0.26, wspace=0.18)

plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium Demodulation Comparison Dashboard saved successfully to: {OUT}")
