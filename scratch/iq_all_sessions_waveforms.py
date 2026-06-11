"""
All-Session Cohort Waveform Dashboard (3x2 Panel Layout)
=========================================================
Produces a 3x2 panel clinical publication-grade dashboard (300 DPI, white background)
overplotting baseband Magnitude and unwrapped Phase of I and Q across all 20 recordings
for both Subject 1 and Subject 2, with full-length recordings in all panels,
and explicit cohort heart rate (BPM) stats.
"""

import h5py
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'iq_all_sessions_waveforms.png')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Enforce post-inflation Korotkoff windows (starts after 20s)
TARGET_DUR_S = 17.5
STETH_OFFSET = 3.5

# Premium Color Palette
C_SUB1      = '#005F73'  # Deep Blue-Green for Sub 1
C_SUB2      = '#CA6702'  # Rust Orange for Sub 2
C_GRID      = '#E5E5E5'  # Light grid
C_TEXT      = '#222222'  # Dark text
C_MEAN1     = '#0A9396'  # Teal mean for Sub 1
C_MEAN2     = '#AE2012'  # Crimson mean for Sub 2

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

# ── LOAD & PROCESS COHORT WAVEFORMS ────────────────────────────────
def load_cohort_waveforms(subj_folder, name_label):
    print(f"Loading waveforms for {name_label}...")
    mags = []
    phases = []
    hbs = []
    vks = []
    hrs = []
    
    t_target = np.linspace(10, 52, 4000)  # Standardized time grid
    
    # Filters
    sos_k = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
    sos_hb = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    
    for i in range(1, 11):
        rf_file = os.path.join(BASE, subj_folder, f'Rec_{i}.h5')
        if os.path.exists(rf_file):
            try:
                with h5py.File(rf_file, 'r') as f:
                    rf_data = f['data'][:]
                i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
                N_rf = len(i_raw)
                t_rf = np.arange(N_rf) / FS_RF
                
                # Circle fit and Phase
                i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
                phi = robust_phase(i_c, q_c)
                mag = np.sqrt(i_raw**2 + q_raw**2)
                
                # Filter components
                vk = np.append(np.diff(sosfiltfilt(sos_k, phi)) * FS_RF, 0) * SCALE
                hb = sosfiltfilt(sos_hb, phi) * SCALE
                
                # Extract Heart Rate (BPM) inside active window
                defl_rf = detect_deflation_onset_rf(vk, t_rf)
                koro_on = max(defl_rf + STETH_OFFSET, 20.0)
                koro_off = min(koro_on + TARGET_DUR_S, t_rf[-1] - 2.0)
                
                mask_k = (t_rf >= koro_on) & (t_rf <= koro_off)
                t_k = t_rf[mask_k]
                hb_k = hb[mask_k]
                
                # Detect negative peaks for heart retraction beats
                peaks, _ = find_peaks(-hb_k, distance=int(FS_RF*0.55), prominence=np.std(hb_k)*0.4)
                if len(peaks) > 1:
                    intervals = np.diff(t_k[peaks])
                    valid_intervals = intervals[(intervals > 0.4) & (intervals < 1.5)]
                    if len(valid_intervals) > 0:
                        hr_val = 60.0 / np.median(valid_intervals)
                        hrs.append(hr_val)
                
                # Interpolate to common time grid to allow averaging
                m_interp = np.interp(t_target, t_rf, mag)
                p_interp = np.interp(t_target, t_rf, phi)
                hb_interp = np.interp(t_target, t_rf, hb)
                vk_interp = np.interp(t_target, t_rf, vk)
                
                mags.append(normalize(m_interp))
                phases.append(normalize(p_interp))
                hbs.append(normalize(hb_interp))
                vks.append(normalize(vk_interp))
                print(f"  Successfully processed Rec {i}")
            except Exception as e:
                print(f"  Skipped Rec {i} due to: {e}")
                
    return t_target, np.array(mags), np.array(phases), np.array(hbs), np.array(vks), np.array(hrs)

t_grid, mags1, phases1, hbs1, vks1, hrs1 = load_cohort_waveforms('Sub_1_Prof_kan', 'Subject 1 (Prof. Kan)')
_, mags2, phases2, hbs2, vks2, hrs2 = load_cohort_waveforms('Sub_2_Rajveer', 'Subject 2 (Rajveer)')

mean_hr1, std_hr1 = np.mean(hrs1), np.std(hrs1)
mean_hr2, std_hr2 = np.mean(hrs2), np.std(hrs2)

# ── PLOT 3x2 COHORT WAVEFORM DASHBOARD ─────────────────────────────
print("Generating publication-grade 3x2 cohort waveform figure...")
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
# ROW 1: BASEBAND MAGNITUDE (NCSam) FOR ALL RECORDINGS
# ===================================================================
# Subplot 1A: Sub 1 Magnitude
ax1A = axes[0, 0]
style_ax_large(ax1A, "Sub 1: Baseband Magnitude of I & Q (10 Sessions)", "Time (s)", "Normalized Magnitude (a.u.)")
for idx, m in enumerate(mags1):
    label = "Individual Sessions" if idx == 0 else ""
    ax1A.plot(t_grid, m, color=C_SUB1, lw=0.8, alpha=0.35, label=label)
ax1A.plot(t_grid, np.mean(mags1, axis=0), color=C_MEAN1, lw=2.2, label='Cohort Mean Magnitude')
ax1A.set_xlim([10, 52])
ax1A.set_ylim([-0.05, 1.05])
ax1A.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 1B: Sub 2 Magnitude
ax1B = axes[0, 1]
style_ax_large(ax1B, "Sub 2: Baseband Magnitude of I & Q (10 Sessions)", "Time (s)", "Normalized Magnitude (a.u.)")
for idx, m in enumerate(mags2):
    label = "Individual Sessions" if idx == 0 else ""
    ax1B.plot(t_grid, m, color=C_SUB2, lw=0.8, alpha=0.35, label=label)
ax1B.plot(t_grid, np.mean(mags2, axis=0), color=C_MEAN2, lw=2.2, label='Cohort Mean Magnitude')
ax1B.set_xlim([10, 52])
ax1B.set_ylim([-0.05, 1.05])
ax1B.legend(fontsize=10.5, framealpha=0.95, loc='upper right')


# ===================================================================
# ROW 2: UNWRAPPED PHASE (NCSph) FOR ALL RECORDINGS
# ===================================================================
# Subplot 2A: Sub 1 Phase
ax2A = axes[1, 0]
style_ax_large(ax2A, r"Sub 1: Centered Phase $\phi(t)$ of I & Q (10 Sessions)", "Time (s)", "Normalized Phase (a.u.)")
for idx, p in enumerate(phases1):
    label = "Individual Sessions" if idx == 0 else ""
    ax2A.plot(t_grid, p, color=C_SUB1, lw=0.8, alpha=0.35, label=label)
ax2A.plot(t_grid, np.mean(phases1, axis=0), color=C_MEAN1, lw=2.2, label=r'Cohort Mean Phase $\phi(t)$')
ax2A.set_xlim([10, 52])
ax2A.set_ylim([-0.05, 1.05])
ax2A.legend(fontsize=10.5, framealpha=0.95, loc='upper right')

# Subplot 2B: Sub 2 Phase
ax2B = axes[1, 1]
style_ax_large(ax2B, r"Sub 2: Centered Phase $\phi(t)$ of I & Q (10 Sessions)", "Time (s)", "Normalized Phase (a.u.)")
for idx, p in enumerate(phases2):
    label = "Individual Sessions" if idx == 0 else ""
    ax2B.plot(t_grid, p, color=C_SUB2, lw=0.8, alpha=0.35, label=label)
ax2B.plot(t_grid, np.mean(phases2, axis=0), color=C_MEAN2, lw=2.2, label=r'Cohort Mean Phase $\phi(t)$')
ax2B.set_xlim([10, 52])
ax2B.set_ylim([-0.05, 1.05])
ax2B.legend(fontsize=10.5, framealpha=0.95, loc='upper right')


# ===================================================================
# ROW 3: COHORT PHYSIOLOGICAL WAVES FOR THE FULL RECORDING LENGTH
# ===================================================================
# Subplot 3A: Sub 1 Filtered Waves (Bottom Left) - FULL RECORDING
ax3A = axes[2, 0]
style_ax_large(ax3A, "Sub 1: Cohort Physiological Waveforms (Full Recording)", "Time (s)", "Normalized Amplitude (a.u.)")
for h, v in zip(hbs1, vks1):
    ax3A.plot(t_grid, h, color=C_SUB1, lw=0.5, alpha=0.25)
    ax3A.plot(t_grid, v - 0.25, color='#83C5BE', lw=0.3, alpha=0.15)
# Overlay cohort means
ax3A.plot(t_grid, np.mean(hbs1, axis=0), color=C_MEAN1, lw=2.0, label='Mean Heartbeat (0.4-3.0 Hz)')
ax3A.plot(t_grid, np.mean(vks1, axis=0) - 0.25, color='#005F73', lw=1.2, label='Mean Korotkoff (10-200 Hz)')
ax3A.set_xlim([10, 52])
ax3A.set_ylim([-0.35, 1.35])
ax3A.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax3A.text(0.03, 0.93, f"Subject 1 Cohort Heart Rate:\n{mean_hr1:.1f} ± {std_hr1:.1f} BPM", transform=ax3A.transAxes, 
          fontsize=12, fontweight='bold', color=C_MEAN1, bbox=TBOX)

# Subplot 3B: Sub 2 Filtered Waves (Bottom Right) - FULL RECORDING
ax3B = axes[2, 1]
style_ax_large(ax3B, "Sub 2: Cohort Physiological Waveforms (Full Recording)", "Time (s)", "Normalized Amplitude (a.u.)")
for h, v in zip(hbs2, vks2):
    ax3B.plot(t_grid, h, color=C_SUB2, lw=0.5, alpha=0.25)
    ax3B.plot(t_grid, v - 0.25, color='#E9D8A6', lw=0.3, alpha=0.15)
# Overlay cohort means
ax3B.plot(t_grid, np.mean(hbs2, axis=0), color=C_MEAN2, lw=2.0, label='Mean Heartbeat (0.4-3.0 Hz)')
ax3B.plot(t_grid, np.mean(vks2, axis=0) - 0.25, color='#9E4700', lw=1.2, label='Mean Korotkoff (10-200 Hz)')
ax3B.set_xlim([10, 52])
ax3B.set_ylim([-0.35, 1.35])
ax3B.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax3B.text(0.03, 0.93, f"Subject 2 Cohort Heart Rate:\n{mean_hr2:.1f} ± {std_hr2:.1f} BPM", transform=ax3B.transAxes, 
          fontsize=12, fontweight='bold', color=C_MEAN2, bbox=TBOX)


# Sup Title and layout adjustment
fig.suptitle("Clinical Cohort Waveform Consistency Audit: Magnitude and Phase of I & Q\n"
             "Overplotting All 20 Sessions of Both Subject Cohorts  |  Normalized Waveforms [0, 1]  |  Full Recording x-Axes",
             color=C_TEXT, fontsize=17, fontweight='bold', y=0.985)

fig.text(0.02, 0.945, "PANEL 1: Baseband Magnitude of I and Q Overlaid across All Sessions", color=C_TEXT, fontsize=13.5, fontweight='bold')
fig.text(0.02, 0.645, "PANEL 2: Centered Unwrapped Phase of I and Q Overlaid across All Sessions", color=C_TEXT, fontsize=13.5, fontweight='bold')
fig.text(0.02, 0.345, "PANEL 3: Extracted Cohort Physiological Component Waveforms (Full Length 10s–52s)", color=C_TEXT, fontsize=13.5, fontweight='bold')

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium Cohort Waveform Dashboard saved successfully to: {OUT}")
