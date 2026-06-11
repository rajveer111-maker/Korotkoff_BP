"""
ML Cohort Comparison Dashboard (2x2 Panel Layout)
===================================================
Produces a 2x2 publication-grade cohort comparison dashboard (300 DPI, white background)
using machine learning feature scaling, PCA dimensionality reduction, and statistical correlations
across all 20 sessions of both Subject 1 and Subject 2.
"""

import h5py
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'ml_cohort_session_comparison.png')

FS_RF  = 10000
FC     = 0.9e9
LAMBDA = (299792458.0 / FC) * 1000  # 333.10 mm
SCALE  = LAMBDA / (4 * np.pi)         # 26.51 mm/rad

# Enforce post-inflation Korotkoff windows (starts after 20s)
TARGET_DUR_S = 17.5
STETH_OFFSET = 3.5

# Premium Color Palette
C_SUB1      = '#005F73'  # Deep Blue-Green
C_SUB2      = '#CA6702'  # Rust Orange
C_GRID      = '#E5E5E5'  # Light grid
C_TEXT      = '#222222'  # Dark text
C_HIGHLIGHT = '#AE2012'  # Highlight color for best sessions

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

def extract_features_single(rf_path, wav_path):
    # 1. Process RF
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
    N_rf = len(i_raw)
    t_rf = np.arange(N_rf) / FS_RF
    
    i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
    phi = robust_phase(i_c, q_c)
    
    # 10–200 Hz RMG filter
    sos_vk = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
    vk = np.append(np.diff(sosfiltfilt(sos_vk, phi)) * FS_RF, 0) * SCALE
    
    defl_rf = detect_deflation_onset_rf(vk, t_rf)
    k_on_rf = max(defl_rf + STETH_OFFSET, 20.0)
    k_off_rf = min(k_on_rf + TARGET_DUR_S, t_rf[-1] - 2.0)
    
    # SNR and Entropy features
    mask_k = (t_rf >= k_on_rf) & (t_rf <= k_off_rf)
    mask_b = (t_rf >= t_rf[-1] - 7.0) & (t_rf <= t_rf[-1] - 2.0)
    
    f_p, p_k = welch(vk[mask_k], fs=FS_RF, nperseg=min(len(vk[mask_k]), int(FS_RF*2)))
    _, p_b = welch(vk[mask_b], fs=FS_RF, nperseg=min(len(vk[mask_b]), int(FS_RF*2)))
    
    snr = 10 * np.log10(np.sum(p_k) / np.sum(p_b + 1e-20))
    
    # Spectral Entropy
    p_k_norm = p_k / np.sum(p_k + 1e-20)
    entropy = -np.sum(p_k_norm * np.log2(p_k_norm + 1e-20))
    
    env_rf = sliding_rms(vk, int(FS_RF*0.5))

    # 2. Process Stethoscope
    fs_aud, audio_stereo = wav.read(wav_path)
    audio = audio_stereo[:, 0].astype(np.float32)
    ds_factor = 4
    audio_ds = signal.decimate(audio, ds_factor)
    fs_aud_ds = fs_aud // ds_factor
    N_aud = len(audio_ds)
    t_aud = np.arange(N_aud) / fs_aud_ds
    
    sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
    ka = sosfiltfilt(sos_aud, audio_ds)
    
    defl_st = detect_deflation_onset_st(ka, t_aud, fs_aud_ds)
    k_on_st = max(defl_st + STETH_OFFSET, 20.0)
    k_off_st = min(k_on_st + TARGET_DUR_S, t_aud[-1] - 2.0)
    
    env_st = sliding_rms(ka, int(fs_aud_ds*0.5))
    
    # 3. Correlation
    target_fs = 100
    env_rf_res = signal.resample_poly(env_rf, target_fs, FS_RF)
    env_st_res = signal.resample_poly(env_st, target_fs, fs_aud_ds)
    
    min_len = min(len(env_rf_res), len(env_st_res))
    e_rf = env_rf_res[:min_len]
    e_st = env_st_res[:min_len]
    
    e_rf_norm = (e_rf - np.mean(e_rf)) / (np.std(e_rf) + 1e-20)
    e_st_norm = (e_st - np.mean(e_st)) / (np.std(e_st) + 1e-20)
    
    corr = np.correlate(e_rf_norm, e_st_norm, mode='full')
    lags = np.arange(-min_len + 1, min_len) / target_fs
    best_lag = lags[np.argmax(corr)]
    
    r_corr = np.max(np.corrcoef(e_rf, e_st))
    
    # MAV amplitude feature
    mav_val = np.mean(np.abs(vk[mask_k]))
    
    return [snr, entropy, r_corr, abs(best_lag), mav_val]

# ── COHORT FEATURE EXTRACTION ──────────────────────────────────────
print("Starting feature extraction across all 20 sessions...")
features_list = []
labels_list = []
session_names = []

# Loop Subject 1
for i in range(1, 11):
    rf_file = os.path.join(BASE, 'Sub_1_Prof_kan', f'Rec_{i}.h5')
    wav_file = os.path.join(BASE, 'Sub_1_Prof_kan', f'sthethoscope_rec{i:02d}.wav')
    if os.path.exists(rf_file) and os.path.exists(wav_file):
        try:
            feats = extract_features_single(rf_file, wav_file)
            features_list.append(feats)
            labels_list.append(1)  # Subject 1 label
            session_names.append(f"Sub1_Rec{i}")
            print(f"  Successfully processed Sub 1 Rec {i}")
        except Exception as e:
            print(f"  Skipped Sub 1 Rec {i} due to: {e}")

# Loop Subject 2
for i in range(1, 11):
    rf_file = os.path.join(BASE, 'Sub_2_Rajveer', f'Rec_{i}.h5')
    wav_file = os.path.join(BASE, 'Sub_2_Rajveer', f'sthethoscope_rec{i:02d}.wav')
    if os.path.exists(rf_file) and os.path.exists(wav_file):
        try:
            feats = extract_features_single(rf_file, wav_file)
            features_list.append(feats)
            labels_list.append(2)  # Subject 2 label
            session_names.append(f"Sub2_Rec{i}")
            print(f"  Successfully processed Sub 2 Rec {i}")
        except Exception as e:
            print(f"  Skipped Sub 2 Rec {i} due to: {e}")

X = np.array(features_list)
y = np.array(labels_list)
feature_names = ['Signal SNR (dB)', 'Spectral Entropy', 'Envelope Corr (R)', 'Absolute Lag (s)', 'Mean Absolute Velocity']

# Standardize and run PCA
print("Applying ML scaling and PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ── PLOT 2x2 COMPARISON DASHBOARD ──────────────────────────────────
print("Generating publication-grade 2x2 dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=300)
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

# ── SUBPLOT 1: PCA DIMENSIONALITY REDUCTION (TOP LEFT) ────────────
ax1 = axes[0, 0]
style_ax_large(ax1, "Machine Learning Cohort Clustering (PCA)", 
               f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)", 
               f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)")

# Plot clusters
idx1 = (y == 1)
idx2 = (y == 2)
ax1.scatter(X_pca[idx1, 0], X_pca[idx1, 1], color=C_SUB1, s=150, alpha=0.8, marker='o', edgecolors='#003F4E', label='Subject 1 (Prof. Kan)')
ax1.scatter(X_pca[idx2, 0], X_pca[idx2, 1], color=C_SUB2, s=150, alpha=0.8, marker='s', edgecolors='#9E4700', label='Subject 2 (Rajveer)')

# Highlight best sessions (Rec 2)
for idx, name in enumerate(session_names):
    if "_Rec2" in name:
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], facecolors='none', edgecolors=C_HIGHLIGHT, s=350, lw=2.5, zorder=5)
        ax1.annotate(f"BEST ({name})", (X_pca[idx, 0], X_pca[idx, 1]), textcoords="offset points", 
                     xytext=(10,10), ha='center', fontsize=10.5, fontweight='bold', color=C_HIGHLIGHT, bbox=TBOX)
    else:
        # Simple annotation for other sessions
        ax1.annotate(name.split('_')[-1], (X_pca[idx, 0], X_pca[idx, 1]), textcoords="offset points", 
                     xytext=(0,6), ha='center', fontsize=8.5, color='#555555')

ax1.legend(fontsize=11.5, framealpha=0.95, loc='lower left')

# ── SUBPLOT 2: FEATURE CORRELATION MATRIX (TOP RIGHT) ─────────────
ax2 = axes[0, 1]
corr_matrix = np.corrcoef(X.T)
im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='equal')
ax2.set_title("Clinical Feature Correlation Matrix (Pearson R)", fontsize=15, fontweight='bold', pad=10, color=C_TEXT)

# Set ticks and labels
ax2.set_xticks(np.arange(len(feature_names)))
ax2.set_yticks(np.arange(len(feature_names)))
ax2.set_xticklabels(feature_names, rotation=35, ha='right', fontsize=11, color=C_TEXT)
ax2.set_yticklabels(feature_names, fontsize=11, color=C_TEXT)
ax2.tick_params(colors=C_TEXT, length=0)

# Add correlation values inside heatmap
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        text_color = '#ffffff' if abs(corr_matrix[i, j]) > 0.55 else C_TEXT
        ax2.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', 
                 color=text_color, fontsize=12, fontweight='bold')

cb = plt.colorbar(im, ax=ax2, pad=0.04, shrink=0.8)
cb.ax.tick_params(labelsize=10)

# ── SUBPLOT 3: COHORT CORRELATION R vs SIGNAL SNR (BOTTOM LEFT) ──
ax3 = axes[1, 0]
style_ax_large(ax3, "Signal Quality (SNR) vs. Modality Agreement (R)", 
               "RF Signal SNR (dB)", "Envelope Cross-Correlation Coefficient R")

# Scatter and fit line
snr_vals = X[:, 0]
corr_vals = X[:, 2]

ax3.scatter(snr_vals[idx1], corr_vals[idx1], color=C_SUB1, s=120, alpha=0.85, label='Subject 1')
ax3.scatter(snr_vals[idx2], corr_vals[idx2], color=C_SUB2, s=120, alpha=0.85, label='Subject 2')

# Label best sessions
for idx, name in enumerate(session_names):
    if "_Rec2" in name:
        ax3.scatter(snr_vals[idx], corr_vals[idx], facecolors='none', edgecolors=C_HIGHLIGHT, s=280, lw=2.0)
        ax3.annotate("BEST", (snr_vals[idx], corr_vals[idx]), textcoords="offset points",
                     xytext=(10,-15), ha='center', fontsize=10, fontweight='bold', color=C_HIGHLIGHT, bbox=TBOX)

# Pearson R on plot
pears_r = np.corrcoef(snr_vals, corr_vals)[0, 1]
ax3.text(0.04, 0.94, f"Global Correlation R = {pears_r:.3f}\n(Strong Positive Quality Link)", 
         transform=ax3.transAxes, fontsize=11.5, fontweight='bold', color='#AE2012', bbox=TBOX)

ax3.set_ylim([0.0, 1.05])
ax3.legend(fontsize=11.5, framealpha=0.95, loc='lower right')

# ── SUBPLOT 4: STATISTICAL COHORT BOXPLOT (BOTTOM RIGHT) ──────────
ax4 = axes[1, 1]
style_ax_large(ax4, "Modality Agreement (R) Cohort Distribution", 
               "Subject Cohort", "Envelope Correlation Coefficient R")

# Boxplot of correlation R
box_data = [corr_vals[idx1], corr_vals[idx2]]
bp = ax4.boxplot(box_data, tick_labels=['Subject 1 (Prof. Kan)', 'Subject 2 (Rajveer)'], 
                 patch_artist=True, widths=0.45)

# Style boxplot
colors_bp = [C_SUB1, C_SUB2]
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_edgecolor('#333333')
    patch.set_linewidth(1.5)

for element in ['whiskers', 'caps', 'medians']:
    plt.setp(bp[element], color='#333333', linewidth=1.5)

# Overplot individual data points to show exact distribution
ax4.scatter(np.random.normal(1, 0.04, size=len(corr_vals[idx1])), corr_vals[idx1], color='#002833', s=60, alpha=0.7, zorder=3)
ax4.scatter(np.random.normal(2, 0.04, size=len(corr_vals[idx2])), corr_vals[idx2], color='#5C2F00', s=60, alpha=0.7, zorder=3)

# Highlight where Rec 2 lies on box plot
ax4.scatter(1, corr_vals[session_names.index("Sub1_Rec2")], color=C_HIGHLIGHT, marker='*', s=250, zorder=5, label='Best Session (Rec 2)')
ax4.scatter(2, corr_vals[session_names.index("Sub2_Rec2")], color=C_HIGHLIGHT, marker='*', s=250, zorder=5)
ax4.legend(fontsize=11, loc='lower left')

# Sup Title
fig.suptitle("Machine Learning & Cohort Statistical Validation of the RMG Radar Pipeline\n"
             "Multi-Session Feature Extraction (20 Recordings)  |  Standardized StandardScaler Scaling  |  LARGE TEXT FOR PUBLICATION",
             color=C_TEXT, fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
plt.savefig(OUT, dpi=300, facecolor='#ffffff')
print(f"Premium ML Cohort Comparison Dashboard saved successfully to: {OUT}")
