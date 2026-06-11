"""
RF Phase vs Stethoscope: 2x2 Publication Layout
Focuses on Time-Domain (CUSUM Onset/Offset) and Frequency-Domain (Turbulence PSD)
"""

import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, welch, hilbert
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec

# ── PUBLICATION STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'font.weight':       'bold',
    'axes.labelsize':    14,
    'axes.labelweight':  'bold',
    'axes.titlesize':    15,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   11,
    'legend.framealpha': 0.92,
    'lines.linewidth':   1.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#E0E0E0',
    'grid.linewidth':    0.8,
})

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE    = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
WAV_PATH = os.path.join(BASE, 'Sub_2_Rajveer', 'sthethoscope_rec04.wav')
OUT     = os.path.join(BASE, 'rf_phase_vs_steth_2x2_Sub2_Rec4.png')

FS_RF     = 10_000
DEC       = 10
FS        = FS_RF // DEC
FC        = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE     = LAMBDA_MM / (4.0 * np.pi)

# Search window for CUSUM
SEARCH_ON = 22.0
SEARCH_OFF = 46.0
T_MAX = 51.0

# ── COLORS ────────────────────────────────────────────────────────────────
CP   = '#C0392B'   # red    – Phase
CS   = '#2980B9'   # blue   - Steth
CC   = '#27AE60'   # green  - CUSUM
CKFILL = '#FEF9EC' # light amber fill
BG   = '#FFFFFF'
CTXT = '#1C1C1C'

# ── HELPERS ───────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - res[2])

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def find_cusum_bounds(env, t_arr, search_mask, lower=0.15, upper=0.90):
    e = env[search_mask]
    t_m = t_arr[search_mask]
    c_sum = np.cumsum(e)
    if c_sum[-1] == 0: return 0, 0, np.zeros_like(e), t_m
    c_sum = c_sum / c_sum[-1]
    
    idx_on = np.where(c_sum >= lower)[0]
    idx_off = np.where(c_sum >= upper)[0]
    
    k_on = t_m[idx_on[0]] if len(idx_on) > 0 else 0
    k_off = t_m[idx_off[0]] if len(idx_off) > 0 else 0
    return k_on, k_off, c_sum, t_m

# ── LOAD RF ───────────────────────────────────────────────────────────────
print("Loading RF data ...")
with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c  = i_raw - xc, q_raw - yc
phi_raw = robust_phase(i_c, q_c)

# Deep clean Phase before derivative
phi_clean = notch(phi_raw, 64.0, FS_RF, Q=30)
phi_clean = notch(phi_clean, 100.6, FS_RF, Q=30)
phi_clean = notch(phi_clean, 50.0, FS_RF, Q=30)

phi_vel_rf = np.append(np.diff(bpf(phi_clean, 10, 200, FS_RF))*FS_RF, 0.0)*SCALE
phi_tkeo_env_rf = smooth_energy(calc_tkeo(phi_vel_rf), 1.5, FS_RF)
phi_tkeo_env = decimate(phi_tkeo_env_rf, DEC, ftype='fir')
t_rf = np.arange(len(phi_tkeo_env)) / FS

# ── LOAD STETHOSCOPE ──────────────────────────────────────────────────────
print("Loading Stethoscope audio ...")
fs_a, audio = wavfile.read(WAV_PATH)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

audio_filt = bpf(audio, 50.0, 1000.0, fs_a)
steth_koro = bpf(np.abs(hilbert(audio_filt)), 20.0, min(200.0, (fs_a/2)-1), fs_a)
steth_tkeo_env_raw = smooth_energy(calc_tkeo(steth_koro), 1.5, fs_a)
t_a_raw = np.arange(len(steth_tkeo_env_raw)) / fs_a

# Interpolate steth to FS for easy alignment
steth_tkeo_env = np.interp(t_rf, t_a_raw, steth_tkeo_env_raw)

# ── NORMALIZE TO CLINICAL DURATION ────────────────────────────────────────
K_ON = 27.380
K_OFF = 42.000
koro_dur = K_OFF - K_ON

search_mask = (t_rf >= K_ON) & (t_rf <= K_OFF)
base_mask = (t_rf >= 15.0) & (t_rf <= K_ON - 2.0)

def norm_env(env):
    b_min = np.percentile(env[base_mask], 5)
    e_shifted = np.maximum(env - b_min, 0)
    return e_shifted / (np.max(e_shifted[search_mask]) + 1e-10)

rf_norm = norm_env(phi_tkeo_env)
steth_norm = norm_env(steth_tkeo_env)

# ── FREQUENCY DOMAIN (PSD) ────────────────────────────────────────────────
# Use the explicit Clinical Window for the Korotkoff PSD mask
mask_koro_rf = (np.arange(len(phi_vel_rf))/FS_RF >= K_ON) & (np.arange(len(phi_vel_rf))/FS_RF <= K_OFF)
mask_base_rf = (np.arange(len(phi_vel_rf))/FS_RF >= 15.0) & (np.arange(len(phi_vel_rf))/FS_RF <= K_ON - 2.0)
f_psd_rf, pxx_koro_rf = welch(phi_vel_rf[mask_koro_rf], fs=FS_RF, nperseg=int(FS_RF*1.0))
_, pxx_base_rf = welch(phi_vel_rf[mask_base_rf], fs=FS_RF, nperseg=int(FS_RF*1.0))
mask_f_rf = (f_psd_rf >= 10) & (f_psd_rf <= 200)
f_psd_rf = f_psd_rf[mask_f_rf]
pxx_koro_rf = 10 * np.log10(pxx_koro_rf[mask_f_rf] + 1e-20)
pxx_base_rf = 10 * np.log10(pxx_base_rf[mask_f_rf] + 1e-20)

# Stethoscope PSD
mask_koro_st = (np.arange(len(audio))/fs_a >= K_ON) & (np.arange(len(audio))/fs_a <= K_OFF)
mask_base_st = (np.arange(len(audio))/fs_a >= 15.0) & (np.arange(len(audio))/fs_a <= K_ON - 2.0)
f_psd_st, pxx_koro_st = welch(audio[mask_koro_st], fs=fs_a, nperseg=int(fs_a*0.25))
_, pxx_base_st = welch(audio[mask_base_st], fs=fs_a, nperseg=int(fs_a*0.25))
mask_f_st = (f_psd_st >= 20) & (f_psd_st <= 400)
f_psd_st = f_psd_st[mask_f_st]
pxx_koro_st = 10 * np.log10(pxx_koro_st[mask_f_st] + 1e-20)
pxx_base_st = 10 * np.log10(pxx_base_st[mask_f_st] + 1e-20)

# Normalize PSDs to [0, 1] for direct shape comparison
pxx_koro_rf_n = (pxx_koro_rf - np.min(pxx_koro_rf)) / (np.max(pxx_koro_rf) - np.min(pxx_koro_rf))
pxx_koro_st_n = (pxx_koro_st - np.min(pxx_koro_st)) / (np.max(pxx_koro_st) - np.min(pxx_koro_st))

# ── PLOTTING ──────────────────────────────────────────────────────────────
print("Building 3-panel overlay dashboard ...")
fig = plt.figure(figsize=(18, 12), dpi=300, facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

def draw_bounds(ax, on, off, color):
    ax.axvspan(on, off, color=color, alpha=0.1, zorder=0)
    ax.axvline(on, color=color, lw=2, ls='--', zorder=2)
    ax.axvline(off, color=color, lw=2, ls='--', zorder=2)
    dur = off - on
    y = ax.get_ylim()[1] * 0.9
    ax.annotate('', xy=(off, y), xytext=(on, y), arrowprops=dict(arrowstyle='<->', color=color, lw=2))
    ax.text((on+off)/2, y + 0.05, f' {dur:.2f} s', va='bottom', ha='center', color=color, fontsize=12, fontweight='bold')

# PANEL A: RF Phase TKEO + Clinical Window
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title(f'(A) Proposed Method: RF Phase TKEO Envelope (Duration: {koro_dur:.2f}s)')
ax0.plot(t_rf, rf_norm, color=CP, lw=2.0, alpha=0.9, label='RF Phase TKEO Envelope')
ax0.set_xlim([15, T_MAX]); ax0.set_ylim([0, 1.2])
ax0.set_xlabel('Time (s)'); ax0.set_ylabel('Normalized Energy')
ax0.grid(True, alpha=0.3)
draw_bounds(ax0, K_ON, K_OFF, CP)
ax0.legend(loc='upper right')

# PANEL B: Stethoscope TKEO + Clinical Window
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title(f'(B) Ground Truth: Stethoscope Acoustic TKEO Envelope (Duration: {koro_dur:.2f}s)')
ax1.plot(t_rf, steth_norm, color=CS, lw=2.0, alpha=0.9, label='Stethoscope TKEO Envelope')
ax1.set_xlim([15, T_MAX]); ax1.set_ylim([0, 1.2])
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Normalized Energy')
ax1.grid(True, alpha=0.3)
draw_bounds(ax1, K_ON, K_OFF, CS)
ax1.legend(loc='upper right')

# PANEL C: Cross-Modality Frequency Overlay (PSD)
ax2 = fig.add_subplot(gs[1, :])
ax2.set_title(f'(C) Cross-Modality Turbulence Overlay: Normalized Frequency PSD during Korotkoff Window ({K_ON:.1f}s - {K_OFF:.1f}s)')

# Highlight the primary matching bandwidth
overlap_band_low = 30
overlap_band_high = 120
ax2.axvspan(overlap_band_low, overlap_band_high, color='#F39C12', alpha=0.15, label='Primary Turbulence Overlap Band (30-120 Hz)')

ax2.plot(f_psd_rf, pxx_koro_rf_n, color=CP, lw=3.0, alpha=0.9, label='RF Phase Velocity Turbulence (Normalized)')
ax2.fill_between(f_psd_rf, 0, pxx_koro_rf_n, color=CP, alpha=0.1)

ax2.plot(f_psd_st, pxx_koro_st_n, color=CS, lw=3.0, alpha=0.8, ls='--', label='Stethoscope Acoustic Turbulence (Normalized)')
ax2.fill_between(f_psd_st, 0, pxx_koro_st_n, color=CS, alpha=0.1)

ax2.set_xlim([20, 200])
ax2.set_ylim([0, 1.05])
ax2.set_xlabel('Turbulence Frequency (Hz)', fontweight='bold')
ax2.set_ylabel('Normalized Spectral Power', fontweight='bold')
ax2.grid(True, alpha=0.4, ls=':')
ax2.legend(loc='upper right', fontsize=12)

# SUPTITLE
fig.suptitle(
    'Radiomyography Cross-Modality Validation: Time-Domain Duration and Frequency Overlap Analysis\n'
    f'Subject 2 (Rajveer) | Rec 04 | True Clinical Duration = {koro_dur:.2f} s ({K_ON:.2f} s - {K_OFF:.2f} s)',
    fontsize=18, fontweight='bold', color=CTXT, y=0.98)

plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
plt.subplots_adjust(hspace=0.25, wspace=0.15)

plt.savefig(OUT, dpi=300, facecolor=BG, bbox_inches='tight')
print("\nDONE: " + OUT)
