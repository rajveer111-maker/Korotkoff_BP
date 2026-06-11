import h5py, os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# ── PUBLICATION STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9.5,
    'legend.framealpha': 0.92,
    'lines.linewidth':   1.4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#E8E8E8',
    'grid.linewidth':    0.6,
})

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE    = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
RF_PATH = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
OUT     = os.path.join(BASE, 'rf_time_domain_2x2_Sub1_Rec6.png')

FS_RF     = 10_000
DEC       = 10
FS        = FS_RF // DEC
FC        = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE     = LAMBDA_MM / (4.0 * np.pi)

K_ON  = 27.7
K_OFF = 43.5
DEFL  = 18.0
T_MAX = 49.0
koro_dur = K_OFF - K_ON
ZOOM_L = max(0.0, K_ON - 7.0)
ZOOM_R = min(T_MAX, K_OFF + 1.5)

# ── COLORS ────────────────────────────────────────────────────────────────
CM   = '#1A6FC4'   # blue   – Magnitude
CP   = '#C0392B'   # red    – Phase
CE   = '#1A1A2E'   # dark   – Envelope
CK   = '#F39C12'   # amber  – Korotkoff lines
CKFILL = '#FEF9EC' # light amber fill
BG   = '#FFFFFF'
CTXT = '#1C1C1C'

# ── HELPERS ───────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def add_koro(ax):
    ax.axvspan(K_ON, K_OFF, color=CKFILL, alpha=0.85, zorder=0)
    ax.axvline(K_ON,  color=CK, lw=1.4, ls='--', zorder=2)
    ax.axvline(K_OFF, color=CK, lw=1.4, ls='--', zorder=2)
    ax.axvline(DEFL, color='#888888', lw=0.9, ls=':', zorder=1)

def yzoom(sig, mask, pad=1.25):
    lo = np.min(sig[mask]) * pad if np.min(sig[mask]) < 0 else np.min(sig[mask]) / pad
    hi = np.max(np.abs(sig[mask])) * pad
    return lo, hi

# ── LOAD ──────────────────────────────────────────────────────────────────
print("Loading RF data ...")
with h5py.File(RF_PATH, 'r') as f:
    rf = f['data'][:]
i_raw, q_raw = -rf[0, :], rf[1, :]

xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c  = i_raw - xc, q_raw - yc

phi_raw = robust_phase(i_c, q_c)

sos_lp  = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))

mag_ds = decimate(mag_raw, DEC, ftype='fir')
phi_ds = decimate(phi_raw, DEC, ftype='fir')
t      = np.arange(len(mag_ds)) / FS

# ── SIGNAL PROCESSING ─────────────────────────────────────────────────────
# Heart Rate Band 0.4-3 Hz
mag_dc       = np.mean(mag_ds)
mag_disp_au  = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
mag_disp     = (mag_disp_au / mag_dc) * SCALE
mag_disp_env = env_smooth(mag_disp, 1.5, FS)

phi_disp     = decimate(bpf(phi_raw, 0.4, 3.0, FS_RF) * SCALE, DEC, ftype='fir')
phi_disp_env = env_smooth(phi_disp, 1.5, FS)

# Korotkoff Band 10-200 Hz (Time Domain)
mag_koro = decimate(bpf(mag_raw, 10, 200, FS_RF), DEC, ftype='fir')
# For phase we typically look at velocity for high freq components
phi_vel  = decimate(np.append(np.diff(bpf(phi_raw, 10, 200, FS_RF))*FS_RF, 0.0)*SCALE, DEC, ftype='fir')

# ── FIGURE (2x2) ──────────────────────────────────────────────────────────
print("Building 2x2 Time-Domain Filter Dashboard ...")
fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=300, facecolor=BG)
fig.patch.set_facecolor(BG)
zm = (t >= ZOOM_L) & (t <= ZOOM_R)

# ── ROW 1: Heart Rate (0.4 - 3 Hz)
ax = axes[0, 0]
ax.set_title('(A)  Magnitude — Heart Rate Band (0.4–3 Hz)')
add_koro(ax)
ax.plot(t, mag_disp,     color=CM, lw=0.8, alpha=0.70, label='Compliance pulse (mm)')
ax.plot(t, mag_disp_env, color=CE, lw=2.2, ls='--',   label='RMS envelope (1.5 s)')
yc_lo, yc_hi = yzoom(mag_disp, zm)
ax.set_xlim([0, T_MAX]); ax.set_ylim([yc_lo, yc_hi])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement (mm)')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='upper left')

ax = axes[0, 1]
ax.set_title('(B)  Phase — Heart Rate Band (0.4–3 Hz)')
add_koro(ax)
ax.plot(t, phi_disp,     color=CP, lw=0.8, alpha=0.70, label='Displacement (mm)')
ax.plot(t, phi_disp_env, color=CE, lw=2.2, ls='--',   label='RMS envelope (1.5 s)')
yd_lo, yd_hi = yzoom(phi_disp, zm)
ax.set_xlim([0, T_MAX]); ax.set_ylim([yd_lo, yd_hi])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement (mm)')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='upper left')

# ── ROW 2: Korotkoff Band (10 - 200 Hz)
ax = axes[1, 0]
ax.set_title('(C)  Magnitude — Korotkoff High-Frequency Band (10–200 Hz)')
add_koro(ax)
ax.plot(t, mag_koro, color=CM, lw=0.8, alpha=0.85, label='Korotkoff filtered (a.u.)')
ym_lo, ym_hi = yzoom(mag_koro, zm, pad=1.5)
ax.set_xlim([0, T_MAX]); ax.set_ylim([ym_lo, ym_hi])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (a.u.)')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='upper left')

ax = axes[1, 1]
ax.set_title('(D)  Phase Velocity — Korotkoff High-Frequency Band (10–200 Hz)')
add_koro(ax)
ax.plot(t, phi_vel, color=CP, lw=0.8, alpha=0.85, label='Phase velocity (rad/s)')
yp_lo, yp_hi = yzoom(phi_vel, zm, pad=1.5)
ax.set_xlim([0, T_MAX]); ax.set_ylim([yp_lo, yp_hi])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (rad/s)')
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(loc='upper left')

fig.suptitle(
    'RF Radar Radiomyography: Time-Domain Validation (Heart Rate vs. Korotkoff Band)\n'
    'Subject 1 (Prof. Kan) | Rec 06 | Korotkoff window: {:.2f}–{:.2f} s  '
    '(Duration = {:.2f} s)'.format(K_ON, K_OFF, koro_dur),
    fontsize=13, fontweight='bold', color=CTXT, y=0.99)

plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.97])
plt.subplots_adjust(hspace=0.35, wspace=0.20)
plt.savefig(OUT, dpi=300, facecolor=BG, bbox_inches='tight')
print("\nDONE: " + OUT)
