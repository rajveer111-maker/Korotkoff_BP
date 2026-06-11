import h5py, os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, decimate
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'rf_cuff_with_signals_sub1_rec6.png')

FS_RF = 10_000
DEC = 10
FS = FS_RF // DEC
FC = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000) / (4.0 * np.pi)

# -- known parameters for Sub 1 Rec 6 --
K_ON = 27.530
K_OFF = 43.330
SBP = 125.0
MAP = 110.0
DBP = 75.0
K_DUR = K_OFF - K_ON

# Helpers
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

print("Loading RF...")
rf_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
with h5py.File(rf_path, 'r') as f:
    rf_data = f['data'][:]
i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc

sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
rf_pulse = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
t_rf = np.arange(len(rf_pulse)) / FS
rf_env = env_smooth(rf_pulse, 1.5, FS)

print("Loading Stethoscope...")
wav_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')
fs_a, audio = wavfile.read(wav_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

audio_filt = bpf(audio, 50.0, 1000.0, fs_a)
audio_env_high = np.abs(hilbert(audio_filt))
steth_pulse_a = bpf(audio_env_high, 0.4, 3.0, fs_a)
steth_env_a = env_smooth(steth_pulse_a, 1.5, fs_a)

t_a = np.arange(len(audio)) / fs_a

steth_pulse = np.interp(t_rf, t_a, steth_pulse_a)
steth_env = np.interp(t_rf, t_a, steth_env_a)

print("Normalizing...")
valid_mask = (t_rf >= 0) & (t_rf <= 50)
rf_norm_factor = np.max(rf_env[valid_mask]) + 1e-10
st_norm_factor = np.max(steth_env[valid_mask]) + 1e-10

rf_env_norm = rf_env / rf_norm_factor
rf_pulse_norm = rf_pulse / rf_norm_factor

st_env_norm = steth_env / st_norm_factor
st_pulse_norm = steth_pulse / st_norm_factor

print("Simulating Cuff Pressure...")
P_start = 155.0
t_peak = 18.0
beta_defl = (SBP - DBP) / K_DUR
P_full_open = 60.0
t_open = K_OFF + (DBP - P_full_open) / beta_defl

cuff_p = np.where(
    t_rf < t_peak,
    (P_start / t_peak) * t_rf,  # inflation ramp
    np.where(
        t_rf <= K_ON,
        P_start - ((P_start - SBP)/(K_ON - t_peak)) * (t_rf - t_peak),
        SBP - beta_defl * (t_rf - K_ON)
    )
)

print("Plotting...")
plt.rcParams.update({'font.family': 'DejaVu Sans'})
fig, ax1 = plt.subplots(figsize=(16, 7), dpi=300, facecolor='#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

ax1.axvspan(0, t_peak, color='#E5E7E9', alpha=0.5, label='Inflation Phase (~18 s)')
ax1.axvspan(t_peak, K_ON, color='#E5E7E9', alpha=0.5, label='Occluded (above SBP)')
ax1.axvspan(K_ON, K_OFF, color='#FEF9E7', alpha=0.8, label=f'Korotkoff Region ({K_DUR:.1f} s)')

ax1.fill_between(t_rf, 0, rf_env_norm, color='#E91E63', alpha=0.08, zorder=2)
ax1.plot(t_rf, rf_env_norm, color='#E91E63', lw=2.2, zorder=3, label='RF Pulse Amplitude Envelope (Positive)')
ax1.plot(t_rf, rf_pulse_norm, color='#880E4F', lw=1.2, alpha=1.0, zorder=4, label='RF Heartbeat Pulses (zero-mean)')

# Negative envelope for symmetry (like in user's image)
ax1.plot(t_rf, -rf_env_norm, color='#F5B7B1', lw=1.0, alpha=0.5, ls='-.', zorder=3, label='RF Pulse Envelope (Negative)')

ax1.plot(t_rf, st_env_norm, color='#2196F3', lw=2.0, ls='--', zorder=3, label='Stethoscope Amplitude Envelope (Positive)')
ax1.plot(t_rf, st_pulse_norm, color='#0D47A1', lw=1.2, alpha=1.0, ls=':', zorder=4, label='Stethoscope Heartbeat Pulses (zero-mean)')

ax2 = ax1.twinx()
ax2.plot(t_rf, cuff_p, color='#7F8C8D', lw=2.0, label='Cuff Pressure (mmHg)')

# Add horizontal arrows/lines for Inflation and Deflation text
y_arrow = -0.8
ax1.annotate('', xy=(0, y_arrow), xytext=(t_peak, y_arrow), arrowprops=dict(arrowstyle='<->', color='#5D6D7E', lw=1.2))
ax1.text(t_peak/2, y_arrow + 0.05, 'Inflation\n~18 s', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2C3E50')

ax1.annotate('', xy=(t_peak, y_arrow), xytext=(49, y_arrow), arrowprops=dict(arrowstyle='<->', color='#5D6D7E', lw=1.2))
ax1.text((t_peak+49)/2, y_arrow + 0.05, 'Deflation\n~30 s', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2C3E50')

# Markers and Vertical Lines
ax2.vlines(t_peak, 0, P_start, colors='#7F8C8D', linestyles='--', lw=1.5, zorder=1)
ax2.plot(t_peak, P_start, 'v', color='#7F8C8D', ms=7, mec='black', zorder=10)
ax2.text(t_peak, P_start - 10, f'Occluded\n{P_start:.0f} mmHg', color='#515A5A', ha='center', va='top', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#7F8C8D', boxstyle='round,pad=0.2'))

ax2.vlines(K_ON, 0, SBP, colors='#E74C3C', linestyles='--', lw=1.5, zorder=1)
ax2.plot(K_ON, SBP, 'o', color='#E74C3C', ms=7, mec='black', zorder=10)
ax1.text(K_ON, 1.15, f'SBP\n{SBP:.0f} mmHg', color='#E74C3C', ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#E74C3C', boxstyle='round,pad=0.2'))

t_map = K_ON + (SBP - MAP) / beta_defl
ax2.vlines(t_map, 0, MAP, colors='#F39C12', linestyles='--', lw=1.5, zorder=1)
ax2.plot(t_map, MAP, '*', color='#F39C12', ms=13, mec='black', zorder=10)
ax1.text(t_map, 1.15, f'MAP (max HR)\n{MAP:.0f} mmHg', color='#E88800', ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#F39C12', boxstyle='round,pad=0.2'))

ax2.vlines(K_OFF, 0, DBP, colors='#2980B9', linestyles='--', lw=1.5, zorder=1)
ax2.plot(K_OFF, DBP, 's', color='#2980B9', ms=7, mec='black', zorder=10)
ax1.text(K_OFF, 1.15, f'DBP\n{DBP:.0f} mmHg', color='#2980B9', ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2980B9', boxstyle='round,pad=0.2'))

ax2.vlines(t_open, 0, P_full_open, colors='#27AE60', linestyles=':', lw=1.5, zorder=1)
ax2.plot(t_open, P_full_open, 'D', color='#27AE60', ms=7, mec='black', zorder=10)
ax1.text(t_open, 1.15, f'Full Open\n{P_full_open:.0f} mmHg', color='#27AE60', ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#27AE60', boxstyle='round,pad=0.2'))

ax1.set_xlim([0, 49])
ax1.set_ylim([-1.15, 1.45])
ax2.set_ylim([0, 185])

ax1.set_xlabel('Time (Sec.)', fontsize=12, fontweight='bold', color='#2C3E50')
ax1.set_ylabel('Normalized Heartbeat Pulse Amplitude (a.u.)', fontsize=12, fontweight='bold', color='#2C3E50')
ax2.set_ylabel('Cuff Pressure (mmHg)', fontsize=12, fontweight='bold', color='#7F8C8D')

ax1.grid(color='#E5E7E9', linestyle='-', linewidth=0.5, alpha=0.7)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left', ncol=2, fontsize=8.5, framealpha=0.95, edgecolor='#BDC3C7')

plt.title(f'Korotkoff Detection — Heartbeat Pulse Amplitude vs. Cuff Pressure\nSubject: Prof. Kan (Sub 1) | Rec 06 | MAP = {MAP:.0f} mmHg (adaptive max HR) | Korotkoff Duration: {K_DUR:.1f} s | 300 DPI', fontsize=13, fontweight='bold', color='#191970', pad=15)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print("\nDONE:", OUT)
