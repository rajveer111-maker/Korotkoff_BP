import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, detrend, spectrogram, hilbert, iirnotch, filtfilt
from scipy.io import wavfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'adaptive_spectrogram_validation.png')
FS_RF = 10000
DEC = 10
FS = FS_RF // DEC
FC = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

def smooth_energy(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.sqrt(np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same'))

def calc_tkeo(x):
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(tkeo, 0)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

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
    return detrend(phase, type='linear')

# Load Subject 1 Rec 6
rf_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
with h5py.File(rf_path, 'r') as f:
    rf_data = f['data'][:]

i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
xc, yc, R = fit_circle(i_raw, q_raw)
i_c, q_c = i_raw - xc, q_raw - yc
phi = robust_phase(i_c, q_c)

# 1. High-Frequency RF Micro-Velocity (10-200 Hz)
v_raw = np.append(np.diff(phi) * FS_RF, 0) * SCALE
v_dec = decimate(v_raw, DEC, ftype='fir')

sos_vk = butter(4, [20, 200], btype='band', fs=FS, output='sos')
vk_dec = sosfiltfilt(sos_vk, v_dec)

# Remove 50 Hz and 100 Hz electrical interference (EMI)
vk_dec = notch(vk_dec, 50.0, FS)
vk_dec = notch(vk_dec, 100.0, FS)
vk_dec = notch(vk_dec, 150.0, FS)

t = np.arange(len(vk_dec)) / FS
# Apply TKEO to highlight transient bursts and suppress background noise
rf_tkeo = calc_tkeo(vk_dec)
vk_env = smooth_energy(rf_tkeo, 1.5, FS)

# 2. Stethoscope Audio Processing
wav_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')
fs_a, audio = wavfile.read(wav_path)
audio = audio.astype(np.float64) / 32768.0
if audio.ndim > 1: audio = audio.mean(axis=1)

audio_filt = bpf(audio, 50.0, 1000.0, fs_a)
steth_env_high = np.abs(hilbert(audio_filt))
high_cut = min(200.0, (fs_a / 2) - 1.0)
steth_koro = bpf(steth_env_high, 20.0, high_cut, fs_a)

# Apply TKEO to stethoscope as well for morphological matching
steth_tkeo = calc_tkeo(steth_koro)
steth_env_raw = smooth_energy(steth_tkeo, 1.5, fs_a)

# 3. Spectrogram Computation
# RF Spectrogram
f_rf, t_rf_spec, Sxx_rf = spectrogram(vk_dec, fs=FS, nperseg=int(FS*0.5), noverlap=int(FS*0.45), scaling='spectrum')
mask_f_rf = (f_rf >= 10) & (f_rf <= 200)
Sxx_rf = Sxx_rf[mask_f_rf, :]
f_rf = f_rf[mask_f_rf]

# Stethoscope Spectrogram
f_a, t_a_spec, Sxx_a = spectrogram(audio, fs=fs_a, nperseg=int(fs_a*0.1), noverlap=int(fs_a*0.08), scaling='spectrum')
mask_f_a = (f_a >= 50) & (f_a <= 1000)
Sxx_a = Sxx_a[mask_f_a, :]
f_a = f_a[mask_f_a]

# Time-Domain Envelope Interpolation
t_a = np.arange(len(steth_env_raw)) / fs_a
steth_env = np.interp(t, t_a, steth_env_raw)

# 4. Adaptive Bounding via CUSUM on Time-Domain Envelopes
search_mask = (t > 22.0) & (t < 46.0)

# Normalize by the maximum in the plotted region and CLIP to fix "waveform goes outside"
plot_mask = (t > 18.0) & (t < t[-1] - 5.0)
rf_norm = np.clip(vk_env / np.max(vk_env[plot_mask]), 0, 1.1)
steth_norm = np.clip(steth_env / np.max(steth_env[plot_mask]), 0, 1.1)

def find_cusum_bounds(env, mask, t_arr, lower=0.15, upper=0.90):
    e = env[mask]
    t_m = t_arr[mask]
    c_sum = np.cumsum(e)
    if c_sum[-1] == 0: return 0, 0
    c_sum = c_sum / c_sum[-1]
    
    idx_on = np.where(c_sum >= lower)[0]
    idx_off = np.where(c_sum >= upper)[0]
    
    k_on = t_m[idx_on[0]] if len(idx_on) > 0 else 0
    k_off = t_m[idx_off[0]] if len(idx_off) > 0 else 0
    return k_on, k_off

rf_k_on, rf_k_off = find_cusum_bounds(rf_norm, search_mask, t, 0.15, 0.90)

# 5. Plotting
plt.rcParams.update({'font.family': 'sans-serif'})
fig, axs = plt.subplots(3, 1, figsize=(14, 12), dpi=300, sharex=True, facecolor='white', gridspec_kw={'height_ratios': [1, 1, 1.2]})

# Top Panel: Stethoscope Spectrogram (Ground Truth)
ax0 = axs[0]
Sxx_a_log = 10 * np.log10(Sxx_a + 1e-10)
pcm0 = ax0.pcolormesh(t_a_spec, f_a, Sxx_a_log, shading='gouraud', cmap='viridis', vmin=np.percentile(Sxx_a_log, 50), vmax=np.percentile(Sxx_a_log, 99))
ax0.axvspan(rf_k_on, rf_k_off, color='#E74C3C', alpha=0.2, label=f'RF Detected Window ({rf_k_on:.1f}s - {rf_k_off:.1f}s)')
ax0.axvline(rf_k_on, color='#E74C3C', lw=2, ls='--')
ax0.axvline(rf_k_off, color='#E74C3C', lw=2, ls='--')
ax0.set_title("Ground Truth: Stethoscope Acoustic Spectrogram (50 - 1000 Hz)", fontweight='bold', fontsize=14)
ax0.set_ylabel("Freq. (Hz)", fontweight='bold')
ax0.set_xlim([15, t[-1]-2])
ax0.set_ylim([50, 1000])
ax0.legend(loc='upper right', framealpha=0.9)
fig.colorbar(pcm0, ax=ax0, label='Audio (dB)')

# Middle Panel: RF Spectrogram
ax1 = axs[1]
Sxx_log = 10 * np.log10(Sxx_rf + 1e-10)
pcm1 = ax1.pcolormesh(t_rf_spec, f_rf, Sxx_log, shading='gouraud', cmap='magma', vmin=np.percentile(Sxx_log, 40), vmax=np.percentile(Sxx_log, 99))
ax1.axvspan(rf_k_on, rf_k_off, color='#E74C3C', alpha=0.2, label=f'RF Detected Window ({rf_k_on:.1f}s - {rf_k_off:.1f}s)')
ax1.axvline(rf_k_on, color='#E74C3C', lw=2, ls='--')
ax1.axvline(rf_k_off, color='#E74C3C', lw=2, ls='--')
ax1.set_title("Proposed Method: RF Radar Phase Spectrogram (10 - 200 Hz)", fontweight='bold', fontsize=14)
ax1.set_ylabel("Freq. (Hz)", fontweight='bold')
ax1.set_xlim([15, t[-1]-2])
ax1.set_ylim([10, 150])
ax1.legend(loc='upper right', framealpha=0.9)
fig.colorbar(pcm1, ax=ax1, label='RF Power (dB)')

# Bottom Panel: TKEO Enhanced Time-Domain Envelopes
ax2 = axs[2]

# Stethoscope Waveform
ax2.plot(t, steth_norm, color='#2980B9', lw=2.5, alpha=0.8, label='Stethoscope TKEO Envelope')

# RF Waveform
ax2.plot(t, rf_norm, color='#C0392B', lw=2.5, alpha=0.8, label='RF Phase TKEO Envelope')

# Universal Bounding Box
ax2.axvspan(rf_k_on, rf_k_off, color='#E74C3C', alpha=0.15, label=f'RF Algorithmic Window ({rf_k_on:.1f}s - {rf_k_off:.1f}s)')
ax2.axvline(rf_k_on, color='#E74C3C', lw=2, ls='--')
ax2.axvline(rf_k_off, color='#E74C3C', lw=2, ls='--')

ax2.set_title("Dual-Modality Time-Domain Alignment: TKEO Enhanced Transients & CUSUM", fontweight='bold', fontsize=14)
ax2.set_xlabel("Time (Sec.)", fontweight='bold')
ax2.set_ylabel("Normalized TKEO Energy", fontweight='bold')
ax2.set_xlim([15, t[-1]-2])
ax2.set_ylim([0, 1.15])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Saved to {OUT}")

print(f"\n[Algorithmic Results]")
print(f"True Clinical SBP (Expected): ~27.5s")
print(f"True Clinical DBP (Expected): ~43.3s")
print(f"RF Radar Algorithmic Bounds (Used for all panels): K_ON = {rf_k_on:.2f}s, K_OFF = {rf_k_off:.2f}s")

