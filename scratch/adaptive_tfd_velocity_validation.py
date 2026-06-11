import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, spectrogram
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'font.weight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'legend.fontsize': 11, 'lines.linewidth': 1.5,
    'axes.grid': True, 'grid.color': '#EEEEEE', 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'adaptive_tfd_velocity_validation.png')
FS_RF = 10000; DEC = 10; FS = 1000

CP = '#C0392B' # Red RF Phase
CS = '#2980B9' # Blue Steth

FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000
SCALE = LAMBDA_MM / (4.0 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def robust_phase(i_c, q_c):
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return -res[0]/2, -res[1]/2

sessions = [
    ('Sub_1_Prof_kan', 'Sub 1', 6, 27.53, 43.33, 24, 46),
    ('Sub_1_Prof_kan', 'Sub 1', 4, 25.50, 42.80, 23, 45),
    ('Sub_2_Rajveer', 'Sub 2', 4, 27.38, 42.00, 24, 45),
    ('Sub_2_Rajveer', 'Sub 2', 8, 20.00, 36.00, 18, 40)
]

fig = plt.figure(figsize=(26, 16), dpi=300, facecolor='white')
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1.5, 1, 1.5])

for col_idx, (sub_dir, sub_name, rec_idx, k_on, k_off, zoom_on, zoom_off) in enumerate(sessions):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    print(f"Processing {sub_name} Rec {rec_idx}...")
    
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    phi_raw = robust_phase(i_raw-xc, q_raw-yc)
    p = notch(notch(notch(phi_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    # Bandpass 30-200 Hz to explicitly isolate Korotkoff turbulence from the massive chest wall heartbeat motion
    vel_rf = np.append(np.diff(bpf(p, 30, 200, FS_RF))*FS_RF, 0.0) * SCALE
    t_rf = np.arange(len(vel_rf))/FS_RF
    
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64)/32768.0
    if audio.ndim>1: audio = audio.mean(axis=1)
    audio_f = bpf(audio, 50, 1000, fs_a)
    t_a = np.arange(len(audio_f))/fs_a
    
    # ── PANEL 1: Stethoscope Raw ──
    ax0 = fig.add_subplot(gs[0, col_idx])
    ax0.plot(t_a, audio_f, color=CS, alpha=0.9)
    ax0.set_xlim([zoom_on, zoom_off])
    ax0.set_ylim([-np.max(np.abs(audio_f[(t_a>zoom_on)&(t_a<zoom_off)])), np.max(np.abs(audio_f[(t_a>zoom_on)&(t_a<zoom_off)]))])
    ax0.set_title(f"{sub_name} Rec {rec_idx} | Stethoscope Audio", fontsize=15)
    ax0.axvspan(k_on, k_off, color='#F39C12', alpha=0.1, zorder=0)
    ax0.axvline(k_on, color='#F39C12', ls='--', lw=2)
    ax0.axvline(k_off, color='#F39C12', ls='--', lw=2)
    if col_idx == 0: ax0.set_ylabel("Amplitude")
    
    # ── PANEL 2: Stethoscope TFD ──
    ax1 = fig.add_subplot(gs[1, col_idx])
    f_a, t_spec_a, Sxx_a = spectrogram(audio_f, fs=fs_a, nperseg=int(fs_a*0.05), noverlap=int(fs_a*0.04), scaling='spectrum')
    mask_f_a = (f_a >= 50) & (f_a <= 1000)
    Sxx_a = Sxx_a[mask_f_a, :]
    f_a = f_a[mask_f_a]
    # Adaptive Scaling
    Sxx_a_koro = Sxx_a[:, (t_spec_a >= k_on) & (t_spec_a <= k_off)]
    vmax_a = np.percentile(Sxx_a_koro, 99)
    pcm1 = ax1.pcolormesh(t_spec_a, f_a, Sxx_a, shading='gouraud', cmap='magma', vmin=0, vmax=vmax_a)
    ax1.set_xlim([zoom_on, zoom_off])
    ax1.set_ylim([50, 1000])
    ax1.set_title("Adaptive TFD: Stethoscope (50 - 1000 Hz)")
    ax1.axvline(k_on, color='#FFFFFF', ls='--', lw=2)
    ax1.axvline(k_off, color='#FFFFFF', ls='--', lw=2)
    if col_idx == 0: ax1.set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm1, ax=ax1, pad=0.01).set_label("Power")

    # ── PANEL 3: RF Velocity ──
    ax2 = fig.add_subplot(gs[2, col_idx])
    ax2.plot(t_rf, vel_rf, color=CP, alpha=0.9)
    ax2.set_xlim([zoom_on, zoom_off])
    ylim_rf = np.percentile(np.abs(vel_rf[(t_rf>zoom_on)&(t_rf<zoom_off)]), 99.9)
    ax2.set_ylim([-ylim_rf, ylim_rf])
    ax2.set_title(f"{sub_name} Rec {rec_idx} | RF Velocity (Korotkoff Turbulence Band)", fontsize=15)
    ax2.axvspan(k_on, k_off, color='#F39C12', alpha=0.1, zorder=0)
    ax2.axvline(k_on, color='#F39C12', ls='--', lw=2)
    ax2.axvline(k_off, color='#F39C12', ls='--', lw=2)
    if col_idx == 0: ax2.set_ylabel("Velocity (mm/s)")
    
    # ── PANEL 4: RF TFD ──
    ax3 = fig.add_subplot(gs[3, col_idx])
    f_rf, t_spec_rf, Sxx_rf = spectrogram(vel_rf, fs=FS_RF, nperseg=int(FS_RF*0.1), noverlap=int(FS_RF*0.08), scaling='spectrum')
    mask_f_rf = (f_rf >= 30) & (f_rf <= 200)
    Sxx_rf = Sxx_rf[mask_f_rf, :]
    f_rf = f_rf[mask_f_rf]
    # Adaptive Scaling
    Sxx_rf_koro = Sxx_rf[:, (t_spec_rf >= k_on) & (t_spec_rf <= k_off)]
    vmax_rf = np.percentile(Sxx_rf_koro, 99)
    pcm3 = ax3.pcolormesh(t_spec_rf, f_rf, Sxx_rf, shading='gouraud', cmap='magma', vmin=0, vmax=vmax_rf)
    ax3.set_xlim([zoom_on, zoom_off])
    ax3.set_ylim([30, 200])
    ax3.set_title("Adaptive TFD: RF Velocity (30 - 200 Hz)")
    ax3.axvline(k_on, color='#FFFFFF', ls='--', lw=2)
    ax3.axvline(k_off, color='#FFFFFF', ls='--', lw=2)
    ax3.set_xlabel("Time (s)", fontsize=14)
    if col_idx == 0: ax3.set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm3, ax=ax3, pad=0.01).set_label("Power")

fig.suptitle('Validation via RF Phase Velocity & Adaptive Time-Frequency Distributions (TFD)\nDirect Visualization of Korotkoff Broadband Streaks in Acoustic and Radar Signals', fontsize=22, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.subplots_adjust(hspace=0.35, wspace=0.1)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"DONE: {OUT}")
