"""
RMG Korotkoff Validation – Definitive 4-Row x 2-Col Figure (300 DPI)
=====================================================================
Per subject (2 columns), 4 rows:
  Row 1 – Physiological Heartbeat Modulation Waveforms (0.9 - 2.5 Hz)
           → proves the low-frequency pulses match beat-by-beat
  Row 2 – Beat-by-Beat Heart Rate (BPM) Step-Plots
           → tracks heart rate sync throughout the Korotkoff window
  Row 3 – RF Phase Velocity STFT Spectrogram (30-200 Hz)
           → visualizes the high-frequency snapping energy bursts directly
  Row 4 – Cumulative CUSUM Energy S-Curves
           → demonstrates energy release profile parity and bounds SBP/DBP
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import (butter, sosfiltfilt, decimate, hilbert,
                           filtfilt, iirnotch, fftconvolve, find_peaks, spectrogram)
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12.5,
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 13.5, 'axes.titleweight': 'bold',
    'legend.fontsize': 11, 'lines.linewidth': 2.0,
    'axes.grid': True, 'grid.color': '#F0F0F0', 'grid.linewidth': 0.9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FS_100 = 100
FC    = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000.0) / (4.0 * np.pi)  # rad -> mm factor

CP  = '#C0392B'   # Red     – RF Phase
CM  = '#8E44AD'   # Purple  – RF Magnitude
CS  = '#2980B9'   # Blue    – Stethoscope GT
CW  = '#F39C12'   # Amber   – Korotkoff zone
CBT = '#16A085'   # Teal    – beat markers

SESSIONS = [
    dict(sub_dir='Sub_1_Prof_kan', label='Subject 1 (Prof. Kan)  |  Rec 06',
         rec=6, k_on=27.53, k_off=43.33, zoom=(22, 48)),
    dict(sub_dir='Sub_2_Rajveer', label='Subject 2 (Rajveer)  |  Rec 04',
         rec=4, k_on=27.38, k_off=42.00, zoom=(22, 46)),
]

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    k = max(1, int(w * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def cusum_detect(env, t, lo=0.08, hi=0.92):
    mask = (t >= 20) & (t <= 52)
    ev   = smooth(env[mask], 3.0, FS)  # 3s macro-smooth bridges inter-beat gaps
    ts   = t[mask]
    if len(ev) == 0 or ev.max() == 0: return 22.0, 42.0
    cs   = np.cumsum(ev) / (ev.sum() + 1e-12)
    ion  = np.where(cs >= lo)[0]
    ioff = np.where(cs >= hi)[0]
    return (float(ts[ion[0]])  if len(ion)  else 22.0,
            float(ts[ioff[0]]) if len(ioff) else 42.0)

def get_cusum_curve(env, t):
    mask = (t >= 20) & (t <= 52)
    ev   = smooth(env[mask], 3.0, FS)
    cs   = np.cumsum(ev) / (ev.sum() + 1e-12)
    curve = np.zeros_like(t)
    curve[mask] = cs
    curve[t < 20] = 0.0
    curve[t > 52] = 1.0
    return curve

# ─── Load & process ─────────────────────────────────────────────────────────
print("Processing sessions...")
datasets = []
for s in SESSIONS:
    print(f"  {s['label']}...")
    rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    wp = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")

    with h5py.File(rp, 'r') as f: raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc);  ic -= xc;  qc -= yc

    # 1. High-Frequency Korotkoff Envelopes (30-200 Hz)
    mag_raw   = np.sqrt(ic**2 + qc**2)
    mag_clean = nf(nf(nf(mag_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    mag_k     = bpf(mag_clean, 30, 200, FS_RF)
    mag_dec   = decimate(smooth(tkeo(mag_k), 0.15, FS_RF), DEC, ftype='fir')

    phi     = robust_phase(ic, qc)
    phi     = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    vel_hi  = np.append(np.diff(bpf(phi, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
    t_rf    = np.arange(len(vel_dec)) / FS

    # Decimate velocity for fast spectrogram
    vel_hi_1k = decimate(vel_hi, 10, ftype='fir')
    f_sp, t_sp, Sxx = spectrogram(vel_hi_1k, fs=1000, nperseg=128, noverlap=110)
    f_mask = (f_sp >= 30) & (f_sp <= 200)
    f_sp_crop = f_sp[f_mask]
    Sxx_crop = Sxx[f_mask, :]

    # Load Stethoscope
    fs_a, aud = wavfile.read(wp)
    aud = aud.astype(np.float64) / 32768.0
    if aud.ndim > 1: aud = aud.mean(1)
    st_bp    = bpf(aud, 30, 1000, fs_a)
    st_hilb  = np.abs(hilbert(st_bp))
    st_koro  = bpf(st_hilb, 20, min(200, fs_a/2-1), fs_a)
    st_fine_a = smooth(tkeo(st_koro), 0.15, fs_a)
    st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)

    st_fine   = np.interp(t_rf, np.arange(len(st_fine_a))/fs_a, st_fine_a)
    st_wide   = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)

    k_on = s['k_on'];  k_off = s['k_off']

    # 2. Low-Frequency Heartbeat Modulation (0.9 - 2.5 Hz) at 100 Hz
    # Downsample raw signals to 100 Hz
    phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
    mag_100 = decimate(decimate(mag_raw, 10, ftype='fir'), 10, ftype='fir')
    t_100   = np.arange(len(phi_100)) / FS_100

    # Filter at 100 Hz
    phi_hr = bpf(phi_100, 0.9, 2.5, FS_100)
    mag_hr = bpf(mag_100, 0.9, 2.5, FS_100)

    # Stethoscope envelope BPF at 100 Hz
    st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
    st_hr = bpf(st_env_100, 0.9, 2.5, FS_100)

    # Local normalization & peak detection inside Korotkoff window
    mask_k = (t_100 >= k_on) & (t_100 <= k_off)
    st_k   = st_hr[mask_k]
    phi_k  = phi_hr[mask_k]
    mag_k  = mag_hr[mask_k]

    st_k_n  = st_k / (np.max(np.abs(st_k)) + 1e-12)
    phi_k_n = phi_k / (np.max(np.abs(phi_k)) + 1e-12)
    mag_k_n = mag_k / (np.max(np.abs(mag_k)) + 1e-12)

    min_dist = int(FS_100 * 0.5)
    # Detect peaks on Steth, -Phase, and Mag
    pks_s, _ = find_peaks(st_k_n, distance=min_dist, prominence=0.12)
    pks_p, _ = find_peaks(-phi_k_n, distance=min_dist, prominence=0.12)
    pks_m, _ = find_peaks(mag_k_n, distance=min_dist, prominence=0.12)

    t_k = t_100[mask_k]
    t_s_pks = t_k[pks_s]
    t_p_pks = t_k[pks_p]
    t_m_pks = t_k[pks_m]

    # CUSUM durations and curves
    vel_wide = smooth(np.maximum(vel_dec, 0), 1.5, FS)
    mag_wide = smooth(np.maximum(mag_dec, 0), 1.5, FS)
    
    vel_cusum = get_cusum_curve(vel_wide, t_rf)
    mag_cusum = get_cusum_curve(mag_wide, t_rf)
    st_cusum  = get_cusum_curve(st_wide, t_rf)

    vel_on, vel_off = cusum_detect(vel_wide, t_rf)
    mag_on, mag_off = cusum_detect(mag_wide, t_rf)

    datasets.append(dict(**s, t=t_rf, t_100=t_100, mask_k=mask_k,
                         st_hr=st_hr, phi_hr=phi_hr, mag_hr=mag_hr,
                         t_s_pks=t_s_pks, t_p_pks=t_p_pks, t_m_pks=t_m_pks,
                         f_sp=f_sp_crop, t_sp=t_sp, Sxx_sp=Sxx_crop,
                         st_cusum=st_cusum, vel_cusum=vel_cusum, mag_cusum=mag_cusum,
                         vel_on=vel_on, vel_off=vel_off, vel_dur=vel_off-vel_on,
                         mag_on=mag_on, mag_off=mag_off, mag_dur=mag_off-mag_on,
                         st_dur=k_off-k_on))

# ─── Figure: 4 rows x 2 cols ────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 25), dpi=300, facecolor='white')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.15,
                       height_ratios=[1.1, 1.0, 1.1, 1.2])

for col, ds in enumerate(datasets):
    t      = ds['t']
    t_100  = ds['t_100']
    k_on   = ds['k_on'];  k_off = ds['k_off']
    zl, zr = ds['zoom']

    def shade(ax):
        ax.axvspan(k_on, k_off, color=CW, alpha=0.12, zorder=0)
        ax.axvline(k_on,  color=CW, ls='--', lw=2.0, zorder=3)
        ax.axvline(k_off, color=CW, ls='--', lw=2.0, zorder=3)

    # ── Row 1: Physiological Heartbeat Modulation Waveforms (0.9 - 2.5 Hz) ──
    ax0 = fig.add_subplot(gs[0, col])
    
    # Locally normalize within zoom window for visualization
    mask_z = (t_100 >= zl) & (t_100 <= zr)
    st_z  = ds['st_hr'][mask_z] / np.max(np.abs(ds['st_hr'][mask_z]))
    phi_z = -ds['phi_hr'][mask_z] / np.max(np.abs(ds['phi_hr'][mask_z])) # Inverted phase
    mag_z = ds['mag_hr'][mask_z] / np.max(np.abs(ds['mag_hr'][mask_z]))

    ax0.plot(t_100[mask_z], st_z, color=CS, lw=2.2, label='Steth Envelope Modulation')
    ax0.plot(t_100[mask_z], phi_z, color=CP, lw=2.2, label='RF Phase (Inverted)')
    ax0.plot(t_100[mask_z], mag_z, color=CM, lw=2.2, ls='--', label='RF Magnitude')

    # Draw peak markers
    ax0.plot(ds['t_s_pks'], np.ones_like(ds['t_s_pks']) * 1.05, 'v', color=CS, ms=8, label='Steth Peaks')
    ax0.plot(ds['t_p_pks'], np.ones_like(ds['t_p_pks']) * 1.20, 'o', color=CP, ms=8, label='RF Phase Peaks')
    ax0.plot(ds['t_m_pks'], np.ones_like(ds['t_m_pks']) * 1.35, 'd', color=CM, ms=8, label='RF Mag Peaks')

    shade(ax0)
    ax0.set_xlim(zl, zr);  ax0.set_ylim(-1.3, 1.55)
    ax0.set_ylabel('Norm. Amplitude', fontsize=13)
    ax0.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax0.tick_params(labelbottom=False)
    ax0.set_title(f"{ds['label']}\n"
                  f"Row 1: Physiological Heartbeat Pulses (0.9-2.5 Hz) — Beat-by-Beat Alignment",
                  fontsize=13, loc='left')

    # ── Row 2: Heart Rate Step-Plots ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, col])
    
    bpm_s = 60.0 / np.diff(ds['t_s_pks'])
    bpm_p = 60.0 / np.diff(ds['t_p_pks'])
    bpm_m = 60.0 / np.diff(ds['t_m_pks'])

    ax1.step(ds['t_s_pks'][:-1], bpm_s, color=CS, lw=2.5, where='post', marker='o', ms=6, label=f'Steth (mean={np.mean(bpm_s):.1f} BPM)')
    ax1.step(ds['t_p_pks'][:-1], bpm_p, color=CP, lw=2.5, where='post', marker='x', ms=6, label=f'RF Phase (mean={np.mean(bpm_p):.1f} BPM)')
    ax1.step(ds['t_m_pks'][:-1], bpm_m, color=CM, lw=2.5, where='post', marker='d', ms=6, ls='--', label=f'RF Mag (mean={np.mean(bpm_m):.1f} BPM)')

    shade(ax1)
    ax1.set_xlim(zl, zr);  ax1.set_ylim(48, 98)
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=13)
    ax1.legend(loc='upper right', ncol=3, framealpha=0.9)
    ax1.tick_params(labelbottom=False)
    ax1.set_title("Row 2: Instantaneous Heart Rate (BPM) Step-Plots — Pulse Synchronization",
                  fontsize=13, loc='left')

    # ── Row 3: RF Phase Velocity STFT Spectrogram (30-200 Hz) ────────────────
    ax2 = fig.add_subplot(gs[2, col])
    
    S_db = 10 * np.log10(ds['Sxx_sp'] + 1e-12)
    vmin = np.max(S_db) - 40
    vmax = np.max(S_db)
    
    im = ax2.pcolormesh(ds['t_sp'], ds['f_sp'], S_db, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax2, pad=0.01, aspect=15)
    cbar.set_label('Power (dB)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Vertical beat marker lines projected from Steth peaks
    for tp in ds['t_s_pks']:
        ax2.axvline(tp, color=CBT, lw=1.2, ls='--', alpha=0.55)

    shade(ax2)
    ax2.set_xlim(zl, zr);  ax2.set_ylim(30, 200)
    ax2.set_ylabel('Frequency (Hz)', fontsize=13)
    ax2.tick_params(labelbottom=False)
    ax2.set_title("Row 3: RF Phase Velocity STFT Spectrogram — Heartbeat-Locked Snapping Bursts",
                  fontsize=13, loc='left')

    # ── Row 4: Cumulative CUSUM Energy S-Curves ─────────────────────────────
    ax3 = fig.add_subplot(gs[3, col])
    
    # Plot CUSUM S-Curves
    ax3.plot(t, ds['st_cusum'],  color=CS, lw=2.8, label=f'Steth GT')
    ax3.plot(t, ds['vel_cusum'], color=CP, lw=2.8, label=f'RF Phase')
    ax3.plot(t, ds['mag_cusum'], color=CM, lw=2.8, ls='--', label=f'RF Mag')

    # Draw horizontal thresholds
    ax3.axhline(0.08, color='#7F8C8D', ls=':', lw=1.5)
    ax3.axhline(0.92, color='#7F8C8D', ls=':', lw=1.5)
    ax3.text(zl + 0.5, 0.08 + 0.02, 'Onset Threshold (8%)', fontsize=9, color='#7F8C8D')
    ax3.text(zl + 0.5, 0.92 + 0.02, 'Offset Threshold (92%)', fontsize=9, color='#7F8C8D')

    # Draw vertical ticks from threshold cross to X-axis
    ax3.vlines([k_on, ds['vel_on'], ds['mag_on']], 0.0, 0.08, colors=[CS, CP, CM], linestyles=':')
    ax3.vlines([k_off, ds['vel_off'], ds['mag_off']], 0.0, 0.92, colors=[CS, CP, CM], linestyles=':')

    shade(ax3)

    # Stacked duration brackets
    brackets = [
        (k_on, k_off, ds['st_dur'],  CS, f'Steth GT: {ds["st_dur"]:.1f} s'),
        (ds['vel_on'], ds['vel_off'], ds['vel_dur'], CP, f'RF Phase: {ds["vel_dur"]:.1f} s'),
        (ds['mag_on'], ds['mag_off'], ds['mag_dur'], CM, f'RF Mag: {ds["mag_dur"]:.1f} s'),
    ]
    for i, (on, off, dur, col_c, lbl) in enumerate(brackets):
        yb = 1.10 + i * 0.20
        ax3.annotate('', xy=(off, yb), xytext=(on, yb),
                     arrowprops=dict(arrowstyle='<->', color=col_c, lw=2.5))
        ax3.text((on+off)/2, yb + 0.03, lbl,
                 ha='center', va='bottom', fontsize=12,
                 color=col_c, fontweight='bold')

    ax3.set_xlim(zl, zr);  ax3.set_ylim(-0.05, 1.75)
    ax3.set_title("Row 4: Cumulative CUSUM Energy Curves — Bounded Deflation Boundary Parity",
                  fontsize=13, loc='left')
    ax3.set_xlabel('Time (s)', fontsize=14)
    ax3.set_ylabel('Cumulative Energy (0-1)', fontsize=13)
    ax3.legend(loc='lower right', framealpha=0.92)

fig.suptitle(
    'Near-Field RF (USRP) Radiomyography vs. Digital Acoustic Stethoscope\n'
    'Continuous Heartbeat Sync (Rows 1 & 2) and Korotkoff Snapping Validation (Rows 3 & 4)  |  300 DPI',
    fontsize=20, y=0.997, fontweight='bold'
)

OUT = os.path.join(BASE, 'rmg_korotkoff_final_proof.png')
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f'DONE -> {OUT}')
