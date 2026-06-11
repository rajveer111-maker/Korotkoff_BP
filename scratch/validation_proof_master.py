"""
Acousto-RF Sensing — 4-Point Scientific Validation (CORRECTED)
================================================================
Panel A: RF Baseband Power Spectrum (Body vs Table) — confirms ultrasound sidebands
Panel B: Physiological-Band PSD (0.5–3 Hz) — confirms cardiac signal in Body only
Panel C: Clean demodulated waveform (Body vs Table, best pair, 4s window)
Panel D: Spectral Noise Floor ratio — quantifies SNR across all 4 pairs

RF carrier: 900 MHz  |  Ultrasound DDC: f0 ≈ 100.71 Hz  |  Sample rate: 10 kHz
"""

import h5py, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir   = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
scratch   = r"d:\Bioview\My_RF_work_v1\scratch"

FS     = 10000
F0     = 100.714               # Ultrasound DDC centre frequency (Hz)
SCALE  = 333333.3 / (4*np.pi) # rad → µm (900 MHz carrier)

# ── Stable analysis windows per recording ──────────────────────
table_files = [
    ("ultra_rftable1.h5",  6,  22),   # long table recording
    ("ultra_rftable2.h5",  2,   8),
    ("ultra_rftable3.h5",  2,   8),
    ("ultra_rftable4.h5",  2,   8),
]
body_files = [
    ("ultra_rfbody01.h5", 29,  43),   # Body 1 — ultrasound active segment
    ("ultra_rfbody1.h5",   2,   8),   # Body 2
    ("ultra_rfbody2.h5",   2,   8),   # Body 3
    ("ultra_rfbody3.h5",   2,   8),   # Body 4
]
pair_labels = ["Pair 1", "Pair 2", "Pair 3", "Pair 4"]

# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════
def load_iq(filepath, t0, t1):
    with h5py.File(filepath, 'r') as f:
        d = f['data'][:]
    iq = d[0] + 1j * d[1]
    return iq[int(t0*FS): int(t1*FS)]

def lp_sos(sig, fc, fs=FS, order=2):
    sos = butter(order, fc / (0.5*fs), btype='low', output='sos')
    return sosfiltfilt(sos, sig)

def best_ddc_sign(iq):
    """Try both ±F0 DDC, return the sign that gives higher baseband magnitude."""
    N  = len(iq)
    t  = np.arange(N) / FS
    m1 = np.mean(np.abs(lp_sos(iq * np.exp( 1j*2*np.pi*F0*t), 15.0)))
    m2 = np.mean(np.abs(lp_sos(iq * np.exp(-1j*2*np.pi*F0*t), 15.0)))
    return +1.0 if m1 > m2 else -1.0

def demodulate(iq, sign=None, lp_hz=2.5, poly=4):
    """DDC at sign×F0 → LP filter → unwrap phase → poly-detrend → µm."""
    N  = len(iq)
    t  = np.arange(N) / FS
    if sign is None:
        sign = best_ddc_sign(iq)
    bb = lp_sos(iq * np.exp(-1j * sign * 2*np.pi*F0*t), lp_hz)
    dp = np.angle(bb[1:] * np.conj(bb[:-1]))
    ph = np.cumsum(np.insert(dp, 0, 0.0))
    coef = np.polyfit(t, ph, poly)
    disp = (ph - np.polyval(coef, t)) * SCALE
    return t, disp

def physio_band_power(disp, fs=FS):
    """Welch PSD power in the cardiac band 0.7–2.5 Hz (µm²/Hz)."""
    f, Pxx = welch(disp, fs=fs, nperseg=min(4096, len(disp)//2),
                   noverlap=None, window='hann')
    mask = (f >= 0.7) & (f <= 2.5)
    return f, Pxx, np.trapz(Pxx[mask], f[mask])

def noise_floor_power(disp, fs=FS):
    """Welch PSD power in the noise band 0.05–0.5 Hz (slow drift/noise)."""
    f, Pxx = welch(disp, fs=fs, nperseg=min(4096, len(disp)//2),
                   noverlap=None, window='hann')
    mask = (f >= 0.05) & (f <= 0.5)
    return np.trapz(Pxx[mask], f[mask])


# ══════════════════════════════════════════════════════════════
# Pre-compute all demodulated signals
# ══════════════════════════════════════════════════════════════
print("Demodulating all recordings …")
body_signals  = []   # list of (t, disp) for each pair
table_signals = []

for (tf, tt0, tt1), (bf, bt0, bt1) in zip(table_files, body_files):
    iq_t = load_iq(os.path.join(ultra_dir, tf), tt0, tt1)
    iq_b = load_iq(os.path.join(ultra_dir, bf), bt0, bt1)
    t_t, d_t = demodulate(iq_t)
    t_b, d_b = demodulate(iq_b)
    table_signals.append((t_t, d_t))
    body_signals.append((t_b, d_b))
    print(f"  ✓  {tf} | {bf}")

# ── Pre-compute Panel B (Physio-band integrated power per pair) ──
body_physio   = [physio_band_power(d)[2] for (_,d) in body_signals]
table_physio  = [physio_band_power(d)[2] for (_,d) in table_signals]

# ── Pre-compute Panel D (SNR = physio / noise-floor) ──────────
def snr_db(disp):
    pb = physio_band_power(disp)[2]
    pn = noise_floor_power(disp)
    return 10 * np.log10(pb / pn) if pn > 0 else 0.0

body_snr  = [snr_db(d) for (_,d) in body_signals]
table_snr = [snr_db(d) for (_,d) in table_signals]

# ── Best pair for Panel C waveform — use Pair 2 (Body 2 / Table 2) ─
# Crop to 3.5–7.5 s within the recording window (offset from t0=2s)
def crop_window(t, d, t_global_start, t_crop_start=3.5, t_crop_end=7.5):
    """Crop to a sub-window. t is relative (0-based); convert via t_global_start."""
    t_abs = t + t_global_start          # absolute time in original file
    # use the indices directly — t is already offset from t0
    t_offset = t_crop_start - 0.0      # within 0-based window
    mask = (t >= t_offset) & (t <= (t_crop_end - 0))
    return t[mask], d[mask]

t_b2, d_b2 = body_signals[1]
t_t2, d_t2 = table_signals[1]
# Crop the middle 4 seconds (t=1.5 to t=5.5 within 0-based 6-s window)
mask_b = (t_b2 >= 1.5) & (t_b2 <= 5.5)
mask_t = (t_t2 >= 1.5) & (t_t2 <= 5.5)
t_wave_b = t_b2[mask_b] - 1.5
t_wave_t = t_t2[mask_t] - 1.5
d_wave_b = d_b2[mask_b]
d_wave_t = d_t2[mask_t]

# Welch PSD for Panel A inset (Body 2 vs Table 2 — physio band zoom)
f_pb, Pxx_pb, _ = physio_band_power(d_b2)
f_pt, Pxx_pt, _ = physio_band_power(d_t2)

# Raw spectrum for Panel A — load IQ directly for the spectral comparison
iq_body2  = load_iq(os.path.join(ultra_dir, "ultra_rfbody1.h5"),  2, 8)
iq_table2 = load_iq(os.path.join(ultra_dir, "ultra_rftable2.h5"), 2, 8)


# ══════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# ══════════════════════════════════════════════════════════════
dark_bg    = '#0e1117'
panel_bg   = '#1a1d27'
body_col   = '#00e5ff'   # cyan  — body
table_col  = '#ff4d6d'   # crimson — table
txt_col    = '#e8e8e8'
grid_col   = '#2e3144'
accent     = '#f5c518'   # yellow for annotation

fig = plt.figure(figsize=(18, 14), facecolor=dark_bg)
fig.patch.set_facecolor(dark_bg)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.36,
                        left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_spec = fig.add_subplot(gs[0, 0])
ax_psd  = fig.add_subplot(gs[0, 1])
ax_wave = fig.add_subplot(gs[1, 0])
ax_snr  = fig.add_subplot(gs[1, 1])

for ax in [ax_spec, ax_psd, ax_wave, ax_snr]:
    ax.set_facecolor(panel_bg)
    ax.tick_params(colors=txt_col, labelsize=10)
    ax.xaxis.label.set_color(txt_col)
    ax.yaxis.label.set_color(txt_col)
    ax.title.set_color(txt_col)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid_col)
    ax.grid(True, color=grid_col, linewidth=0.6, alpha=0.8)


# ══════════════════════════════════════════════════════════════
# PANEL A — RF Baseband Power Spectrum (−600 to +600 Hz)
# ══════════════════════════════════════════════════════════════
for iq, label, col in [
        (iq_table2, "Table 2 — Static Control",  table_col),
        (iq_body2,  "Body 2 — Human Wrist (RF+US)", body_col)]:
    N  = len(iq)
    fr = np.fft.fftshift(np.fft.fftfreq(N, 1/FS))
    pw = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq)))**2 + 1e-10)
    m  = (fr >= -620) & (fr <= 620)
    ax_spec.plot(fr[m], pw[m], linewidth=1.1, label=label, color=col, alpha=0.88)

# Mark harmonics
for h in range(-6, 7):
    if h == 0:
        continue
    ax_spec.axvline(h*F0, color=accent, linestyle='--',
                    alpha=0.25 if abs(h) > 1 else 0.75, linewidth=0.8)
ax_spec.axvline(F0,  color=accent, linestyle='--', alpha=0.85, linewidth=1.4,
                label=f"Ultrasound harmonics (n·f₀, f₀ = {F0:.2f} Hz)")

ax_spec.set_xlim(-620, 620)
ax_spec.set_xlabel("Baseband Frequency (Hz)", fontsize=11, labelpad=4)
ax_spec.set_ylabel("Power (dB)", fontsize=11)
ax_spec.set_title("Panel A  ·  RF Baseband Power Spectrum\n"
                  "Ultrasound creates discrete sideband carriers at n·f₀",
                  fontsize=11, weight='bold', pad=8)
ax_spec.legend(fontsize=9, facecolor=panel_bg, labelcolor=txt_col,
               framealpha=0.85, loc='upper right')


# ══════════════════════════════════════════════════════════════
# PANEL B — Physiological-Band PSD (0.5–3 Hz) comparison
#           Welch PSD of DEMODULATED PHASE across all 4 pairs
# ══════════════════════════════════════════════════════════════
# Use Pair 2 for the spectral overlay and bar-inset for all pairs
ax_psd.semilogy(f_pt[f_pt<=3.0], Pxx_pt[f_pt<=3.0],
                color=table_col, linewidth=1.6, label="Table 2 — Static Control", alpha=0.85)
ax_psd.semilogy(f_pb[f_pb<=3.0], Pxx_pb[f_pb<=3.0],
                color=body_col,  linewidth=2.0, label="Body 2 — Human Wrist", alpha=0.95)

ax_psd.axvspan(0.7, 2.5, alpha=0.10, color=body_col, zorder=0,
               label="Cardiac band (0.7–2.5 Hz)")

# Mark expected HR range
ax_psd.axvline(1.0, color=accent, linestyle=':', linewidth=1.2, alpha=0.7)
ax_psd.axvline(2.0, color=accent, linestyle=':', linewidth=1.2, alpha=0.7)
ax_psd.text(1.0, ax_psd.get_ylim()[0] if ax_psd.get_ylim()[0]>0 else 1e-5,
            " 60 bpm", color=accent, fontsize=8, va='bottom')
ax_psd.text(2.0, ax_psd.get_ylim()[0] if ax_psd.get_ylim()[0]>0 else 1e-5,
            " 120 bpm", color=accent, fontsize=8, va='bottom')

ax_psd.set_xlim(0, 3.0)
ax_psd.set_xlabel("Frequency (Hz)", fontsize=11, labelpad=4)
ax_psd.set_ylabel("PSD (µm²/Hz)  [log scale]", fontsize=11)
ax_psd.set_title("Panel B  ·  Demodulated-Phase PSD (0–3 Hz)\n"
                 "Body shows elevated spectral power in cardiac band",
                 fontsize=11, weight='bold', pad=8)
ax_psd.legend(fontsize=9, facecolor=panel_bg, labelcolor=txt_col,
              framealpha=0.85, loc='upper right')

# Annotate integrated cardiac-band power for each pair
summary_lines = []
for i, (lbl, bp, tp) in enumerate(zip(pair_labels, body_physio, table_physio)):
    ratio = bp / tp if tp > 0 else float('inf')
    summary_lines.append(f"{lbl}: Body {bp:.1f}  /  Table {tp:.1f}  µm²  (×{ratio:.1f})")

summary_text = "Cardiac-band integrated power (µm²):\n" + "\n".join(summary_lines)
ax_psd.text(0.02, 0.97, summary_text, transform=ax_psd.transAxes,
            fontsize=8, color=txt_col, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc=dark_bg, ec=grid_col, alpha=0.85))


# ══════════════════════════════════════════════════════════════
# PANEL C — Demodulated Waveform: Body 2 vs Table 2
# ══════════════════════════════════════════════════════════════
ax_wave.plot(t_wave_t, d_wave_t,
             color=table_col, linewidth=1.5,
             label="Table 2 — Static Control (noise floor)", alpha=0.80)
ax_wave.plot(t_wave_b, d_wave_b,
             color=body_col,  linewidth=2.2,
             label="Body 2 — Cardiac micro-displacement (RF+US)", alpha=0.95)

# Annotate rhythmic peaks automatically
from scipy.signal import find_peaks
peaks, _ = find_peaks(d_wave_b, prominence=np.std(d_wave_b)*0.6, distance=int(0.4*FS))
# peaks indices are in the sub-array — must scale back to sub-array time
# t_wave_b has shape (N,) at FS=10000 samples/s; convert peak idx → time
t_peaks = t_wave_b[peaks] if len(peaks) > 0 else []
ax_wave.scatter(t_peaks, d_wave_b[peaks], color=accent, zorder=5,
                s=60, marker='v', label=f"Detected pulse peaks (n={len(peaks)})")

ax_wave.set_xlim(t_wave_b[0], t_wave_b[-1])
ax_wave.set_xlabel("Time (s)", fontsize=11, labelpad=4)
ax_wave.set_ylabel("RF Phase Displacement (µm)", fontsize=11)
ax_wave.set_title("Panel C  ·  Demodulated Waveform — Body 2 vs Table 2\n"
                  "2.5 Hz lowpass + 4th-order poly detrend (1f₀ DDC)",
                  fontsize=11, weight='bold', pad=8)
ax_wave.legend(fontsize=9, facecolor=panel_bg, labelcolor=txt_col,
               framealpha=0.85, loc='upper right')

# Annotate RMS values
rms_b = np.std(d_wave_b)
rms_t = np.std(d_wave_t)
ax_wave.text(0.02, 0.95,
             f"Body RMS: {rms_b:.2f} µm\nTable RMS: {rms_t:.2f} µm\nRatio: ×{rms_b/rms_t:.1f}",
             transform=ax_wave.transAxes, fontsize=9, color=txt_col, va='top',
             bbox=dict(boxstyle='round,pad=0.4', fc=dark_bg, ec=grid_col, alpha=0.85))


# ══════════════════════════════════════════════════════════════
# PANEL D — Cardiac-Band SNR: Body vs Table across 4 pairs
#           SNR = 10·log₁₀(cardiac-band power / noise-floor power)
# ══════════════════════════════════════════════════════════════
x    = np.arange(4)
w    = 0.32
bars_t = ax_snr.bar(x - w/2, table_snr, width=w, color=table_col,
                    label="Table (Static Control)", alpha=0.9,
                    edgecolor='white', linewidth=0.5)
bars_b = ax_snr.bar(x + w/2, body_snr,  width=w, color=body_col,
                    label="Body (Human Subject)",   alpha=0.9,
                    edgecolor='white', linewidth=0.5)

for bar, val in zip(bars_t, table_snr):
    ax_snr.text(bar.get_x()+bar.get_width()/2, max(val+0.3, 0.3),
                f"{val:.1f}", ha='center', va='bottom', fontsize=9,
                color=txt_col, weight='bold')
for bar, val in zip(bars_b, body_snr):
    ax_snr.text(bar.get_x()+bar.get_width()/2, max(val+0.3, 0.3),
                f"{val:.1f}", ha='center', va='bottom', fontsize=9,
                color=txt_col, weight='bold')

# Annotate delta
for i, (bs, ts) in enumerate(zip(body_snr, table_snr)):
    delta = bs - ts
    ypos  = max(bs, ts) + 1.5
    sign  = "+" if delta >= 0 else ""
    ax_snr.text(i, ypos, f"Δ{sign}{delta:.1f} dB",
                ha='center', va='bottom', fontsize=9,
                color=accent, weight='bold')

ax_snr.set_xticks(x)
ax_snr.set_xticklabels(pair_labels, color=txt_col, fontsize=10)
ax_snr.set_ylabel("Cardiac-Band SNR (dB)", fontsize=11)
ax_snr.set_xlabel("Recording Pair", fontsize=11, labelpad=4)
ax_snr.set_title("Panel D  ·  Cardiac-Band SNR per Recording Pair\n"
                 "SNR = 10·log₁₀(power 0.7–2.5 Hz  /  power 0.05–0.5 Hz)",
                 fontsize=11, weight='bold', pad=8)
ax_snr.legend(fontsize=9, facecolor=panel_bg, labelcolor=txt_col,
              framealpha=0.85, loc='upper right')
ax_snr.axhline(0, color='white', linewidth=0.6, linestyle='--', alpha=0.5)


# ══════════════════════════════════════════════════════════════
# Super-title & Save
# ══════════════════════════════════════════════════════════════
fig.suptitle(
    "Acousto-RF Sensing  ·  4-Point Scientific Validation of Ultrasound Effect on 900 MHz RF\n"
    "RF Carrier: 900 MHz  ·  Ultrasound DDC: f₀ ≈ 100.71 Hz  ·  n = 4 pairs (Body vs Static Table)",
    fontsize=13, weight='bold', color=txt_col, y=0.975
)

out_path = os.path.join(out_dir, "acousto_rf_4point_validation_v2.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print(f"\nSaved 300 DPI figure: {out_path}")

import shutil
shutil.copy(out_path, os.path.join(scratch, "acousto_rf_4point_validation_v2.png"))
print("Copied to scratch/")

# ── Print summary table ──────────────────────────────────────
print("\n─── Cardiac-Band Power Summary (µm²) ───────────────────────")
print(f"{'Pair':<10}{'Body Power':>14}{'Table Power':>14}{'Ratio':>10}{'Body SNR':>12}{'Table SNR':>12}")
for i, lbl in enumerate(pair_labels):
    bp = body_physio[i];  tp = table_physio[i]
    ratio = bp/tp if tp>0 else float('inf')
    print(f"{lbl:<10}{bp:>14.2f}{tp:>14.2f}{ratio:>10.2f}x{body_snr[i]:>11.1f} dB{table_snr[i]:>10.1f} dB")
