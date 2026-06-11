"""
RF Radar Velocity Derivation — Publication Analysis (Corrected)
================================================================
Uses np.unwrap() on the raw IQ angle to get a physically continuous
phase signal. After linear detrending, the phase stays in a small
radian range, giving realistic displacement (mm) and velocity (mm/s).

Physics:
    phi(t) = (4*pi/lambda) * d(t)     [rad]
    d(t)   = phi(t) * SCALE            [mm],  SCALE = lambda/(4*pi)
    v(t)   = d'(t)  = phi'(t) * SCALE [mm/s]

    USRP B210  fc = 0.9 GHz:
        lambda = c/fc = 333.10 mm
        SCALE  = 333.10 / (4*pi) = 26.51 mm/rad
"""

import h5py
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ─────────────────────────────────────────────────────────
RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rf_velocity_derivation_analysis.png'

FS_RF     = 10_000
FC_HZ     = 0.9e9
C_LIGHT   = 299_792_458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000          # 333.10 mm
SCALE     = LAMBDA_MM / (4 * np.pi)            # 26.51 mm/rad

KORO_ON   = 24.0
KORO_OFF  = 41.5

# ── HELPERS ────────────────────────────────────────────────────────
def iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sin_phi = p3 / np.sqrt(p1*p2 + 1e-20)
    cos_phi = np.sqrt(max(1 - sin_phi**2, 1e-10))
    alpha   = np.sqrt(p2 / (p1 + 1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sin_phi, -1, 1)))) < 90:
        qc2 = (qc - sin_phi * ic) / (alpha * cos_phi + 1e-15)
    else:
        qc2 = qc
    return ic + 1j * qc2

# ── LOAD ───────────────────────────────────────────────────────────
print("Loading RF data ...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = data[0, :], data[1, :]
N = len(i_raw)
t = np.arange(N) / FS_RF
print(f"  {N:,} samples | {t[-1]:.1f} s | fc={FC_HZ/1e9} GHz | lambda={LAMBDA_MM:.2f} mm | SCALE={SCALE:.4f} mm/rad")

# ── PHASE DERIVATION (np.unwrap — physically correct) ──────────────
iq_cond   = iq_condition(-i_raw + 1j * q_raw)

# Wrapped angle (before unwrap) — for display only
phi_wrapped = np.angle(iq_cond)

# Unwrapped continuous phase
phi_unwrapped = np.unwrap(phi_wrapped)

# Remove linear carrier drift → small-radian residual
phi = signal.detrend(phi_unwrapped, type='linear')

print(f"  Unwrapped phase range : {phi.min():.2f} to {phi.max():.2f} rad")

# ── DISPLACEMENT & VELOCITY ────────────────────────────────────────
# Heartbeat displacement (0.4-3 Hz)
sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
dh    = sosfiltfilt(sos_h, phi) * SCALE          # mm

# Korotkoff velocity (10-200 Hz band-pass then differentiate)
sos_k = butter(4, [10, 200], btype='band', fs=FS_RF, output='sos')
pk    = sosfiltfilt(sos_k, phi)
vk    = np.append(np.diff(pk) * FS_RF, 0) * SCALE   # mm/s

# ── STATISTICS ────────────────────────────────────────────────────
mask_k    = (t >= KORO_ON)  & (t <= KORO_OFF)
mask_base = (t >= 5.0)      & (t <= 15.0)

vk_k    = vk[mask_k]
vk_base = vk[mask_base]
dh_k    = dh[mask_k]

rms_k    = np.sqrt(np.mean(vk_k**2))
rms_base = np.sqrt(np.mean(vk_base**2))
snr      = rms_k / (rms_base + 1e-20)

print(f"\nVelocity statistics (10-200 Hz band):")
print(f"  Peak |vk|  (Koro window) : {np.max(np.abs(vk_k)):.2f} mm/s")
print(f"  RMS  |vk|  (Koro window) : {rms_k:.2f} mm/s")
print(f"  RMS  |vk|  (baseline)    : {rms_base:.2f} mm/s")
print(f"  SNR  Koro/Base           : {snr:.2f} x")
print(f"  Displacement peak (HR)   : {np.max(np.abs(dh_k)):.4f} mm")

# ── PSD ───────────────────────────────────────────────────────────
f_psd, p_k    = welch(vk_k,    fs=FS_RF, nperseg=min(len(vk_k),    int(FS_RF*2)))
_,     p_base = welch(vk_base, fs=FS_RF, nperseg=min(len(vk_base), int(FS_RF*2)))

# ── SPECTROGRAM ───────────────────────────────────────────────────
ds_fs = 600
vk_ds = signal.resample_poly(vk, up=ds_fs, down=FS_RF)
t_ds  = np.arange(len(vk_ds)) / ds_fs
nps   = min(len(vk_ds)//4, int(ds_fs * 0.15))
f_sg, t_sg, Sxx = signal.spectrogram(vk_ds, fs=ds_fs, window='hann',
                                      nperseg=nps, noverlap=nps*7//8, nfft=1024)
P_db = 10 * np.log10(np.sqrt(np.abs(Sxx)) + 1e-20)

# ── PLOT ──────────────────────────────────────────────────────────
print("\nGenerating figure ...")
ds = max(1, N // 50000)
t_p   = t[::ds];   i_p = i_raw[::ds];  q_p = q_raw[::ds]
phi_w = phi_wrapped[::ds];  phi_u = phi_unwrapped[::ds];  phi_p = phi[::ds]
dh_p  = dh[::ds];  vk_p = vk[::ds]

fig = plt.figure(figsize=(22, 28), dpi=300)
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.30,
                         left=0.07, right=0.96, top=0.94, bottom=0.04)

GOLD=  '#FFD700'; CYAN='#00FFFF'; LIME='#39FF14'
CORAL= '#FF6B6B'; PURP='#BD93F9'; ORAN='#FFB347'
WHT=   '#F8F8F2'; BGX= '#161b22'

def sax(ax, title, xlabel='Time (s)', ylabel=''):
    ax.set_facecolor(BGX)
    ax.set_title(title, color=WHT, fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, color=WHT, fontsize=8.5)
    ax.set_ylabel(ylabel, color=WHT, fontsize=8.5)
    ax.tick_params(colors=WHT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.6, alpha=0.8)

kspan = dict(color=GOLD, alpha=0.13)
TBOX  = dict(boxstyle='round', facecolor='#21262d', alpha=0.88)

# Panel 1 — Raw I and Q
ax = fig.add_subplot(gs[0, 0]); sax(ax, 'Panel 1 — Raw IQ Baseband Signals', ylabel='Amplitude (a.u.)')
ax.plot(t_p, i_p, color=CYAN,  lw=0.35, alpha=0.8, label='I channel')
ax.plot(t_p, q_p, color=CORAL, lw=0.35, alpha=0.8, label='Q channel')
ax.axvspan(KORO_ON, KORO_OFF, **kspan, label=f'Korotkoff [{KORO_ON}-{KORO_OFF}s]')
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 2 — Wrapped vs Unwrapped phase (replaces IQ constellation)
ax = fig.add_subplot(gs[0, 1]); sax(ax, 'Panel 2 — Wrapped Angle vs np.unwrap() Phase', ylabel='Phase (rad)')
ax.plot(t_p, phi_w, color=CORAL, lw=0.3, alpha=0.7, label='Wrapped angle(IQ)  ±π')
ax.plot(t_p, phi_u, color=CYAN,  lw=0.5, alpha=0.85, label='np.unwrap(angle(IQ))  continuous')
ax.axvspan(KORO_ON, KORO_OFF, **kspan)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)
ax.text(0.01, 0.97, 'np.unwrap() gives continuous\nphase for correct d(t) & v(t)',
        transform=ax.transAxes, color=LIME, fontsize=8, va='top',
        family='monospace', bbox=TBOX)

# Panel 3 — Detrended phase (residual, small radians)
ax = fig.add_subplot(gs[1, 0]); sax(ax, 'Panel 3 — Detrended Phase phi(t) [rad]  (tissue motion only)', ylabel='Phase (rad)')
ax.plot(t_p, phi_p, color=PURP, lw=0.5, alpha=0.85)
ax.axvspan(KORO_ON, KORO_OFF, **kspan)
ax.axhline(0, color='#555', lw=0.8, ls='--')
ph_rng = phi.max() - phi.min()
ax.text(0.01, 0.97,
        f'phi = unwrap(angle(IQ)) linearly detrended\n'
        f'Range: {phi.min():.3f} to {phi.max():.3f} rad  ({ph_rng:.3f} rad)\n'
        f'=> displacement range: {ph_rng*SCALE:.3f} mm',
        transform=ax.transAxes, color=ORAN, fontsize=8, va='top',
        family='monospace', bbox=TBOX)

# Panel 4 — Displacement d(t) 0.4-3 Hz
ax = fig.add_subplot(gs[1, 1]); sax(ax, 'Panel 4 — Tissue Displacement d(t) [mm]  (0.4–3 Hz HR band)', ylabel='Displacement (mm)')
ax.plot(t_p, dh_p, color=CYAN, lw=0.6, alpha=0.9, label=f'd(t) = phi_h * SCALE  (SCALE={SCALE:.2f} mm/rad)')
ax.axvspan(KORO_ON, KORO_OFF, **kspan)
ax.fill_between(t_p, dh_p, alpha=0.15, color=CYAN)
ax.text(0.01, 0.97,
        f'Peak displacement (Koro): {np.max(np.abs(dh_k)):.4f} mm\n'
        f'Typical HR chest motion: 0.01–1 mm (CW radar)',
        transform=ax.transAxes, color=LIME, fontsize=8, va='top',
        family='monospace', bbox=TBOX)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 5 — Korotkoff velocity full recording
ax = fig.add_subplot(gs[2, 0]); sax(ax, 'Panel 5 — Korotkoff Velocity vk(t) [mm/s]  (10–200 Hz)', ylabel='Velocity (mm/s)')
ax.plot(t_p, vk_p, color=LIME, lw=0.45, alpha=0.85, label='vk = d(BPF(phi))/dt * SCALE')
ax.axvspan(KORO_ON, KORO_OFF, color=GOLD, alpha=0.18, label=f'Korotkoff window ({KORO_ON}–{KORO_OFF}s)')
ax.text(0.01, 0.97,
        f'vk[n] = diff(BPF(phi, 10-200 Hz))[n] x fs x SCALE\n'
        f'      = diff(pk)[n] x {FS_RF} x {SCALE:.2f}  mm/s',
        transform=ax.transAxes, color=ORAN, fontsize=8, va='top',
        family='monospace', bbox=TBOX)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 6 — Zoomed Korotkoff velocity window
ax = fig.add_subplot(gs[2, 1]); sax(ax, 'Panel 6 — Zoomed Korotkoff Window vk(t) [mm/s]', ylabel='Velocity (mm/s)')
t_kz = t[mask_k][::max(1, mask_k.sum()//5000)]
vk_kz = vk[mask_k][::max(1, mask_k.sum()//5000)]
ax.plot(t_kz, vk_kz, color=LIME, lw=0.7, alpha=0.9)
ax.axhline(rms_k,  color=GOLD, ls='--', lw=1.5, label=f'RMS = {rms_k:.2f} mm/s')
ax.axhline(-rms_k, color=GOLD, ls='--', lw=1.5)
ax.fill_between(t_kz, vk_kz, alpha=0.18, color=LIME)
ax.text(0.01, 0.97,
        f'Peak  |vk|: {np.max(np.abs(vk_k)):.2f} mm/s\n'
        f'RMS   |vk|: {rms_k:.2f} mm/s  (Koro window)\n'
        f'RMS   |vk|: {rms_base:.2f} mm/s  (baseline)\n'
        f'SNR (Koro/Base): {snr:.2f} x',
        transform=ax.transAxes, color=LIME, fontsize=8.5, va='top',
        family='monospace', bbox=TBOX)
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 7 — PSD comparison
ax = fig.add_subplot(gs[3, 0]); sax(ax, 'Panel 7 — PSD: Korotkoff Window vs Baseline', xlabel='Frequency (Hz)', ylabel='Power (dB/Hz)')
mask_f = f_psd <= 300
ax.plot(f_psd[mask_f], 10*np.log10(p_k[mask_f]    + 1e-20), color=LIME,  lw=1.5, label=f'Koro window ({KORO_ON}–{KORO_OFF}s)')
ax.plot(f_psd[mask_f], 10*np.log10(p_base[mask_f] + 1e-20), color=PURP,  lw=1.2, ls='--', label='Baseline (5–15s)')
ax.axvspan(10, 200, color=GOLD, alpha=0.08, label='Korotkoff band (10–200 Hz)')
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 8 — Velocity histogram
ax = fig.add_subplot(gs[3, 1]); sax(ax, 'Panel 8 — Velocity Amplitude Distribution', xlabel='Velocity (mm/s)', ylabel='Probability Density')
clip = np.percentile(np.abs(vk_k), 99)
bins = np.linspace(-clip*1.1, clip*1.1, 100)
ax.hist(vk_base, bins=bins, density=True, color=PURP, alpha=0.55, label=f'Baseline  RMS={rms_base:.2f} mm/s')
ax.hist(vk_k,    bins=bins, density=True, color=LIME, alpha=0.55, label=f'Korotkoff RMS={rms_k:.2f} mm/s')
ax.axvline(0, color='white', lw=0.8, ls='--')
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT)

# Panel 9 — Spectrogram
ax = fig.add_subplot(gs[4, 0]); sax(ax, 'Panel 9 — Korotkoff Velocity Spectrogram (10–200 Hz)', ylabel='Frequency (Hz)')
fm = (f_sg >= 10) & (f_sg <= 200)
va, vb = np.percentile(P_db[fm], [15, 99])
im = ax.imshow(P_db[fm], extent=[t_sg[0], t_sg[-1], f_sg[fm][0], f_sg[fm][-1]],
               aspect='auto', origin='lower', cmap='magma', vmin=va, vmax=vb, interpolation='bilinear')
ax.axvline(KORO_ON,  color=GOLD, ls='--', lw=2, label=f'Koro onset  {KORO_ON}s')
ax.axvline(KORO_OFF, color=GOLD, ls='--', lw=2, label=f'Koro offset {KORO_OFF}s')
ax.legend(fontsize=7, facecolor='#21262d', labelcolor=WHT, loc='upper right')
cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label('dB', color=WHT, fontsize=8); cb.ax.tick_params(colors=WHT, labelsize=7)

# Panel 10 — Derivation equation card
ax = fig.add_subplot(gs[4, 1]); ax.set_facecolor(BGX); ax.axis('off')
for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
eq = [
    "RF VELOCITY DERIVATION  (np.unwrap method)",
    "=" * 44,
    f"  fc = {FC_HZ/1e9:.2f} GHz | fs = {FS_RF} Hz",
    f"  lambda = c/fc = {LAMBDA_MM:.2f} mm",
    f"  SCALE  = lambda/(4*pi) = {SCALE:.4f} mm/rad",
    "",
    "STEP 1: IQ conditioning",
    "  Remove DC, correct quadrature imbalance",
    "",
    "STEP 2: Phase extraction (np.unwrap)",
    "  phi_wrapped = angle(IQ)         [±pi rad]",
    "  phi_unwrapped = np.unwrap(phi_wrapped)",
    "  phi = linear_detrend(phi_unwrapped)",
    f"  phi range: {phi.min():.3f} to {phi.max():.3f} rad",
    "",
    "STEP 3: Displacement",
    "  d(t) = phi(t) * SCALE            [mm]",
    f"  HR band 0.4-3 Hz peak: {np.max(np.abs(dh_k)):.4f} mm",
    "",
    "STEP 4: Korotkoff velocity (10-200 Hz)",
    "  pk   = BPF(phi, 10-200 Hz)",
    "  vk[n]= diff(pk)[n] * fs * SCALE  [mm/s]",
    "",
    "RESULTS:",
    f"  Peak  |vk| (Koro win): {np.max(np.abs(vk_k)):.2f} mm/s",
    f"  RMS   |vk| (Koro win): {rms_k:.2f} mm/s",
    f"  RMS   |vk| (baseline): {rms_base:.2f} mm/s",
    f"  SNR   Koro/Base:       {snr:.2f} x",
    "=" * 44,
]
ax.text(0.03, 0.97, '\n'.join(eq), transform=ax.transAxes, fontsize=8.5,
        family='monospace', color=WHT, va='top',
        bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.95, lw=0))
ax.set_title('Panel 10 — Derivation Equation Card', color=WHT, fontsize=10.5, fontweight='bold', pad=6)

fig.suptitle('RF Radar Micro-Velocity Derivation Analysis  (np.unwrap Phase Method)\n'
             'USRP B210 @ 0.9 GHz  |  Radiomyography (RMG)  |  Korotkoff BP Sensing',
             color=WHT, fontsize=13, fontweight='bold', y=0.972)

plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nFigure saved -> {OUTPUT_IMG}")
