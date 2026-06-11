"""
RF Confirmation Dashboard: Magnitude vs Phase -- 3x2 Publication Layout
Generates validation dashboards for both Subject 1 (Prof. Kan, Rec 06) and Subject 2 (Rajveer, Rec 04)
at 300 DPI.
"""

import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, welch, hilbert, find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

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

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'

# ── HELPERS ───────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def notch(x, f0, fs, Q=30):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

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

def run_validation(sub_select):
    if sub_select == 1:
        sub_name = "Subject 1 (Prof. Kan)"
        rec_name = "Rec 06"
        rf_path  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
        wav_path = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')
        out_path = os.path.join(BASE, 'rf_confirm_mag_phase_validation_Sub1_Rec6_Final.png')
        k_on     = 27.530
        k_off    = 43.330
        defl     = 18.0
        t_max    = 52.0
        lag      = 1.7083
        notches  = [100.71, 201.43, 302.14, 402.86]
    else:
        sub_name = "Subject 2 (Rajveer)"
        rec_name = "Rec 04"
        rf_path  = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
        wav_path = os.path.join(BASE, 'Sub_2_Rajveer', 'sthethoscope_rec04.wav')
        out_path = os.path.join(BASE, 'rf_confirm_mag_phase_validation_Sub2_Rec4_Final.png')
        k_on     = 27.380
        k_off    = 42.000
        defl     = 18.6
        t_max    = 51.0
        lag      = 2.6042
        notches  = [50.0, 64.0, 100.6, 201.2]

    fs_rf     = 10_000
    dec       = 10
    fs        = fs_rf // dec
    fc        = 0.9e9
    lambda_mm = (299_792_458.0 / fc) * 1000
    scale     = lambda_mm / (4.0 * np.pi)
    koro_dur  = k_off - k_on
    zoom_l    = max(0.0, k_on - 7.0)
    zoom_r    = min(t_max, k_off + 1.5)

    cm   = '#1A6FC4'   # blue   – Magnitude
    cp   = '#C0392B'   # red    – Phase
    ce   = '#1A1A2E'   # dark   – Envelope
    ck   = '#F39C12'   # amber  – Korotkoff lines
    ckfill = '#FEF9EC' # light amber fill
    bg   = '#FFFFFF'
    ctxt = '#1C1C1C'

    def add_koro(ax, full=True):
        ax.axvspan(k_on, k_off, color=ckfill, alpha=0.85, zorder=0)
        ax.axvline(k_on,  color=ck, lw=1.4, ls='--', zorder=2)
        ax.axvline(k_off, color=ck, lw=1.4, ls='--', zorder=2)
        if full:
            ax.axvline(defl, color='#888888', lw=0.9, ls=':', zorder=1)

    # ── LOAD RF ───────────────────────────────────────────────────────────────
    print(f"Loading RF data for {sub_name}...")
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0, :], rf_data[1, :]

    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c  = i_raw - xc, q_raw - yc
    phi_raw = robust_phase(i_c, q_c)

    sos_lp  = butter(4, 300.0, btype='low', fs=fs_rf, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))

    mag_ds = decimate(mag_raw, dec, ftype='fir')
    phi_ds = decimate(phi_raw, dec, ftype='fir')
    t      = np.arange(len(mag_ds)) / fs

    # ── LOAD STETHOSCOPE ──────────────────────────────────────────────────────
    print(f"Loading Stethoscope audio for {sub_name}...")
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)

    audio_filt = bpf(audio, 50.0, 1000.0, fs_a)
    steth_koro = bpf(np.abs(hilbert(audio_filt)), 20.0, min(200.0, (fs_a/2)-1), fs_a)
    steth_tkeo_env = smooth_energy(calc_tkeo(steth_koro), 1.5, fs_a)

    # Align Stethoscope time axis
    t_a = (np.arange(len(steth_tkeo_env)) / fs_a) + lag
    
    # Define clean deflation window to exclude valve and dump transients
    t_start_clean = defl + 3.0
    t_end_clean   = k_off + 1.2
    
    # Zero out stethoscope signal outside clean deflation window
    steth_tkeo_env[(t_a < t_start_clean) | (t_a > t_end_clean)] = 0.0
    
    mask_act_a = (t_a >= k_on) & (t_a <= k_off)
    # Stethoscope TKEO SNR baseline (pre-Korotkoff is stable and clean)
    mask_bas_a = (t_a >= 22.0) & (t_a <= k_on - 2.0)
    b_min_a = np.percentile(steth_tkeo_env[mask_bas_a], 5)
    e_shifted_a = np.maximum(steth_tkeo_env - b_min_a, 0)
    steth_tkeo_n = e_shifted_a / (np.max(e_shifted_a[mask_act_a]) + 1e-10)

    # Deep clean Phase and Magnitude before derivative to prevent noise explosion!
    phi_clean = phi_raw.copy()
    mag_clean = mag_raw.copy()
    for freq in notches:
        phi_clean = notch(phi_clean, freq, fs_rf, Q=30)
        mag_clean = notch(mag_clean, freq, fs_rf, Q=30)

    mag_vel_rf = np.append(np.diff(bpf(mag_clean, 30, 180, fs_rf))*fs_rf, 0.0)
    phi_vel_rf = np.append(np.diff(bpf(phi_clean, 30, 180, fs_rf))*fs_rf, 0.0)*scale

    # Zero out high-frequency RF velocity outside clean deflation window
    t_rf_full = np.arange(len(mag_vel_rf)) / fs_rf
    mag_vel_rf[(t_rf_full < t_start_clean) | (t_rf_full > t_end_clean)] = 0.0
    phi_vel_rf[(t_rf_full < t_start_clean) | (t_rf_full > t_end_clean)] = 0.0

    # ── TKEO Envelopes (SNR Comparison) ───────────────────────────────
    mag_tkeo_env_rf = smooth_energy(calc_tkeo(mag_vel_rf), 1.5, fs_rf)
    phi_tkeo_env_rf = smooth_energy(calc_tkeo(phi_vel_rf), 1.5, fs_rf)

    mag_tkeo_env = decimate(mag_tkeo_env_rf, dec, ftype='fir')
    phi_tkeo_env = decimate(phi_tkeo_env_rf, dec, ftype='fir')

    mask_act = (t >= k_on) & (t <= k_off)
    mask_bas = (t >= 22.0) & (t <= k_on - 2.0)

    def norm_envelope(env):
        b_min = np.percentile(env[mask_bas], 5)
        e_shifted = np.maximum(env - b_min, 0)
        return e_shifted / (np.max(e_shifted[mask_act]) + 1e-10)

    mag_tkeo_n = norm_envelope(mag_tkeo_env)
    phi_tkeo_n = norm_envelope(phi_tkeo_env)

    # ── Displacement 0.4–3 Hz ─────────────────────────────────────────
    mag_dc       = np.mean(mag_ds)
    mag_disp_au  = decimate(bpf(mag_raw, 0.4, 3.0, fs_rf), dec, ftype='fir')
    mag_disp     = (mag_disp_au / mag_dc) * scale
    
    # Zero out compliance pulses outside clean deflation window
    mag_disp[(t < t_start_clean) | (t > t_end_clean)] = 0.0
    mag_disp_env = env_smooth(mag_disp, 1.5, fs)
    mag_disp_env[(t < t_start_clean) | (t > t_end_clean)] = 0.0
    
    phi_disp     = decimate(bpf(phi_raw, 0.4, 3.0, fs_rf) * scale, dec, ftype='fir')
    phi_disp[(t < t_start_clean) | (t > t_end_clean)] = 0.0
    phi_disp_env = env_smooth(phi_disp, 1.5, fs)
    phi_disp_env[(t < t_start_clean) | (t > t_end_clean)] = 0.0

    # Compute PSD for Phase Velocity (Korotkoff vs Quiet Post-Korotkoff Baseline)
    mask_koro_rf = (np.arange(len(phi_vel_rf))/fs_rf >= k_on) & (np.arange(len(phi_vel_rf))/fs_rf <= k_off)
    mask_base_rf = (np.arange(len(phi_vel_rf))/fs_rf >= k_off + 1.5) & (np.arange(len(phi_vel_rf))/fs_rf <= t_max)
    f_psd, pxx_koro = welch(phi_vel_rf[mask_koro_rf], fs=fs_rf, nperseg=int(fs_rf*1.0))
    f_psd, pxx_base = welch(phi_vel_rf[mask_base_rf], fs=fs_rf, nperseg=int(fs_rf*1.0))
    mask_f = (f_psd >= 10) & (f_psd <= 80)
    f_psd = f_psd[mask_f]
    pxx_koro = 10 * np.log10(pxx_koro[mask_f] + 1e-20)
    pxx_base = 10 * np.log10(pxx_base[mask_f] + 1e-20)

    # Calculate heart rate (HR) from stable window (exactly 18.0s to end) of compliance pulses
    mask_hr = (t >= 18.0) & (t <= t[-1])
    t_stable = t[mask_hr]
    min_dist_hr = int(fs * 0.5)
    
    # Magnitude HR
    mag_stable = mag_disp[mask_hr]
    peaks_m, _ = find_peaks(mag_stable, distance=min_dist_hr, prominence=np.std(mag_stable)*0.4)
    hr_m = 60.0 / np.mean(np.diff(t_stable[peaks_m])) if len(peaks_m) > 1 else 0.0
    
    # Phase HR
    phi_stable = phi_disp[mask_hr]
    peaks_p, _ = find_peaks(phi_stable, distance=min_dist_hr, prominence=np.std(phi_stable)*0.4)
    hr_p = 60.0 / np.mean(np.diff(t_stable[peaks_p])) if len(peaks_p) > 1 else 0.0

    # ── BUILD FIGURE ──────────────────────────────────────────────────────────
    print(f"Building publication 3x2 dashboard for {sub_name}...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), dpi=300, facecolor=bg)
    fig.patch.set_facecolor(bg)

    def yzoom(sig, mask, pad=1.25):
        lo = np.min(sig[mask]) * pad if np.min(sig[mask]) < 0 else np.min(sig[mask]) / pad
        hi = np.max(np.abs(sig[mask])) * pad
        return lo, hi

    zm = (t >= zoom_l) & (t <= zoom_r)

    # PANEL A — Magnitude High-Frequency Vibration Waveform
    ax = axes[0, 0]
    ax.set_title('(A)  RF Magnitude — Korotkoff High-Frequency Waveform (30-180 Hz)')
    add_koro(ax)
    mag_vel_n = mag_vel_rf / (np.max(np.abs(mag_vel_rf)) + 1e-10)
    ax.plot(t_rf_full[::10], mag_vel_n[::10], color=cm, lw=0.45, alpha=0.6, label='Magnitude Korotkoff Signal')
    ax.set_xlim([15, t_max]); ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Amplitude (a.u.)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    # PANEL B — Phase High-Frequency Vibration Waveform
    ax = axes[0, 1]
    ax.set_title('(B)  RF Phase Velocity — Korotkoff High-Frequency Waveform (30-180 Hz)')
    add_koro(ax)
    phi_vel_n = phi_vel_rf / (np.max(np.abs(phi_vel_rf)) + 1e-10)
    ax.plot(t_rf_full[::10], phi_vel_n[::10], color=cp, lw=0.45, alpha=0.6, label='Phase Velocity Korotkoff Signal')
    ax.set_xlim([15, t_max]); ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Amplitude (a.u.)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    # PANEL C — Magnitude Displacement (ZOOMED)
    ax = axes[1, 0]
    ax.set_title(f'(C)  Magnitude — Arterial Compliance Pulse (HR = {hr_p:.1f} BPM)')
    add_koro(ax)
    ax.plot(t, mag_disp,     color=cm, lw=1.5, alpha=0.9, label='Compliance pulse mm (0.4-3 Hz)')
    yc_lo, yc_hi = yzoom(mag_disp, zm)
    # Give it vertical headroom for the legend
    ax.set_xlim([0, t_max]); ax.set_ylim([yc_lo, yc_hi * 1.35])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement (mm)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc='upper left', frameon=False)

    # PANEL D — Phase Displacement (ZOOMED)
    ax = axes[1, 1]
    ax.set_title(f'(D)  Phase — Physical Arterial Displacement (HR = {hr_p:.1f} BPM)')
    add_koro(ax)
    ax.plot(t, phi_disp,     color=cp, lw=1.5, alpha=0.9, label='Displacement mm (0.4-3 Hz)')
    yd_lo, yd_hi = yzoom(phi_disp, zm)
    # Give it vertical headroom for the legend
    ax.set_xlim([0, t_max]); ax.set_ylim([yd_lo, yd_hi * 1.35])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Displacement (mm)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc='upper left', frameon=False)

    # PANEL E — Phase Frequency Domain (PSD Zoom 10-80 Hz)
    ax = axes[2, 0]
    ax.set_title('(E)  Phase Velocity — Frequency Domain PSD (10-80 Hz)')
    ax.plot(f_psd, pxx_base, color='#555555', lw=1.5, ls='--', label='Quiet Baseline')
    ax.plot(f_psd, pxx_koro, color=cp, lw=2.0, alpha=0.9, label='Active Window (RF)')
    ax.fill_between(f_psd, pxx_base, pxx_koro, where=(pxx_koro > pxx_base), color=cp, alpha=0.15)
    ax.set_xlim([10, 80])
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power Spectral Density (dB)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    # PANEL F — Stethoscope Ground Truth High-Frequency Acoustic Waveform
    ax = axes[2, 1]
    ax.set_title('(F)  Ground Truth: Stethoscope High-Frequency Acoustic Waveform (50-1000 Hz)')
    add_koro(ax)
    audio_filt_clean = audio_filt.copy()
    t_a_full = (np.arange(len(audio_filt_clean)) / fs_a) + lag
    audio_filt_clean[(t_a_full < t_start_clean) | (t_a_full > t_end_clean)] = 0.0
    audio_n = audio_filt_clean / (np.max(np.abs(audio_filt_clean)) + 1e-10)
    ax.plot(t_a_full[::8], audio_n[::8], color='#2980B9', lw=0.35, alpha=0.55, label='Steth Acoustic Signal')
    ax.set_xlim([15, t_max]); ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Amplitude (a.u.)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    # ── SUPTITLE ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f'RF Radar Radiomyography: Cross-Domain Validation against Stethoscope Ground Truth\n'
        f'{sub_name} | {rec_name} | Korotkoff window: {k_on:.2f}–{k_off:.2f} s  '
        f'(Duration = {koro_dur:.2f} s)',
        fontsize=16, fontweight='bold', color=ctxt, y=0.999)

    plt.tight_layout(rect=[0.0, 0.034, 1.0, 0.994])
    plt.subplots_adjust(hspace=0.38, wspace=0.26)

    plt.savefig(out_path, dpi=300, facecolor=bg, bbox_inches='tight')
    plt.close(fig)

    # Calculate numeric SNR based on Peak Korotkoff vs Mean Baseline Noise
    peak_m = np.max(mag_tkeo_n[mask_act])
    noise_m = np.mean(mag_tkeo_n[mask_bas])
    snr_m = 10 * np.log10(peak_m / (noise_m + 1e-10))

    peak_p = np.max(phi_tkeo_n[mask_act])
    noise_p = np.mean(phi_tkeo_n[mask_bas])
    snr_p = 10 * np.log10(peak_p / (noise_p + 1e-10))

    print("\nDONE: " + out_path)
    print("=" * 55)
    print("  Subject          : " + sub_name)
    print("  Korotkoff Window : {:.3f} - {:.2f} s  ({:.3f} s)".format(k_on, k_off, koro_dur))
    print("  Magnitude TKEO SNR (Peak/Noise) : {:+.1f} dB".format(snr_m))
    print("  Phase Vel TKEO SNR (Peak/Noise) : {:+.1f} dB".format(snr_p))
    print("=" * 55)

def main():
    for s in [1, 2]:
        run_validation(s)

if __name__ == '__main__':
    main()
