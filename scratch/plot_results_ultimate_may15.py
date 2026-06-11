"""
Ultimate RF Signal Diagnostic & Spectral Analysis Dashboard
===========================================================
Loads the 'rec_koro_may15.h5' dataset, applies advanced IQ conditioning 
and robust phase unwrapping, performs physical unit conversions,
filters physiological bands, and executes spectral (Welch PSD & STFT) analysis.
"""
import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch, spectrogram
import matplotlib.pyplot as plt

# ── CONFIGURATION ────────────────────────────────────────────────
FILE_NAME  = 'rec_koro_may15.h5'
DATA_DIR   = r'd:\Bioview\My_RF_work_v1\data_new'
FILE_PATH  = os.path.join(DATA_DIR, FILE_NAME)
OUTPUT_IMG = os.path.join(DATA_DIR, 'ultimate_analysis_may15.png')

FS         = 10_000   # Sampling rate (Hz)
FC_HZ      = 0.9e9    # Radar carrier frequency (0.9 GHz)
C          = 299792458.0
LAMBDA_MM  = (C / FC_HZ) * 1000
SCALE      = LAMBDA_MM / (4 * np.pi)

# ── SIGNAL PROCESSING HELPERS ────────────────────────────────────
def sliding_rms(x, w): 
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)

def ac_couple_signal(x, fs, window_sec=3.0):
    w = int(window_sec * fs)
    if w % 2 == 0:
        w += 1
    rolling_mean = pd.Series(x).rolling(window=w, center=True, min_periods=1).mean().values
    return x - rolling_mean

def apply_iq(i, q):
    return -i + 1j * q  # Standard B210 IQ mode

def iq_condition(iq, keep_dc=True):
    """
    Performs hardware-level IQ balance correction (amplitude & phase imbalance).
    If keep_dc is True, preserves the raw static DC offsets to keep the calculated
    phase and displacement in their true, physical sub-millimeter scales.
    """
    i_mean, q_mean = iq.real.mean(), iq.imag.mean()
    ic, qc = iq.real - i_mean, iq.imag - q_mean
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc_corr = (qc - sp*ic) / (al*cp + 1e-15)
    else:
        qc_corr = qc
    if keep_dc:
        return (ic + i_mean) + 1j * (qc_corr + q_mean)
    else:
        return ic + 1j * qc_corr


def robust_phase_unwrap(iq):
    """
    Implements a robust phase unwrapping algorithm based on the v5 pipeline,
    which uses the median of dphi for stable carrier offset estimation and clips outlier jumps
    to completely eliminate noise-induced random walks and oscillator drifts.
    """
    phase_unwrap = np.unwrap(np.angle(iq))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.median(dphi)
    dphi_clean = dphi - carrier_offset
    dphi_clean = np.clip(dphi_clean, -0.5, 0.5)
    phase_clean = np.insert(np.cumsum(dphi_clean), 0, 0.0)
    return signal.detrend(phase_clean)


# ── MAIN EXECUTION ────────────────────────────────────────────────
def run_diagnostic():
    print(f"Loading H5 file: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Target H5 file not found: {FILE_PATH}")
        
    with h5py.File(FILE_PATH, 'r') as f:
        data = f['data'][:]
        attrs = dict(f.attrs)
    
    fs_actual = float(attrs.get('sample_rate', attrs.get('fs', FS)))
    i_raw, q_raw = data[0,:], data[1,:]
    N = len(i_raw)
    time = np.arange(N) / fs_actual
    rec_dur = time[-1]
    
    print(f"  Successfully loaded {N} samples | {rec_dur:.1f}s duration | fs={fs_actual:.0f} Hz")

    # 1. Centering & IQ balance conditioning on full signal (to get stable parameters)
    iq_raw_full = apply_iq(i_raw, q_raw)
    iq_clean_full = iq_condition(iq_raw_full)

    # 2. Start from 0.0 seconds to process and display all the time data
    start_sample = int(0.0 * fs_actual)
    time = time[start_sample:]
    N = len(time)
    iq_clean = iq_clean_full[start_sample:]
    i_raw = i_raw[start_sample:]
    q_raw = q_raw[start_sample:]

    # 3. Raw amplitude & phase
    magnitude_au = np.abs(iq_clean)
    
    # Unwrap phase strictly for the deflation period (t >= 20.0s) to bypass the huge cuff-inflation step function
    idx_deflation = int(20.0 * fs_actual)
    phase_rad_deflation = robust_phase_unwrap(iq_clean[idx_deflation:])
    
    # Fill the first 20 seconds with a smooth linear ramp from 0.0 to the starting phase of deflation
    phase_rad_raw = np.zeros(len(iq_clean))
    phase_rad_raw[idx_deflation:] = phase_rad_deflation
    phase_rad_raw[:idx_deflation] = np.linspace(0.0, phase_rad_deflation[0], idx_deflation)
    
    # AC-couple the unwrapped phase to achieve maximum peak-to-peak physiological visibility
    phase_rad = ac_couple_signal(phase_rad_raw, fs_actual, window_sec=3.0)

    # 4. Physical conversions (mm)
    magnitude_mm = magnitude_au * SCALE
    displacement_mm = phase_rad * SCALE

    # 5. Velocity calculation (mm/s)
    velocity_mms = np.diff(displacement_mm) * fs_actual
    velocity_mms = np.append(velocity_mms, velocity_mms[-1])  # maintain length

    # 6. Heart Rate and Korotkoff bandpass filtering
    sos_hr = butter(4, [0.7, 2.5], btype='bandpass', fs=fs_actual, output='sos')
    hr_pulse_mm = sosfiltfilt(sos_hr, displacement_mm)

    sos_koro = butter(4, [10, 50], btype='bandpass', fs=fs_actual, output='sos')
    koro_signal_mms = sosfiltfilt(sos_koro, velocity_mms)

    # 6. Spectral/PSD calculations
    # Heart Rate PSD
    hr_detrend = signal.detrend(hr_pulse_mm)
    freqs_hr, psd_hr = welch(hr_detrend, fs=fs_actual, nperseg=min(len(hr_detrend), int(fs_actual * 15)))
    mask_hr_psd = (freqs_hr >= 0.5) & (freqs_hr <= 3.0)
    hr_psd_bpm = freqs_hr[mask_hr_psd][np.argmax(psd_hr[mask_hr_psd])] * 60.0 if np.any(mask_hr_psd) else 0.0

    # Korotkoff PSD
    koro_detrend = signal.detrend(koro_signal_mms)
    freqs_koro, psd_koro = welch(koro_detrend, fs=fs_actual, nperseg=min(len(koro_detrend), int(fs_actual * 4)))

    # STFT Spectrogram for Korotkoff velocity
    nperseg_s = int(fs_actual / 4)
    f_spec, t_spec, Sxx = spectrogram(koro_signal_mms, fs_actual, nperseg=nperseg_s, noverlap=int(nperseg_s * 0.75))

    # Time-Domain Heart Rate Peaks
    t_stable = hr_pulse_mm[int(10*fs_actual):int(20*fs_actual)]
    pth = np.std(t_stable) * 0.8
    peaks_hr, _ = signal.find_peaks(-hr_pulse_mm, distance=int(fs_actual*0.5), prominence=pth)
    if len(peaks_hr) > 1:
        iv = np.diff(time[peaks_hr])
        valid_iv = iv[(iv > 0.4) & (iv < 1.5)]
        hr_peaks_bpm = 60.0 / np.median(valid_iv) if len(valid_iv) > 0 else 0.0
    else:
        hr_peaks_bpm = 0.0

    # 6. Automatic Korotkoff Window Detection (from v5 algorithm)
    ph_energy = sliding_rms(koro_signal_mms, int(fs_actual * 0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(fs_actual * 2), center=True).mean().fillna(0).values
    
    T_SKIP = 5.0
    vs = int(T_SKIP * fs_actual)
    ve = min(int(len(sm_energy) - T_SKIP * fs_actual), int(40.0 * fs_actual))
    
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth:
        si -= 1
    while ei < len(sm_energy) - 1 and sm_energy[ei] > eth:
        ei += 1
        
    on_s = time[max(si, int(T_SKIP * fs_actual))]
    off_s = time[min(ei, int((time[-1] - T_SKIP) * fs_actual))]
    dur = off_s - on_s
    
    # Enforce standard duration constraints
    if dur < 4.0:
        p = (4.0 - dur) / 2
        on_s, off_s = max(20.0, on_s - p), min(time[-1], off_s + p)
    elif dur > 18.0:
        p = (dur - 18.0) / 2
        on_s, off_s = on_s + p, off_s - p
    dur = off_s - on_s

    # 7. Zoomed Active Region Drift Calculations
    mask_active = (time >= on_s) & (time <= off_s)
    time_active = time[mask_active]
    disp_active = displacement_mm[mask_active]
    koro_snaps_active = koro_signal_mms[mask_active]
    
    # Welch PSD for the Korotkoff snaps ONLY in the active Korotkoff window
    freqs_koro_active, psd_koro_active = welch(koro_snaps_active, fs=fs_actual, nperseg=min(len(koro_snaps_active), int(fs_actual * 2)))

    # ── PLOTTING HIGH-FIDELITY DIAGNOSTIC ──
    print("Generating comprehensive diagnostic plots...")
    fig = plt.figure(figsize=(20, 26))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)

    # --- ROW 1: Raw Complex Magnitude and Unwrapped Phase (radians) ---
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(time, magnitude_au - np.mean(magnitude_au), color='teal', lw=0.6)
    ax1.axvspan(on_s, off_s, color='yellow', alpha=0.15, label=f'Koro Window ({dur:.1f}s)')
    ax1.set_title('1a. Complex Magnitude Overview (a.u.)', fontweight='bold')
    ax1.set_ylabel('Magnitude Deviation (a.u.)'); ax1.set_xlabel('Time (s)'); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

    ax2 = plt.subplot(5, 2, 2)
    ax2.plot(time, phase_rad, color='crimson', lw=0.6)
    ax2.axvspan(on_s, off_s, color='yellow', alpha=0.15)
    ax2.set_title('1b. Unwrapped Phase Overview (radians)', fontweight='bold')
    ax2.set_ylabel('Phase (radians)'); ax2.set_xlabel('Time (s)'); ax2.grid(True, alpha=0.3)

    # --- ROW 2: Displacement (RMG Scale) and Zoomed Velocity (Change in Displacement) ---
    ax3 = plt.subplot(5, 2, 3)
    ax3.plot(time, displacement_mm, color='firebrick', lw=0.6)
    ax3.axvspan(on_s, off_s, color='yellow', alpha=0.15)
    ax3.set_title('2a. Physical Displacement (mm, like RMG paper)', fontweight='bold')
    ax3.set_ylabel('Displacement (mm)'); ax3.set_xlabel('Time (s)'); ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(5, 2, 4)
    # Velocity overview for all the time data with yellow highlighted active window
    ax4.plot(time, velocity_mms, color='purple', lw=0.6, alpha=0.4, label='Raw Velocity (d/dt)')
    ax4.plot(time, koro_signal_mms, color='indigo', lw=0.8, label='BPF snaps (10-50 Hz)')
    ax4.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax4.set_title('2b. Change in Displacement (Velocity) Overview', fontweight='bold')
    ax4.set_ylabel('Velocity (mm/s)'); ax4.set_xlabel('Time (s)'); ax4.grid(True, alpha=0.3); ax4.legend(fontsize=8)

    # --- ROW 3: Filtered Physiological Waves ---
    ax5 = plt.subplot(5, 2, 5)
    ax5.plot(time, hr_pulse_mm, color='darkred', lw=0.8)
    if len(peaks_hr) > 0:
        ax5.plot(time[peaks_hr], hr_pulse_mm[peaks_hr], 'bo', ms=4, label=f'Beats ({hr_peaks_bpm:.0f} BPM)')
    ax5.set_title(f'3a. Heart Rate Pulse (0.7-2.5 Hz band) | {hr_peaks_bpm:.1f} BPM', fontweight='bold')
    ax5.set_ylabel('Displacement (mm)'); ax5.set_xlabel('Time (s)'); ax5.grid(True, alpha=0.3); ax5.legend(fontsize=8)

    ax6 = plt.subplot(5, 2, 6)
    ax6.plot(time, koro_signal_mms, color='indigo', lw=0.6, label='Velocity Koro (10-50 Hz)')
    ax6.axvspan(on_s, off_s, color='yellow', alpha=0.2, label='Koro Window')
    ax6.set_title('3b. Korotkoff Acoustic Signal (10-50 Hz band)', fontweight='bold')
    ax6.set_ylabel('Velocity (mm/s)'); ax6.set_xlabel('Time (s)'); ax6.grid(True, alpha=0.3); ax6.legend(fontsize=8)

    # --- ROW 4: Spectral Domain Analysis (t >= 20.0s) ---
    ax7 = plt.subplot(5, 2, 7)
    ax7.semilogy(freqs_hr, psd_hr, color='darkred', lw=1.5)
    ax7.axvline(hr_psd_bpm / 60.0, color='blue', ls='--', label=f'Peak: {hr_psd_bpm:.1f} BPM')
    ax7.set_xlim(0, 5)
    ax7.set_title('4a. Heart Rate Power Spectral Density (PSD)', fontweight='bold')
    ax7.set_ylabel('PSD (mm²/Hz)'); ax7.set_xlabel('Frequency (Hz)'); ax7.grid(True, alpha=0.3); ax7.legend(fontsize=8)

    ax8 = plt.subplot(5, 2, 8)
    P_db = 10 * np.log10(Sxx + 1e-20)
    f_mask = (f_spec >= 0) & (f_spec <= 60)
    va, vb = np.percentile(P_db[f_mask,:], [25, 99.5])
    im = ax8.pcolormesh(t_spec, f_spec[f_mask], P_db[f_mask,:], shading='gouraud', cmap='magma', vmin=va, vmax=vb)
    ax8.axvline(on_s, color='w', ls='--', lw=1.5)
    ax8.axvline(off_s, color='w', ls='--', lw=1.5)
    ax8.set_title('4b. Time-Frequency Spectrogram of Korotkoff Signal', fontweight='bold')
    ax8.set_ylabel('Frequency (Hz)'); ax8.set_xlabel('Time (s)')
    plt.colorbar(im, ax=ax8, label='Relative Power Density (dB)')

    # --- ROW 5: Comprehensive Summary Report ---
    ax9 = plt.subplot(5, 1, 5)
    ax9.axis('off')
    
    summary_txt = [
        f"RF SIGNAL COMPREHENSIVE DIAGNOSTIC REPORT",
        f"{'='*60}",
        f"Target Dataset   : {FILE_NAME}",
        f"Carrier Freq     : {FC_HZ/1e9:.2f} GHz (Sensing Wavelength: {LAMBDA_MM:.2f} mm)",
        f"Sensing Scale    : 1 radian = {SCALE:.4f} mm physical displacement",
        f"Total Samples    : {N} samples over {rec_dur:.2f} seconds",
        f"Actual fs        : {fs_actual:.0f} Hz",
        f"",
        f"PHYSIOLOGICAL MEASUREMENTS:",
        f"  Time-Domain HR : {hr_peaks_bpm:.2f} BPM (from peak interval calculation)",
        f"  Spectral HR    : {hr_psd_bpm:.2f} BPM (from Welch PSD dominant peak)",
        f"  Peak Agreement : {abs(hr_peaks_bpm - hr_psd_bpm):.2f} BPM difference",
        f"  Max Chest Disp : {np.max(displacement_mm) - np.min(displacement_mm):.4f} mm peak-to-peak",
        f"  Max Koro Vel   : {np.max(np.abs(koro_signal_mms)):.4f} mm/s peak-to-peak",
        f"{'='*60}",
        f"STATUS           : DIAGNOSTIC RETRIEVED SUCCESSFULLY",
    ]
    ax9.text(0.05, 0.95, '\n'.join(summary_txt), fontsize=13, family='monospace',
             fontweight='bold', va='top', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    fig.suptitle(f'Ultimate RF Diagnostic Dashboard — {FILE_NAME}', fontsize=24, fontweight='bold', y=0.99)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved successfully: {OUTPUT_IMG}")


if __name__ == '__main__':
    run_diagnostic()
