import h5py, numpy as np, os, pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_may12_2.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\advanced_koro_validation_v3_may12_2.png'

def sliding_kurtosis(x, w): return pd.Series(x).rolling(window=w).kurt().fillna(0).values
def sliding_rms(x, w): return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)
def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def run():
    print(f"Starting v3 Analysis for: {os.path.basename(FILE_PATH)}")
    if not os.path.exists(FILE_PATH): print("File not found"); return

    with h5py.File(FILE_PATH, 'r') as f: data = f['data'][:]
    i_raw, q_raw = data[0,:], data[1,:]
    time = np.arange(len(i_raw)) / FS

    # PRE-PROCESSING
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    mag = np.abs(iq)
    phase_rad = signal.detrend(np.unwrap(np.angle(iq)))
    lambda_mm = (299792458 / 0.9e9) * 1000
    phase_mm = (phase_rad * lambda_mm) / (4 * np.pi)

    # 50 Hz NOTCH
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag = signal.filtfilt(b50, a50, mag)
    phase_mm = signal.filtfilt(b50, a50, phase_mm)

    # CUFF TREND
    cuff_trend = signal.sosfiltfilt(signal.butter(4, 0.2, btype='low', fs=FS, output='sos'), mag)

    # MAGNITUDE PULSE 1-49 Hz
    mag_pulse = signal.sosfiltfilt(signal.butter(4, [1, 49], btype='band', fs=FS, output='sos'), mag)
    mag_env = np.abs(signal.hilbert(mag_pulse))

    # PHASE VELOCITY 10-49 Hz
    phase_vel_raw = np.append(np.diff(phase_mm) * FS, 0)
    phase_vel = signal.sosfiltfilt(signal.butter(4, [10, 49], btype='band', fs=FS, output='sos'), phase_vel_raw)

    # WIDEBAND 0.5-150 Hz for spectrograms
    sos_w = signal.butter(4, [0.5, 150], btype='band', fs=FS, output='sos')
    mag_wide = signal.sosfiltfilt(sos_w, mag)
    phase_wide = signal.sosfiltfilt(sos_w, phase_mm)

    # ---- KOROTKOFF WINDOW DETECTION (improved: multi-metric, longer window) ----
    vel_tkeo = calc_tkeo(phase_vel)
    # Use energy envelope (smoother than raw TKEO) for window boundaries
    phase_energy = sliding_rms(phase_vel, int(FS * 0.3))**2
    # Smooth the energy with 2s window for stable boundaries
    smooth_energy = pd.Series(phase_energy).rolling(window=int(FS*2.0), center=True).mean().fillna(0).values
    v_s, v_e = int(2 * FS), int(len(smooth_energy) - FS)
    if v_s < v_e:
        center_idx = v_s + np.argmax(smooth_energy[v_s:v_e])
    else:
        center_idx = np.argmax(smooth_energy)

    # Lower threshold (8% of max) for wider window capture
    e_thresh = np.max(smooth_energy[v_s:v_e]) * 0.08
    si, ei = center_idx, center_idx
    while si > 0 and smooth_energy[si] > e_thresh: si -= 1
    while ei < len(smooth_energy)-1 and smooth_energy[ei] > e_thresh: ei += 1
    on_s, off_s = time[si], time[ei]

    # Enforce 4-15s duration
    dur = off_s - on_s
    if dur < 4.0:
        pad = (4.0 - dur) / 2.0
        on_s, off_s = max(0, on_s - pad), min(time[-1], off_s + pad)
    elif dur > 15.0:
        pad = (dur - 15.0) / 2.0
        on_s, off_s = on_s + pad, off_s - pad
    dur = off_s - on_s

    # ---- BEAT DETECTION (quiet-segment threshold) ----
    q_start = int(off_s * FS) + int(2*FS)
    q_end = min(len(mag_pulse), q_start + int(8*FS))
    if q_end > q_start + int(2*FS):
        prom_th = np.std(mag_pulse[q_start:q_end]) * 1.5
    else:
        prom_th = np.std(mag_pulse) * 1.0
    peaks, _ = signal.find_peaks(mag_pulse, distance=int(FS*0.4), prominence=prom_th)
    if len(peaks) > 1:
        ivals = np.diff(time[peaks])
        v_ivals = ivals[(ivals > 0.4) & (ivals < 1.5)]
        hr_bpm_t = 60.0/np.median(v_ivals) if len(v_ivals) > 0 else 0
    else:
        hr_bpm_t = 0

    # SLIDING STATS
    wl = int(FS * 0.5)
    mag_kurt = sliding_kurtosis(mag_pulse, wl)
    phase_kurt = sliding_kurtosis(phase_vel, wl)
    phase_jitter = sliding_rms(phase_vel, wl)
    koro_energy_env = sliding_rms(phase_vel, int(FS*0.2))**2
    koro_thresh_line = np.mean(koro_energy_env) + 2*np.std(koro_energy_env)

    # INSTANTANEOUS FREQUENCY
    inst_p = np.unwrap(np.angle(signal.hilbert(phase_vel)))
    inst_f = np.append(np.diff(inst_p)/(2*np.pi)*FS, 0)
    inst_f_sm = np.clip(pd.Series(inst_f).rolling(window=int(FS*0.05)).median().fillna(0).values, 0, 100)

    # PHASE VELOCITY CHANGE RATE (derivative of velocity envelope)
    vel_env = sliding_rms(phase_vel, int(FS*0.1))
    vel_change_rate = np.abs(np.append(np.diff(vel_env)*FS, 0))
    vel_change_smooth = pd.Series(vel_change_rate).rolling(window=int(FS*0.2)).mean().fillna(0).values

    # HR PSD (quiet segment)
    if q_end > q_start + int(2*FS):
        qp = phase_mm[q_start:q_end]
        f_hr, p_hr = signal.welch(qp, fs=FS, nperseg=min(len(qp), int(FS*5)))
    else:
        f_hr, p_hr = signal.welch(phase_mm, fs=FS, nperseg=int(FS*10))
    hr_mask = (f_hr >= 0.8) & (f_hr <= 3.0)
    hr_peak_f = f_hr[hr_mask][np.argmax(p_hr[hr_mask])] if np.any(hr_mask) else 0
    hr_bpm_f = hr_peak_f * 60

    # Active vs Noise (post-Koro noise)
    snr_db = 0
    if on_s < off_s:
        io, ie = int(on_s*FS), int(off_s*FS)
        act_v = phase_vel[io:ie]
        ns = min(ie + int(2*FS), len(phase_vel) - len(act_v))
        ns = max(ns, 0)
        noi_v = phase_vel[ns:ns+len(act_v)]
        nps = min(1024, len(act_v))
        f_act, p_act = signal.welch(act_v, fs=FS, nperseg=nps)
        f_noi, p_noi = signal.welch(noi_v, fs=FS, nperseg=nps)
        km = (f_act >= 10) & (f_act <= 49)
        if np.any(km) and np.mean(p_noi[km]) > 0:
            snr_db = 10*np.log10(np.mean(p_act[km])/np.mean(p_noi[km]))
    else:
        f_act = f_noi = p_act = p_noi = np.zeros(513)

    # ==================== PLOTTING: 10 rows x 2 cols = 20 panels ====================
    fig, axes = plt.subplots(10, 2, figsize=(22, 52))
    plt.subplots_adjust(hspace=0.50)
    yw = dict(color='yellow', alpha=0.2)

    # R1: Overview
    ax = axes[0,0]
    ax.plot(time, mag, color='blue', alpha=0.3, label='Magnitude')
    ax.plot(time, cuff_trend, color='black', lw=2, label='Cuff Trend (0.2 Hz LP)')
    if on_s < off_s:
        ax.axvspan(on_s, off_s, **yw, label=f'Koro Window ({dur:.1f}s)')
        ax.axvline(on_s, color='red', ls='--', lw=2, label='SYS'); ax.axvline(off_s, color='blue', ls='--', lw=2, label='DIA')
    ax.set_title('1. Magnitude + Cuff Trend'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[0,1]
    ax.plot(time, phase_mm, color='red', alpha=0.8, label='Detrended Displacement')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('1. Physical Displacement'); ax.set_ylabel('Displacement (mm)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R2: Filtered + Beats
    ax = axes[1,0]
    ax.plot(time, mag_pulse, color='blue', alpha=0.5, label='Bandpass 1-49 Hz')
    ax.plot(time, mag_env, color='gray', alpha=0.5, label='Hilbert Envelope')
    ax.plot(time[peaks], mag_pulse[peaks], 'ro', ms=5, label=f'Beats ({len(peaks)})')
    ax.text(0.02, 0.85, f'Time HR: {hr_bpm_t:.1f} BPM\n({len(peaks)} beats)', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title('2. Magnitude Pulse (1-49 Hz)'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[1,1]
    ax.plot(time, phase_vel, color='darkred', label='Phase Velocity (10-49 Hz)')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('2. Phase Velocity dφ/dt (10-49 Hz)'); ax.set_ylabel('Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R3: PSD
    ax = axes[2,0]
    fm, pm = signal.welch(mag, fs=FS, nperseg=FS*2)
    ax.semilogy(fm, pm, color='blue', label='Magnitude PSD')
    ax.axvspan(10, 49, **yw, label='Koro Band'); ax.axvline(50, color='red', ls=':', alpha=0.5, label='50 Hz Notch')
    ax.set_xlim(0,60); ax.set_ylim(bottom=1e-12); ax.set_title('3. Magnitude PSD'); ax.set_ylabel('PSD (a.u.²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    ax = axes[2,1]
    fp, pp = signal.welch(phase_mm, fs=FS, nperseg=FS*2)
    ax.semilogy(fp, pp, color='red', label='Displacement PSD')
    ax.axvspan(10, 49, **yw, label='Koro Band'); ax.axvline(50, color='red', ls=':', alpha=0.5, label='50 Hz Notch')
    ax.set_xlim(0,60); ax.set_ylim(bottom=1e-12); ax.set_title('3. Displacement PSD'); ax.set_ylabel('PSD (mm²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # R4: Spectrograms
    ax = axes[3,0]
    fs1, ts1, Z1 = signal.stft(mag_wide, fs=FS, nperseg=4096, noverlap=3072)
    P1 = 10*np.log10(np.abs(Z1)**2+1e-12); v1,v2 = np.percentile(P1,[50,99.9])
    im1 = ax.pcolormesh(ts1, fs1, P1, shading='gouraud', cmap='viridis', vmin=v1, vmax=v2)
    if on_s < off_s: ax.axvline(on_s, color='w', ls='--'); ax.axvline(off_s, color='w', ls='--')
    ax.set_ylim(0,150); ax.set_title('4. Magnitude Spectrogram (0.5-150 Hz)'); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im1, ax=ax, label='Power (dB)')

    ax = axes[3,1]
    fs2, ts2, Z2 = signal.stft(phase_wide, fs=FS, nperseg=4096, noverlap=3072)
    P2 = 10*np.log10(np.abs(Z2)**2+1e-12); v3,v4 = np.percentile(P2,[50,99.9])
    im2 = ax.pcolormesh(ts2, fs2, P2, shading='gouraud', cmap='plasma', vmin=v3, vmax=v4)
    if on_s < off_s: ax.axvline(on_s, color='w', ls='--'); ax.axvline(off_s, color='w', ls='--')
    ax.set_ylim(0,150); ax.set_title('4. Displacement Spectrogram (0.5-150 Hz)'); ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im2, ax=ax, label='Power (dB)')

    # R5: Kurtosis
    ax = axes[4,0]
    ax.plot(time, mag_kurt, color='purple'); 
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('5. Magnitude Sliding Kurtosis'); ax.set_ylabel('Kurtosis (unitless)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[4,1]
    ax.plot(time, phase_kurt, color='purple')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('5. Phase Velocity Sliding Kurtosis'); ax.set_ylabel('Kurtosis (unitless)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R6: Energy Envelope + Jitter
    ax = axes[5,0]
    ax.plot(time, koro_energy_env, color='teal', label='Phase Vel Energy')
    ax.axhline(koro_thresh_line, color='red', ls='--', label='Threshold (mean+2σ)')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6. Korotkoff Energy Envelope + Threshold'); ax.set_ylabel('Energy ((mm/s)²)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[5,1]
    ax.plot(time, phase_jitter, color='darkred', label='Velocity Jitter (RMS)')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6. Phase Velocity Jitter (Sliding RMS)'); ax.set_ylabel('RMS Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R7: FFT of Koro Window
    ax = axes[6,0]
    if on_s < off_s:
        wm = mag[int(on_s*FS):int(off_s*FS)]; wm = wm - np.mean(wm); Nw = len(wm)
        fw = np.fft.rfftfreq(Nw, 1/FS)
        ax.plot(fw, np.abs(np.fft.rfft(wm))/Nw, color='blue', label='Magnitude FFT')
        ax.set_xlim(0,60)
    ax.set_title('7. Amplitude Spectrum: Magnitude (Koro Window)'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    ax = axes[6,1]
    if on_s < off_s:
        wv = phase_vel[int(on_s*FS):int(off_s*FS)]; Nv = len(wv)
        fv = np.fft.rfftfreq(Nv, 1/FS)
        ax.plot(fv, np.abs(np.fft.rfft(wv))/Nv, color='red', label='Phase Vel FFT')
        ax.set_xlim(0,60)
    ax.set_title('7. Amplitude Spectrum: Phase Velocity (Koro Window)'); ax.set_ylabel('Amplitude (mm/s)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # R8: TKEO + Velocity Change Rate (WHERE velocity changes in phase)
    ax = axes[7,0]
    ax.plot(time, vel_tkeo, color='teal', label='TKEO Energy')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('8. TKEO (Teager-Kaiser Energy)'); ax.set_ylabel('TKEO Energy ((mm/s)²)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[7,1]
    ax.plot(time, vel_change_smooth, color='magenta', label='|d/dt(Velocity Envelope)|')
    # Mark the top velocity transition points
    trans_peaks, _ = signal.find_peaks(vel_change_smooth, distance=int(FS*0.3), prominence=np.std(vel_change_smooth)*3)
    if len(trans_peaks) > 0:
        top_n = min(10, len(trans_peaks))
        top_idx = trans_peaks[np.argsort(vel_change_smooth[trans_peaks])[-top_n:]]
        ax.plot(time[top_idx], vel_change_smooth[top_idx], 'rv', ms=8, label=f'Velocity Transitions ({len(top_idx)})')
    if on_s < off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('8. Phase Velocity Change Rate (Transition Detection)'); ax.set_ylabel('|d/dt Vel Env| ((mm/s)/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R9: HR PSD + Active vs Noise
    ax = axes[8,0]
    ax.semilogy(f_hr, p_hr, color='black', label='Displacement PSD (quiet)')
    if hr_peak_f > 0:
        ax.plot(hr_peak_f, p_hr[np.argmin(np.abs(f_hr-hr_peak_f))], 'ro', ms=8)
        ax.axvline(hr_peak_f, color='red', ls='--')
        ax.text(0.5, 0.80, f'Freq HR: {hr_bpm_f:.1f} BPM\nTime HR: {hr_bpm_t:.1f} BPM', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_xlim(0,5); ax.set_ylim(bottom=1e-8); ax.set_title('9. HR PSD (Quiet Post-Koro Segment)'); ax.set_ylabel('PSD (mm²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    ax = axes[8,1]
    if on_s < off_s:
        ax.semilogy(f_act, p_act, color='red', label='Active Koro Window')
        ax.semilogy(f_noi, p_noi, color='blue', alpha=0.5, label='Post-Koro Noise')
        ax.text(0.5, 0.80, f'SNR (10-49 Hz): {snr_db:.1f} dB', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='lime', alpha=0.5))
    ax.set_xlim(10,50); ax.set_title('9. Korotkoff Band: Active vs Noise + SNR'); ax.set_ylabel('PSD ((mm/s)²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # R10: ZOOMED OVERLAYS (Heartbeat vs Koro / TKEO)
    hr_sig = signal.sosfiltfilt(signal.butter(4, [0.8, 3.0], btype='band', fs=FS, output='sos'), mag)
    if on_s < off_s:
        zi, ze = max(0, int(on_s*FS) - int(2*FS)), min(len(time), int(off_s*FS) + int(2*FS))
        zt = time[zi:ze]
        z_hr = hr_sig[zi:ze]; z_vel = phase_vel[zi:ze]; z_tkeo = vel_tkeo[zi:ze]
        z_hr_n = z_hr / (np.max(np.abs(z_hr))+1e-9)
        z_vel_n = z_vel / (np.max(np.abs(z_vel))+1e-9)
        z_tkeo_n = z_tkeo / (np.max(np.abs(z_tkeo))+1e-9)
        # Also show velocity change rate zoomed
        z_vcr = vel_change_smooth[zi:ze]
        z_vcr_n = z_vcr / (np.max(np.abs(z_vcr))+1e-9)

        ax = axes[9,0]
        ax.plot(zt, z_hr_n, color='black', lw=2, label='Heartbeat (0.8-3 Hz)')
        ax.plot(zt, z_vel_n, color='red', alpha=0.7, label='Koro Snaps (10-49 Hz)')
        ax.axvspan(on_s, off_s, **yw, label='Koro Window')
        ax.set_xlim(zt[0], zt[-1])
        ax.set_title('10. Zoomed: Heartbeats vs Korotkoff Snaps'); ax.set_ylabel('Normalized Amplitude'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

        ax = axes[9,1]
        ax.plot(zt, z_hr_n, color='black', lw=2, label='Heartbeat (0.8-3 Hz)')
        ax.plot(zt, z_vcr_n, color='magenta', alpha=0.8, label='Velocity Change Rate')
        ax.axvspan(on_s, off_s, **yw, label='Koro Window')
        ax.set_xlim(zt[0], zt[-1])
        ax.set_title('10. Zoomed: Heartbeats vs Phase Velocity Transitions'); ax.set_ylabel('Normalized'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    else:
        axes[9,0].set_title('10. Zoomed Overlay: No Window Found')
        axes[9,1].set_title('10. Zoomed Overlay: No Window Found')

    plt.suptitle("Korotkoff Validation Dashboard v3: Extended Window + Velocity Transitions", fontsize=22, fontweight='bold', y=0.925)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\n20-Panel Plot saved to {OUTPUT_IMG}")
    print(f"\n{'='*50}")
    print(f"  KOROTKOFF ANALYSIS REPORT (v3)")
    print(f"{'='*50}")
    print(f"  Koro Window     : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)")
    print(f"  Time-Domain HR  : {hr_bpm_t:.1f} BPM ({len(peaks)} beats)")
    print(f"  Freq-Domain HR  : {hr_bpm_f:.1f} BPM")
    print(f"  Koro SNR        : {snr_db:.1f} dB (10-49 Hz)")
    print(f"{'='*50}\n")

if __name__ == '__main__': run()
