import h5py, numpy as np, os, pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_may12_2.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\advanced_koro_validation_v4_may12_2.png'

def sliding_kurtosis(x, w): return pd.Series(x).rolling(window=w).kurt().fillna(0).values
def sliding_rms(x, w): return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)
def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def run():
    print(f"Starting v4 Analysis for: {os.path.basename(FILE_PATH)}")
    if not os.path.exists(FILE_PATH): print("File not found"); return
    with h5py.File(FILE_PATH, 'r') as f: data = f['data'][:]
    i_raw, q_raw = data[0,:], data[1,:]
    time = np.arange(len(i_raw)) / FS

    # === SIGNAL CHAIN (with intermediate signals kept for plotting) ===
    # Step A: DC removal + complex IQ
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    mag = np.abs(iq)

    # Step B: Phase extraction (radians) - unwrap + detrend
    phase_unwrapped = np.unwrap(np.angle(iq))        # raw unwrapped (rad)
    phase_detrended = signal.detrend(phase_unwrapped) # detrended (rad)

    # Step C: Phase -> Displacement (mm):  d = (phi * lambda) / (4*pi)
    lambda_mm = (299792458 / 0.9e9) * 1000  # 333.1 mm
    phase_mm = (phase_detrended * lambda_mm) / (4 * np.pi)

    # Step D: Displacement -> Velocity (mm/s):  v = d(displacement)/dt
    phase_vel_raw = np.append(np.diff(phase_mm) * FS, 0)  # unfiltered velocity

    # Step E: Bandpass velocity 10-49 Hz (Korotkoff band)
    # 50 Hz notch first
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag = signal.filtfilt(b50, a50, mag)
    phase_mm_notched = signal.filtfilt(b50, a50, phase_mm)
    phase_vel_notched = np.append(np.diff(phase_mm_notched) * FS, 0)
    phase_vel = signal.sosfiltfilt(signal.butter(4, [10, 49], btype='band', fs=FS, output='sos'), phase_vel_notched)

    # Other derived signals
    cuff_trend = signal.sosfiltfilt(signal.butter(4, 0.2, btype='low', fs=FS, output='sos'), mag)
    mag_pulse = signal.sosfiltfilt(signal.butter(4, [1, 49], btype='band', fs=FS, output='sos'), mag)
    mag_env = np.abs(signal.hilbert(mag_pulse))
    sos_w = signal.butter(4, [0.5, 150], btype='band', fs=FS, output='sos')
    mag_wide = signal.sosfiltfilt(sos_w, mag)
    phase_wide = signal.sosfiltfilt(sos_w, phase_mm_notched)

    # KOROTKOFF WINDOW DETECTION (energy envelope, 8% threshold, 2s smoothing)
    vel_tkeo = calc_tkeo(phase_vel)
    phase_energy = sliding_rms(phase_vel, int(FS*0.3))**2
    smooth_energy = pd.Series(phase_energy).rolling(window=int(FS*2.0), center=True).mean().fillna(0).values
    v_s, v_e = int(2*FS), int(len(smooth_energy)-FS)
    center_idx = v_s + np.argmax(smooth_energy[v_s:v_e]) if v_s < v_e else np.argmax(smooth_energy)
    e_thresh = np.max(smooth_energy[v_s:v_e]) * 0.08
    si, ei = center_idx, center_idx
    while si > 0 and smooth_energy[si] > e_thresh: si -= 1
    while ei < len(smooth_energy)-1 and smooth_energy[ei] > e_thresh: ei += 1
    on_s, off_s = time[si], time[ei]
    dur = off_s - on_s
    if dur < 4.0:
        pad = (4.0-dur)/2; on_s, off_s = max(0, on_s-pad), min(time[-1], off_s+pad)
    elif dur > 15.0:
        pad = (dur-15.0)/2; on_s, off_s = on_s+pad, off_s-pad
    dur = off_s - on_s

    # BEAT DETECTION (quiet-segment threshold)
    qs, qe = int(off_s*FS)+int(2*FS), min(len(mag_pulse), int(off_s*FS)+int(10*FS))
    prom_th = np.std(mag_pulse[qs:qe])*1.5 if qe > qs+int(2*FS) else np.std(mag_pulse)
    peaks, _ = signal.find_peaks(mag_pulse, distance=int(FS*0.4), prominence=prom_th)
    if len(peaks) > 1:
        iv = np.diff(time[peaks]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_bpm_t = 60.0/np.median(viv) if len(viv) > 0 else 0
    else: hr_bpm_t = 0

    # SLIDING STATS
    wl = int(FS*0.5)
    mag_kurt = sliding_kurtosis(mag_pulse, wl)
    phase_kurt = sliding_kurtosis(phase_vel, wl)
    phase_jitter = sliding_rms(phase_vel, wl)
    koro_e_env = sliding_rms(phase_vel, int(FS*0.2))**2
    koro_th = np.mean(koro_e_env) + 2*np.std(koro_e_env)

    # VELOCITY CHANGE RATE
    vel_env = sliding_rms(phase_vel, int(FS*0.1))
    vel_cr = np.abs(np.append(np.diff(vel_env)*FS, 0))
    vel_cr_sm = pd.Series(vel_cr).rolling(window=int(FS*0.2)).mean().fillna(0).values

    # HR PSD (quiet segment)
    if qe > qs+int(2*FS):
        f_hr, p_hr = signal.welch(phase_mm_notched[qs:qe], fs=FS, nperseg=min(qe-qs, int(FS*5)))
    else:
        f_hr, p_hr = signal.welch(phase_mm_notched, fs=FS, nperseg=int(FS*10))
    hm = (f_hr>=0.8)&(f_hr<=3.0)
    hr_pk = f_hr[hm][np.argmax(p_hr[hm])] if np.any(hm) else 0
    hr_bpm_f = hr_pk*60

    # Active vs Noise
    snr_db = 0
    if on_s < off_s:
        io, ie = int(on_s*FS), int(off_s*FS)
        av = phase_vel[io:ie]; ns = min(ie+int(2*FS), len(phase_vel)-len(av))
        nv = phase_vel[max(0,ns):max(0,ns)+len(av)]
        nps = min(1024, len(av))
        f_a, p_a = signal.welch(av, fs=FS, nperseg=nps)
        f_n, p_n = signal.welch(nv, fs=FS, nperseg=nps)
        km = (f_a>=10)&(f_a<=49)
        if np.any(km) and np.mean(p_n[km])>0: snr_db = 10*np.log10(np.mean(p_a[km])/np.mean(p_n[km]))
    else: f_a=f_n=p_a=p_n=np.zeros(513)

    # ==================== PLOTTING: 11 rows x 2 cols = 22 panels ====================
    fig, axes = plt.subplots(11, 2, figsize=(22, 56))
    plt.subplots_adjust(hspace=0.55)
    yw = dict(color='yellow', alpha=0.2)

    # === ROW 1: Raw Overview ===
    ax = axes[0,0]
    ax.plot(time, mag, color='blue', alpha=0.3, label='Magnitude')
    ax.plot(time, cuff_trend, color='black', lw=2, label='Cuff Trend (0.2 Hz LP)')
    if on_s<off_s:
        ax.axvspan(on_s, off_s, **yw, label=f'Koro Window ({dur:.1f}s)')
        ax.axvline(on_s, color='red', ls='--', lw=2, label='SYS'); ax.axvline(off_s, color='blue', ls='--', lw=2, label='DIA')
    ax.set_title('1. Magnitude + Cuff Trend'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[0,1]
    ax.plot(time, phase_unwrapped, color='gray', alpha=0.4, label='Unwrapped Phase (before detrend)')
    ax.plot(time, phase_detrended, color='green', lw=1.5, label='Detrended Phase')
    ax.set_title('1. Phase Extraction: Unwrap + Detrend'); ax.set_ylabel('Phase (radians)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # === ROW 2: DERIVATION CHAIN — Phase(rad) → Displacement(mm) ===
    ax = axes[1,0]
    ax.plot(time, phase_detrended, color='green', label='Detrended Phase φ(t)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('2a. Detrended Phase φ(t)'); ax.set_ylabel('Phase (radians)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.text(0.02, 0.02, 'Step A: φ(t) = detrend(unwrap(∠IQ))', transform=ax.transAxes, fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    ax = axes[1,1]
    ax.plot(time, phase_mm, color='red', label='Displacement d(t)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('2b. Phase → Displacement: d = (φ × λ) / (4π)'); ax.set_ylabel('Displacement (mm)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.text(0.02, 0.02, f'Step B: d(t) = φ(t) × {lambda_mm:.1f}mm / (4π)', transform=ax.transAxes, fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    # === ROW 3: DERIVATION CHAIN — Displacement(mm) → Velocity(mm/s) ===
    ax = axes[2,0]
    ax.plot(time, phase_vel_raw, color='orange', alpha=0.6, label='Raw Velocity (unfiltered)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('3a. Displacement → Raw Velocity: v = d(d)/dt'); ax.set_ylabel('Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.text(0.02, 0.02, 'Step C: v(t) = Δd/Δt = diff(d) × Fs', transform=ax.transAxes, fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    ax = axes[2,1]
    ax.plot(time, phase_vel, color='darkred', label='Filtered Velocity (10-49 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('3b. Bandpass Filtered Velocity (Korotkoff Band)'); ax.set_ylabel('Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.text(0.02, 0.02, 'Step D: BPF 10-49 Hz + 50 Hz Notch', transform=ax.transAxes, fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    # === ROW 4: Magnitude Pulse + Beats ===
    ax = axes[3,0]
    ax.plot(time, mag_pulse, color='blue', alpha=0.5, label='Bandpass 1-49 Hz')
    ax.plot(time, mag_env, color='gray', alpha=0.5, label='Hilbert Envelope')
    ax.plot(time[peaks], mag_pulse[peaks], 'ro', ms=5, label=f'Beats ({len(peaks)})')
    ax.text(0.02, 0.85, f'Time HR: {hr_bpm_t:.1f} BPM\n({len(peaks)} beats)', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title('4. Magnitude Pulse (1-49 Hz)'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[3,1]
    fm, pm = signal.welch(mag, fs=FS, nperseg=FS*2)
    ax.semilogy(fm, pm, color='blue', label='Magnitude PSD')
    ax.axvspan(10, 49, **yw, label='Koro Band'); ax.axvline(50, color='red', ls=':', alpha=0.5, label='50 Hz Notch')
    ax.set_xlim(0,60); ax.set_ylim(bottom=1e-12); ax.set_title('4. Magnitude PSD'); ax.set_ylabel('PSD (a.u.²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # === ROW 5: Spectrograms ===
    ax = axes[4,0]
    fs1,ts1,Z1 = signal.stft(mag_wide, fs=FS, nperseg=4096, noverlap=3072)
    P1 = 10*np.log10(np.abs(Z1)**2+1e-12); v1,v2 = np.percentile(P1,[50,99.9])
    im1 = ax.pcolormesh(ts1, fs1, P1, shading='gouraud', cmap='viridis', vmin=v1, vmax=v2)
    if on_s<off_s: ax.axvline(on_s, color='w', ls='--'); ax.axvline(off_s, color='w', ls='--')
    ax.set_ylim(0,150); ax.set_title('5. Magnitude Spectrogram (0.5-150 Hz)'); ax.set_ylabel('Freq (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im1, ax=ax, label='Power (dB)')

    ax = axes[4,1]
    fs2,ts2,Z2 = signal.stft(phase_wide, fs=FS, nperseg=4096, noverlap=3072)
    P2 = 10*np.log10(np.abs(Z2)**2+1e-12); v3,v4 = np.percentile(P2,[50,99.9])
    im2 = ax.pcolormesh(ts2, fs2, P2, shading='gouraud', cmap='plasma', vmin=v3, vmax=v4)
    if on_s<off_s: ax.axvline(on_s, color='w', ls='--'); ax.axvline(off_s, color='w', ls='--')
    ax.set_ylim(0,150); ax.set_title('5. Displacement Spectrogram (0.5-150 Hz)'); ax.set_ylabel('Freq (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im2, ax=ax, label='Power (dB)')

    # === ROW 6: Kurtosis ===
    ax = axes[5,0]
    ax.plot(time, mag_kurt, color='purple')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6. Magnitude Sliding Kurtosis'); ax.set_ylabel('Kurtosis (unitless)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[5,1]
    ax.plot(time, phase_kurt, color='purple')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6. Phase Velocity Sliding Kurtosis'); ax.set_ylabel('Kurtosis (unitless)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # === ROW 7: Energy Envelope + Jitter ===
    ax = axes[6,0]
    ax.plot(time, koro_e_env, color='teal', label='Phase Vel Energy')
    ax.axhline(koro_th, color='red', ls='--', label='Threshold (mean+2σ)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('7. Korotkoff Energy Envelope'); ax.set_ylabel('Energy ((mm/s)²)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[6,1]
    ax.plot(time, phase_jitter, color='darkred', label='Velocity Jitter (RMS)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('7. Phase Velocity Jitter (Sliding RMS)'); ax.set_ylabel('RMS Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # === ROW 8: TKEO + Velocity Change Rate ===
    ax = axes[7,0]
    ax.plot(time, vel_tkeo, color='teal', label='TKEO Energy')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('8. TKEO (Teager-Kaiser Energy)'); ax.set_ylabel('TKEO Energy ((mm/s)²)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[7,1]
    ax.plot(time, vel_cr_sm, color='magenta', label='|d/dt(Vel Envelope)|')
    tp, _ = signal.find_peaks(vel_cr_sm, distance=int(FS*0.3), prominence=np.std(vel_cr_sm)*3)
    if len(tp) > 0:
        top = tp[np.argsort(vel_cr_sm[tp])[-min(10,len(tp)):]]
        ax.plot(time[top], vel_cr_sm[top], 'rv', ms=8, label=f'Transitions ({len(top)})')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('8. Phase Velocity Change Rate'); ax.set_ylabel('|dv/dt| ((mm/s)/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # === ROW 9: HR PSD + Active vs Noise ===
    ax = axes[8,0]
    ax.semilogy(f_hr, p_hr, color='black', label='Displacement PSD (quiet)')
    if hr_pk > 0:
        ax.plot(hr_pk, p_hr[np.argmin(np.abs(f_hr-hr_pk))], 'ro', ms=8); ax.axvline(hr_pk, color='red', ls='--')
        ax.text(0.5, 0.80, f'Freq HR: {hr_bpm_f:.1f} BPM\nTime HR: {hr_bpm_t:.1f} BPM', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_xlim(0,5); ax.set_ylim(bottom=1e-8); ax.set_title('9. HR PSD (Quiet Post-Koro)'); ax.set_ylabel('PSD (mm²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    ax = axes[8,1]
    if on_s<off_s:
        ax.semilogy(f_a, p_a, color='red', label='Active Koro Window')
        ax.semilogy(f_n, p_n, color='blue', alpha=0.5, label='Post-Koro Noise')
        ax.text(0.5, 0.80, f'SNR (10-49 Hz): {snr_db:.1f} dB', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='lime', alpha=0.5))
    ax.set_xlim(10,50); ax.set_title('9. Active vs Noise PSD + SNR'); ax.set_ylabel('PSD ((mm/s)²/Hz)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # === ROW 10: FFT of Koro Window ===
    ax = axes[9,0]
    if on_s<off_s:
        wm = mag[int(on_s*FS):int(off_s*FS)]; wm -= np.mean(wm); Nw = len(wm)
        fw = np.fft.rfftfreq(Nw, 1/FS)
        ax.plot(fw, np.abs(np.fft.rfft(wm))/Nw, color='blue', label='Magnitude FFT')
        ax.set_xlim(0,60)
    ax.set_title('10. Koro Window FFT: Magnitude'); ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    ax = axes[9,1]
    if on_s<off_s:
        wv = phase_vel[int(on_s*FS):int(off_s*FS)]; Nv = len(wv)
        fv = np.fft.rfftfreq(Nv, 1/FS)
        ax.plot(fv, np.abs(np.fft.rfft(wv))/Nv, color='red', label='Phase Vel FFT')
        ax.set_xlim(0,60)
    ax.set_title('10. Koro Window FFT: Phase Velocity'); ax.set_ylabel('Amplitude (mm/s)'); ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=7)

    # === ROW 11: Zoomed Overlays ===
    hr_sig = signal.sosfiltfilt(signal.butter(4, [0.8, 3.0], btype='band', fs=FS, output='sos'), mag)
    if on_s<off_s:
        zi, ze = max(0, int(on_s*FS)-int(2*FS)), min(len(time), int(off_s*FS)+int(2*FS))
        zt = time[zi:ze]
        zhn = hr_sig[zi:ze]; zhn = zhn/(np.max(np.abs(zhn))+1e-9)
        zvn = phase_vel[zi:ze]; zvn = zvn/(np.max(np.abs(zvn))+1e-9)
        zvc = vel_cr_sm[zi:ze]; zvc = zvc/(np.max(np.abs(zvc))+1e-9)

        ax = axes[10,0]
        ax.plot(zt, zhn, color='black', lw=2, label='Heartbeat (0.8-3 Hz)')
        ax.plot(zt, zvn, color='red', alpha=0.7, label='Koro Snaps (10-49 Hz)')
        ax.axvspan(on_s, off_s, **yw, label='Koro Window'); ax.set_xlim(zt[0], zt[-1])
        ax.set_title('11. Zoomed: Heartbeats vs Korotkoff Snaps'); ax.set_ylabel('Normalized Amplitude'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

        ax = axes[10,1]
        ax.plot(zt, zhn, color='black', lw=2, label='Heartbeat (0.8-3 Hz)')
        ax.plot(zt, zvc, color='magenta', alpha=0.8, label='Velocity Change Rate')
        ax.axvspan(on_s, off_s, **yw, label='Koro Window'); ax.set_xlim(zt[0], zt[-1])
        ax.set_title('11. Zoomed: Heartbeats vs Velocity Transitions'); ax.set_ylabel('Normalized'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    else:
        axes[10,0].set_title('11. No Window Found'); axes[10,1].set_title('11. No Window Found')

    plt.suptitle("Korotkoff Dashboard v4: Full Velocity Derivation Chain + Multi-Domain Validation", fontsize=22, fontweight='bold', y=0.93)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\n22-Panel Plot saved to {OUTPUT_IMG}")
    print(f"\n{'='*55}")
    print(f"  KOROTKOFF ANALYSIS REPORT (v4)")
    print(f"{'='*55}")
    print(f"  Signal Chain: IQ -> Phase(rad) -> Displacement(mm) -> Velocity(mm/s)")
    print(f"  Lambda = {lambda_mm:.1f} mm (at 0.9 GHz)")
    print(f"  Koro Window  : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)")
    print(f"  Time-Domain HR: {hr_bpm_t:.1f} BPM ({len(peaks)} beats)")
    print(f"  Freq-Domain HR: {hr_bpm_f:.1f} BPM")
    print(f"  Koro SNR      : {snr_db:.1f} dB (10-49 Hz)")
    print(f"{'='*55}\n")

if __name__ == '__main__': run()
