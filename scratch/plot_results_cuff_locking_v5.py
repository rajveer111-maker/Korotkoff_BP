import h5py, numpy as np, os, pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_below.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_dashboard_v5d_rec_koro_below.png'

def sliding_rms(x, w): return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)
def sliding_kurtosis(x, w): return pd.Series(x).rolling(window=w).kurt().fillna(0).values
def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def run():
    if not os.path.exists(FILE_PATH): print("File not found"); return
    with h5py.File(FILE_PATH, 'r') as f: data = f['data'][:]
    i_raw, q_raw = data[0,:], data[1,:]
    time = np.arange(len(i_raw)) / FS
    N = len(i_raw)

    # === SIGNAL PROCESSING ===
    # Use average reference to center the IF circle (v4 standard)
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    mag_raw = np.abs(iq)

    # Phase: unwrap -> remove carrier offset -> clip artifacts -> reconstruct
    phase_unwrap = np.unwrap(np.angle(iq))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.median(dphi)  # rad/sample (carrier freq offset)
    dphi_clean = dphi - carrier_offset
    dphi_clean = np.clip(dphi_clean, -0.5, 0.5)  # remove unwrap failures
    phase_clean = np.insert(np.cumsum(dphi_clean), 0, 0)
    phase_clean = signal.detrend(phase_clean)  # remove residual linear drift

    # Physical conversion
    LAMBDA_MM = (299792458 / 0.9e9) * 1000  # 333.1 mm
    SCALE = LAMBDA_MM / (4 * np.pi)          # 26.53 mm/rad

    # 50 Hz notch
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag = signal.filtfilt(b50, a50, mag_raw)

    # ---- DERIVATION CHAIN ----
    # A) Phase (clean, rad) -> bandpass into HR and Koro bands
    sos_hr = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
    sos_koro = signal.butter(4, [10, 49], btype='band', fs=FS, output='sos')

    phase_hr = signal.sosfiltfilt(sos_hr, phase_clean)     # HR phase (rad)
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)  # Koro phase (rad)

    # B) Phase -> Displacement: d = phase * lambda/(4*pi)
    disp_hr = phase_hr * SCALE      # mm
    disp_koro = phase_koro * SCALE   # mm

    # C) Displacement -> Velocity: v = d(disp)/dt
    vel_hr = np.append(np.diff(disp_hr) * FS, 0)      # mm/s
    vel_koro = np.append(np.diff(disp_koro) * FS, 0)   # mm/s

    # Other signals
    hr_mag = signal.sosfiltfilt(sos_hr, mag)

    # Koro window detection — skip first 5s and last 5s (noise/transients)
    vel_tkeo = calc_tkeo(vel_koro)
    ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
    T_SKIP = 5  # seconds to skip at start and end
    vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS)) # ignore motion after 40s
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    on_s, off_s = time[max(si, int(T_SKIP*FS))], time[min(ei, int((time[-1]-T_SKIP)*FS))]
    dur = off_s - on_s
    if dur < 4.0:
        p = (4.0-dur)/2; on_s, off_s = max(0, on_s-p), min(time[-1], off_s+p)
    elif dur > 15.0:
        p = (dur-15.0)/2; on_s, off_s = on_s+p, off_s-p
    dur = off_s - on_s

    # Beat detection
    qs = int(off_s*FS)+int(2*FS); qe = min(len(hr_mag), qs+int(8*FS))
    pth = np.std(hr_mag[qs:qe])*1.5 if qe > qs+int(2*FS) else np.std(hr_mag)
    peaks, _ = signal.find_peaks(hr_mag, distance=int(FS*0.4), prominence=pth)
    if len(peaks) > 1:
        iv = np.diff(time[peaks]); viv = iv[(iv>0.4)&(iv<1.5)]
        hr_bpm_t = 60.0/np.median(viv) if len(viv) > 0 else 0
    else: hr_bpm_t = 0

    # Stats
    wl = int(FS*0.5)
    vel_kurt = sliding_kurtosis(vel_koro, wl)
    vel_jitter = sliding_rms(vel_koro, wl)
    koro_env = sliding_rms(vel_koro, int(FS*0.2))**2
    koro_th = np.mean(koro_env) + 2*np.std(koro_env)
    vel_e = sliding_rms(vel_koro, int(FS*0.1))
    vel_cr = pd.Series(np.abs(np.append(np.diff(vel_e)*FS, 0))).rolling(int(FS*0.2)).mean().fillna(0).values

    # HR PSD
    if qe > qs+int(2*FS):
        f_hr, p_hr = signal.welch(disp_hr[qs:qe], fs=FS, nperseg=min(qe-qs, int(FS*5)))
    else:
        f_hr, p_hr = signal.welch(disp_hr, fs=FS, nperseg=int(FS*10))
    hm = (f_hr>=0.5)&(f_hr<=3.0)
    hr_pk = f_hr[hm][np.argmax(p_hr[hm])] if np.any(hm) else 0
    hr_bpm_f = hr_pk*60

    # Active vs Noise
    snr_db = 0
    if on_s < off_s:
        io, ie = int(on_s*FS), int(off_s*FS)
        av = vel_koro[io:ie]; ns = min(ie+int(2*FS), len(vel_koro)-len(av))
        nv = vel_koro[max(0,ns):max(0,ns)+len(av)]; nps = min(1024, len(av))
        f_a, p_a = signal.welch(av, fs=FS, nperseg=nps)
        f_n, p_n = signal.welch(nv, fs=FS, nperseg=nps)
        km = (f_a>=10)&(f_a<=49)
        if np.any(km) and np.mean(p_n[km])>0: snr_db = 10*np.log10(np.mean(p_a[km])/np.mean(p_n[km]))
    else: f_a=f_n=p_a=p_n=np.zeros(513)

    # Print summary
    co_hz = carrier_offset * FS / (2*np.pi)
    print(f"  Carrier offset      : {co_hz:.1f} Hz")
    print(f"  lambda              : {LAMBDA_MM:.1f} mm")
    print(f"  Scale (mm/rad)      : {SCALE:.2f}")
    print(f"  Phase HR max        : {np.max(np.abs(phase_hr)):.4f} rad")
    print(f"  Disp HR max         : {np.max(np.abs(disp_hr)):.4f} mm")
    print(f"  Phase Koro max      : {np.max(np.abs(phase_koro)):.4f} rad")
    print(f"  Disp Koro max       : {np.max(np.abs(disp_koro)):.4f} mm")
    print(f"  Vel Koro max        : {np.max(np.abs(vel_koro)):.2f} mm/s")

    # Calculate Y-axis zoom limits based on the Korotkoff window (to ignore massive motion artifacts)
    if on_s < off_s:
        io, ie = int(on_s*FS), int(off_s*FS)
        ph_hr_lim = max(0.1, np.percentile(np.abs(phase_hr[io:ie]), 99.5) * 1.5)
        ph_koro_lim = max(0.1, np.percentile(np.abs(phase_koro[io:ie]), 99.5) * 1.5)
        vel_koro_lim = max(100, np.percentile(np.abs(vel_koro[io:ie]), 99.5) * 1.5)
    else:
        ph_hr_lim, ph_koro_lim, vel_koro_lim = 1.0, 1.0, 1000
    disp_hr_lim = ph_hr_lim * SCALE
    disp_koro_lim = ph_koro_lim * SCALE

    # ==================== PLOTTING: 3x2 ====================
    fig, axes = plt.subplots(8, 2, figsize=(22, 40))
    plt.subplots_adjust(hspace=0.55)
    yw = dict(color='yellow', alpha=0.2)

    # R1: Magnitude overview | Bandpassed Phase overlay
    ax = axes[0,0]
    ax.plot(time, mag, 'b', alpha=0.6, label='Magnitude (50 Hz notched)')
    if on_s<off_s:
        ax.axvspan(on_s, off_s, **yw, label=f'Koro Window ({dur:.1f}s)')
        ax.axvline(on_s, color='red', ls='--', lw=2, label='SYS')
        ax.axvline(off_s, color='blue', ls='--', lw=2, label='DIA')
    # Y-axis: use middle 90% of data (exclude initial transient spike)
    mid_mag = mag[int(T_SKIP*FS):int((time[-1]-T_SKIP)*FS)]
    ym = np.percentile(mid_mag, [1, 99])
    ax.set_ylim(max(0, ym[0] - 0.2*(ym[1]-ym[0])), ym[1] + 0.3*(ym[1]-ym[0]))
    ax.set_title('1. Magnitude Overview (50 Hz notched)')
    ax.set_ylabel('Amplitude (a.u.)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[0,1]
    ax.plot(time, phase_koro, 'darkgreen', alpha=0.8, label='Phase Koro (10-49 Hz)')
    ax.plot(time, phase_hr, 'green', alpha=0.4, label='Phase HR (0.5-3 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw)
    ax.set_title(f'1. Bandpassed Phase (carrier {co_hz:.0f} Hz removed, then BPF)')
    ax.set_ylabel('Phase (rad)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.set_ylim(-max(ph_hr_lim, ph_koro_lim), max(ph_hr_lim, ph_koro_lim))

    ax = axes[1,0]
    ax.plot(time, disp_hr, 'red', label='Displacement HR (0.5-3 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw)
    ax.set_title(f'2a. Phase->Disp: d=phase x {SCALE:.1f} mm/rad (HR)')
    ax.set_ylabel('Displacement (mm)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.set_ylim(-disp_hr_lim, disp_hr_lim)
    ax.text(0.02, 0.02, f"d = phase x lambda/(4pi) = phase x {SCALE:.2f}", transform=ax.transAxes, fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    ax = axes[1,1]
    ax.plot(time, disp_koro, 'darkred', label='Displacement Koro (10-49 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw)
    ax.set_title(f'2b. Phase->Disp: d=phase x {SCALE:.1f} mm/rad (Koro)')
    ax.set_ylabel('Displacement (mm)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.set_ylim(-disp_koro_lim, disp_koro_lim)

    # R4: Velocity (mm/s)
    ax = axes[2,0]
    vel_hr_n = vel_hr / (np.max(np.abs(vel_hr))+1e-9)
    ax.plot(time, vel_hr_n, 'blue', alpha=0.7, label='Velocity HR (0.5-3 Hz)')
    try: ax.plot(time[peaks], vel_hr_n[peaks], 'ro', ms=5, label=f'Beats ({len(peaks)})')
    except: pass
    ax.set_title('3a. Velocity HR (0.5-3 Hz) - Normalized')
    ax.set_ylabel('Normalized'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7); ax.set_ylim(-1.1,1.1)

    ax = axes[2,1]
    ax.plot(time, vel_koro, 'darkred', label='Velocity Koro (10-49 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('3b. Velocity Koro (10-49 Hz)')
    ax.set_ylabel('Velocity (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    ax.set_ylim(-vel_koro_lim, vel_koro_lim)

    # R5: STFT HR | STFT Koro
    f1, t1, Z1 = signal.stft(disp_hr, fs=FS, nperseg=4096, noverlap=3840)
    P1 = 10*np.log10(np.abs(Z1)**2+1e-20); m1=f1<=5.0
    v1a,v1b = np.percentile(P1[m1],[30,99.5])
    ax = axes[3,0]
    im1 = ax.pcolormesh(t1, f1[m1], P1[m1], shading='gouraud', cmap='viridis', vmin=v1a, vmax=v1b)
    if on_s<off_s: ax.axvline(on_s,color='w',ls='--'); ax.axvline(off_s,color='w',ls='--')
    ax.set_ylim(0,5); ax.set_title('4a. STFT: HR Band (0.5-3 Hz)')
    ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im1, ax=ax, label='dB')

    f2, t2, Z2 = signal.stft(disp_koro, fs=FS, nperseg=4096, noverlap=3840)
    P2 = 10*np.log10(np.abs(Z2)**2+1e-20); m2=(f2>=8)&(f2<=60)
    v2a,v2b = np.percentile(P2[m2],[30,99.5])
    ax = axes[3,1]
    im2 = ax.pcolormesh(t2, f2[m2], P2[m2], shading='gouraud', cmap='plasma', vmin=v2a, vmax=v2b)
    if on_s<off_s: ax.axvline(on_s,color='w',ls='--'); ax.axvline(off_s,color='w',ls='--')
    ax.set_ylim(8,60); ax.set_title('4b. STFT: Korotkoff Band (8-60 Hz)')
    ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)'); plt.colorbar(im2, ax=ax, label='dB')

    # R6: Kurtosis
    ax = axes[4,0]
    ax.plot(time, sliding_kurtosis(hr_mag, int(FS*0.5)), 'purple')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('5a. Magnitude Kurtosis'); ax.set_ylabel('Kurtosis'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[4,1]
    ax.plot(time, sliding_kurtosis(vel_koro, int(FS*0.1)), 'purple')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('5b. Velocity Kurtosis'); ax.set_ylabel('Kurtosis'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R7: Energy
    ax = axes[5,0]
    ax.plot(time, sm_energy, 'teal', label='Energy Envelope'); ax.axhline(eth, color='r', ls='--', label='Threshold')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6a. Korotkoff Energy Envelope'); ax.set_ylabel('Energy ((mm/s)^2)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[5,1]
    ax.plot(time, sliding_rms(vel_koro, int(FS*0.1)), 'darkred', label='Velocity Jitter')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('6b. Velocity Jitter (RMS)'); ax.set_ylabel('RMS (mm/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R8: TKEO + Velocity change
    ax = axes[6,0]
    ax.plot(time, vel_tkeo, 'teal', label='TKEO')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('7a. TKEO Energy'); ax.set_ylabel('((mm/s)^2)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    ax = axes[6,1]
    ax.plot(time, np.abs(np.diff(np.append(vel_koro, 0)))*FS, 'm', label='|d/dt(Vel Env)|')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_title('7b. Velocity Change Rate'); ax.set_ylabel('(mm/s)/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)

    # R11: Normalized Overlays (Full File)
    # Normalize based on the valid middle section to avoid squashing from end-artifacts
    valid_z = slice(int(10*FS), int((time[-1]-10)*FS))
    zhr = hr_mag / (np.max(np.abs(hr_mag[valid_z])) + 1e-9)
    zvl = vel_koro / (np.max(np.abs(vel_koro[valid_z])) + 1e-9)
    zvc = np.abs(np.diff(np.append(vel_koro, 0)))*FS
    zvc = zvc / (np.max(np.abs(zvc[valid_z])) + 1e-9)

    ax = axes[7,0]
    ax.plot(time, zhr, 'k', lw=2, label='Heartbeat (0.5-3 Hz)')
    ax.plot(time, zvl, 'r', alpha=0.7, label='Koro Snaps (10-49 Hz)')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('8a. Full Overlay: Heart vs Koro'); ax.set_ylabel('Normalized'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    
    ax = axes[7,1]
    ax.plot(time, zhr, 'k', lw=2, label='Heartbeat (0.5-3 Hz)')
    ax.plot(time, zvc, 'm', alpha=0.8, label='Vel Change Rate')
    if on_s<off_s: ax.axvspan(on_s, off_s, **yw, label='Koro Window')
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('8b. Full Overlay: Heart vs Vel Transitions'); ax.set_ylabel('Normalized'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7)
    plt.suptitle("Korotkoff Dashboard v5c: Clean Phase + Derivation Chain + Dual-Band STFT", fontsize=20, fontweight='bold', y=0.93)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {OUTPUT_IMG}")
    print(f"{'='*55}")
    print(f"  Koro Window  : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)")
    print(f"  Time HR      : {hr_bpm_t:.1f} BPM ({len(peaks)} beats)")
    print(f"  Freq HR      : {hr_bpm_f:.1f} BPM")
    print(f"  Koro SNR     : {snr_db:.1f} dB")
    print(f"{'='*55}")

if __name__ == '__main__': run()
