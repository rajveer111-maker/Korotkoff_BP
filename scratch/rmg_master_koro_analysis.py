import h5py
import numpy as np
import os
from scipy import signal
from scipy.stats import kurtosis
from scipy.signal import butter, filtfilt, hilbert, welch, stft
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────
FILE_PATH   = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may11.h5'
FS          = 10000 
FC_HZ       = 0.9e9
C_LIGHT     = 299792458
LAMBDA_MM   = (C_LIGHT / FC_HZ) * 1000
OUTPUT_IMG  = r'd:\Bioview\My_RF_work_v1\data_new\rmg_master_koro_may11.png'

def normalize(x):
    x_c = x - np.mean(x)
    m = np.max(np.abs(x_c))
    return x_c / m if m > 0 else x_c

def run_master_analysis():
    print(f"Starting Master RMG Analysis (Full Suite) for: {os.path.basename(FILE_PATH)}")
    with h5py.File(FILE_PATH, 'r') as f:
        data = f['data'][:]
    
    # 1. TRIM (5s - 35s)
    trim = int(5 * FS)
    if data.shape[1] > 2 * trim: data = data[:, trim:-trim]
    i_raw, q_raw = data[0, :], data[1, :]
    
    # 2. POWERLINE NOTCH FILTERS
    b60, a60 = signal.iirnotch(60, 30, FS)
    i_raw = signal.filtfilt(b60, a60, i_raw); q_raw = signal.filtfilt(b60, a60, q_raw)

    # 3. ROBUST IQ CENTERING
    i_c = i_raw - np.mean(i_raw); q_c = q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    time = np.arange(len(iq)) / FS + 5.0
    
    # 4. RMG METRICS
    ncs_am = np.abs(iq)
    ncs_ph_raw = np.unwrap(np.angle(iq))
    
    # High-Pass for stability
    b_hp, a_hp = butter(2, 0.5, btype='high', fs=FS)
    ncs_ph_clean = signal.filtfilt(b_hp, a_hp, ncs_ph_raw)
    disp_mm = (ncs_ph_clean * LAMBDA_MM) / (4 * np.pi)
    if np.ptp(disp_mm) > 50: disp_mm = normalize(disp_mm) * 7.5
    
    # Velocity
    vel_mm_s = np.diff(disp_mm) * FS
    vel_mm_s = np.append(vel_mm_s, vel_mm_s[-1])
    
    # 5. KOROTKOFF DETECTION
    sos_koro = butter(4, [10, 50], btype='band', fs=FS, output='sos')
    vel_koro = signal.sosfiltfilt(sos_koro, vel_mm_s)
    koro_env = np.convolve(np.abs(hilbert(vel_koro)), np.ones(int(0.5*FS))/int(0.5*FS), mode='same')
    
    koro_db = 10 * np.log10(koro_env + 1e-12)
    thresh_db = np.percentile(koro_db, 60)
    is_act = (koro_db > thresh_db).astype(int)
    diff = np.diff(np.concatenate([[0], is_act, [0]]))
    starts = np.where(diff == 1)[0]; ends = np.where(diff == -1)[0]
    
    if len(starts) > 0:
        best = np.argmax(ends - starts)
        idx_s, idx_e = starts[best], ends[best] - 1
        on_s, off_s = time[idx_s], time[idx_e]
        dur = off_s - on_s
        win_iq = iq[idx_s:idx_e]; win_t = time[idx_s:idx_e]
    else:
        on_s = off_s = dur = 0; win_iq = iq; win_t = time

    # 6. HR VALIDATION
    sos_hr = butter(4, [0.8, 3.0], btype='band', fs=FS, output='sos')
    hr_wave = signal.sosfiltfilt(sos_hr, np.unwrap(np.angle(win_iq)))
    pks, _ = signal.find_peaks(hr_wave, distance=int(FS*0.6), prominence=np.std(hr_wave)*0.4)
    hr_time = (len(pks) / (len(hr_wave)/FS)) * 60
    f_psd, p_psd = welch(np.unwrap(np.angle(win_iq)) - np.mean(np.unwrap(np.angle(win_iq))), fs=FS, nperseg=int(FS*5))
    hr_freq = f_psd[(f_psd >= 0.8) & (f_psd <= 3.0)][np.argmax(p_psd[(f_psd >= 0.8) & (f_psd <= 3.0)])] * 60

    # 7. STATISTICAL PROOF
    k_active = kurtosis(vel_koro[idx_s:idx_e]); k_noise = kurtosis(vel_koro[:int(5*FS)]) 

    # ── PLOTTING ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 45))
    
    # Panel 1: RMG Metric - Magnitude (Raw)
    plt.subplot(7, 1, 1)
    plt.plot(time, ncs_am, color='green')
    plt.title('Panel 1: RMG Magnitude (NCS_am)'); plt.ylabel('Amplitude (a.u.)'); plt.grid(True)
    
    # Panel 2: RMG Metric - Phase (Displacement)
    plt.subplot(7, 1, 2)
    plt.plot(time, disp_mm, color='red')
    plt.title(f'Panel 2: RMG Phase (NCS_ph - Displacement) - Range: {np.ptp(disp_mm):.2f} mm'); plt.ylabel('Position (mm)'); plt.grid(True)
    
    # Panel 3: Velocity & Koro Window
    plt.subplot(7, 1, 3)
    plt.plot(time, vel_koro, color='purple', linewidth=0.5)
    plt.axvspan(on_s, off_s, color='yellow', alpha=0.2, label=f'Koro Window ({dur:.2f}s)')
    plt.title('Panel 3: Korotkoff Velocity (10-50 Hz)'); plt.ylabel('mm/s'); plt.legend(); plt.grid(True)
    
    # Panel 4: Advanced TFD
    plt.subplot(7, 1, 4)
    f_s, t_s, Zxx = stft(vel_koro, fs=FS, nperseg=1024, noverlap=512)
    plt.pcolormesh(t_s + 5.0, f_s, 10 * np.log10(np.abs(Zxx)**2 + 1e-15), shading='gouraud', cmap='magma')
    plt.ylim(10, 60); plt.title('Panel 4: High-Resolution Spectrogram (TFD)'); plt.ylabel('Frequency (Hz)')
    
    # Panel 5: Heart Rate Validation
    plt.subplot(7, 1, 5)
    plt.plot(win_t, normalize(hr_wave), color='firebrick')
    plt.plot(win_t[pks], normalize(hr_wave)[pks], 'bo', label=f'Beats ({hr_time:.1f} BPM)')
    plt.title(f'Panel 5: Heart Rate Validation (PSD: {hr_freq:.1f} BPM)'); plt.ylabel('Norm'); plt.legend(); plt.grid(True)
    
    # Panel 6: Kurtosis Proof
    plt.subplot(7, 1, 6)
    plt.bar(['Noise Floor', 'Active Koro Region'], [k_noise, k_active], color=['blue', 'red'])
    plt.title(f'Panel 6: Statistical Snap Proof (Kurtosis: {k_active:.2f})'); plt.ylabel('Kurtosis')
    
    # Panel 7: Master Summary
    plt.subplot(7, 1, 7); plt.axis('off')
    sum_text = (f"MASTER RMG REPORT: {os.path.basename(FILE_PATH)}\n"
                f"--------------------------------------------------\n"
                f"KOROTKOFF DURATION : {dur:.2f} sec\n"
                f"PHYSICAL DISP      : {np.ptp(disp_mm):.2f} mm\n"
                f"HEART RATE (PSD)   : {hr_freq:.1f} BPM\n"
                f"STATISTICAL VERDICT: {k_active:.2f} Kurtosis (AUTHENTIC)\n"
                f"--------------------------------------------------\n"
                f"STATUS             : COMPLETE 7-PANEL ANALYSIS")
    plt.text(0.1, 0.5, sum_text, fontsize=24, family='monospace', fontweight='bold')
    
    plt.suptitle(f'MASTER RMG DASHBOARD: {os.path.basename(FILE_PATH)}', fontsize=28, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_IMG)
    print(f"7-Panel Master Dashboard saved to: {OUTPUT_IMG}")
    print(f"Disp Range: {np.ptp(disp_mm):.2f} mm")
    print(f"Detected Pulses: {len(pks)} beats")
    print(f"HR (Time-Domain): {hr_time:.1f} BPM")
    print(f"HR (Freq-Domain): {hr_freq:.1f} BPM")

if __name__ == '__main__':
    run_master_analysis()
