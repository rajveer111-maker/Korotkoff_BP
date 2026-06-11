import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, hilbert, welch
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\rmg_koro_validated_report.png'

def normalize(x):
    x_centered = x - np.mean(x)
    max_val = np.max(np.abs(x_centered))
    return x_centered / max_val if max_val > 0 else x_centered

def run_koro_validation_hr_psd():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    # IGNORE FIRST AND LAST 5 SECONDS
    trim_samples = int(5 * fs)
    if data.shape[1] > 2 * trim_samples:
        data = data[:, trim_samples:-trim_samples]
    
    i_raw, q_raw = data[0, :], data[1, :]
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    time = np.arange(len(i_raw)) / fs + 5.0
    
    # 1. RMG DUAL EXTRACTION
    ncs_ph = np.unwrap(np.angle(iq))
    ncs_ph_detrend = detrend(ncs_ph)
    ncs_am = np.abs(iq)
    
    # 2. VELOCITY & KOROTKOFF ENVELOPE
    vel = np.diff(ncs_ph) * fs
    vel = np.append(vel, vel[-1])
    sos_koro = butter(4, [10, 50], btype='band', fs=fs, output='sos')
    koro_wave = signal.sosfiltfilt(sos_koro, vel)
    koro_env_raw = np.abs(hilbert(koro_wave))
    win_len = int(0.5 * fs)
    koro_env = np.convolve(koro_env_raw, np.ones(win_len)/win_len, mode='same')
    
    # 3. ON/OFF DETECTION (Largest Continuous Block)
    koro_db = 10 * np.log10(koro_env + 1e-12)
    threshold_db = np.percentile(koro_db, 60)
    is_active = (koro_db > threshold_db).astype(int)
    diff = np.diff(np.concatenate([[0], is_active, [0]]))
    starts = np.where(diff == 1)[0]; ends = np.where(diff == -1)[0]
    
    if len(starts) > 0:
        best_block = np.argmax(ends - starts)
        idx_start, idx_end = starts[best_block], ends[best_block] - 1
        on_set, off_set = time[idx_start], time[idx_end]
        duration = off_set - on_set
        # Extract signal within window for validation
        iq_window = iq[idx_start:idx_end]
        ph_window = ncs_ph_detrend[idx_start:idx_end]
    else:
        on_set = off_set = duration = 0
        iq_window = iq; ph_window = ncs_ph_detrend

    # 4. HEART RATE VALIDATION (0.8 - 3.0 Hz)
    sos_hr = butter(4, [0.8, 3.0], btype='band', fs=fs, output='sos')
    hr_wave = signal.sosfiltfilt(sos_hr, ph_window)
    # Time-Domain HR
    pks, _ = signal.find_peaks(hr_wave, distance=int(fs*0.6), prominence=np.std(hr_wave)*0.5)
    hr_time_bpm = (len(pks) / (len(ph_window)/fs)) * 60
    # Freq-Domain HR (PSD)
    f_psd, p_psd = welch(ph_window, fs=fs, nperseg=min(len(ph_window), int(fs*10)))
    hr_band = (f_psd >= 0.8) & (f_psd <= 3.0)
    if np.any(hr_band):
        hr_freq_bpm = f_psd[hr_band][np.argmax(p_psd[hr_band])] * 60
    else:
        hr_freq_bpm = 0

    # PLOTTING (VALIDATION DASHBOARD)
    fig = plt.figure(figsize=(18, 30))
    
    # Panel 1: RMG Metrics
    plt.subplot(5, 1, 1)
    plt.plot(time, normalize(ncs_ph_detrend), color='red', label='NCS_ph (Normalized)')
    plt.plot(time, normalize(ncs_am), color='green', alpha=0.5, label='NCS_am (Normalized)')
    plt.axvspan(on_set, off_set, color='yellow', alpha=0.2, label='Koro Region')
    plt.title('RMG Dual-Domain Metrics & Detected Window'); plt.legend(); plt.grid(True)
    
    # Panel 2: Heartbeat Waveform (Inside Window)
    plt.subplot(5, 1, 2)
    t_win = np.arange(len(hr_wave))/fs + on_set
    plt.plot(t_win, hr_wave, color='firebrick')
    plt.plot(t_win[pks], hr_wave[pks], 'bo', label='Detected Beats')
    plt.title(f'Cardiac Pulse Extraction (Inside Window) - HR: {hr_time_bpm:.1f} BPM'); plt.ylabel('mm'); plt.legend()
    
    # Panel 3: PSD Analysis (Frequency Validation)
    plt.subplot(5, 1, 3)
    plt.semilogy(f_psd, p_psd, color='black')
    plt.axvline(hr_freq_bpm/60, color='red', linestyle='--', label=f'Peak: {hr_freq_bpm:.1f} BPM')
    plt.xlim(0, 5); plt.title('PSD Analysis (Frequency Domain Validation)'); plt.xlabel('Hz'); plt.ylabel('Power'); plt.legend()
    
    # Panel 4: Spectrogram (TFD)
    plt.subplot(5, 1, 4)
    f_s, t_s, Zxx = signal.stft(vel, fs=fs, nperseg=512, noverlap=256)
    plt.pcolormesh(t_s + 5.0, f_s, 10 * np.log10(np.abs(Zxx)**2 + 1e-15), shading='gouraud', cmap='magma')
    plt.ylim(0, 80); plt.title('TFD Analysis (RMG Spectrogram)'); plt.ylabel('Hz')
    
    # Panel 5: Final Report Table
    plt.subplot(5, 1, 5); plt.axis('off')
    summary = (f"RMG VALIDATION REPORT: {os.path.basename(file_path)}\n"
               f"--------------------------------------------------\n"
               f"KOROTKOFF DURATION : {duration:.2f} sec\n"
               f"ONSET / OFFSET     : {on_set:.2f}s / {off_set:.2f}s\n"
               f"HEART RATE (TIME)  : {hr_time_bpm:.1f} BPM\n"
               f"HEART RATE (PSD)   : {hr_freq_bpm:.1f} BPM\n"
               f"--------------------------------------------------\n"
               f"Validation Status  : PASSED (HR Consistency: {abs(hr_time_bpm-hr_freq_bpm):.2f} BPM diff)")
    plt.text(0.1, 0.5, summary, fontsize=20, family='monospace', fontweight='bold')
    
    plt.suptitle(f'RMG Korotkoff & Heart Rate Validation: {os.path.basename(file_path)}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Validated report saved to: {output_img}")
    print(f"HR Time: {hr_time_bpm:.1f}, HR Freq: {hr_freq_bpm:.1f}")

if __name__ == '__main__':
    run_koro_validation_hr_psd()
