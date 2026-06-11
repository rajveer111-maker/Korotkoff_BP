import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert
from scipy.io import wavfile

FS_RF = 10_000
FC_HZ = 0.9e9
C_LIGHT = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
CSV_REPORT = os.path.join(BASE, "Multi_Subject_Summary", "cross_subject_report.csv")

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic*qc)
    sp = np.clip(p3/np.sqrt(p1*p2+1e-20), -1, 1)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    i_new = ic
    q_new = (qc - ic*sp/al) / cp
    return i_new + 1j*q_new

def check_subject(sub_name, sub_label, rec_idx, steth_file):
    sub_folder = os.path.join(BASE, sub_name)
    h5_path = os.path.join(sub_folder, f"Rec_{rec_idx}.h5")
    wav_path = os.path.join(sub_folder, steth_file)
    
    df = pd.read_csv(CSV_REPORT)
    match = df[(df['subject'] == sub_label) & (df['rec'] == rec_idx)]
    if match.empty:
        print(f"No match in CSV report for {sub_label} Rec {rec_idx}")
        return
        
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    print(f"\n==========================================")
    print(f"SUBJECT: {sub_label} (Rec {rec_idx:02d})")
    print(f"Active Window: {onset:.2f} s to {offset:.2f} s")
    print(f"==========================================")
    
    # ── 1. EXTRACT ACOUSTIC HEARTBEAT ENVELOPE ──
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_a = np.arange(len(audio)) / fs_a
    
    # Bandpass filter for heart sound harmonics (20 - 150 Hz)
    sos_a = butter(4, [20, 150], btype='band', fs=fs_a, output='sos')
    audio_filt = sosfiltfilt(sos_a, audio)
    
    # Hilbert envelope to extract heart sound loudness
    audio_env = np.abs(hilbert(audio_filt))
    
    # Cardiac bandpass filter [0.4, 3.0] Hz on the acoustic envelope to get acoustic heartbeats!
    sos_ah = butter(4, [0.4, 3.0], btype='band', fs=fs_a, output='sos')
    dh_acoustic = sosfiltfilt(sos_ah, audio_env)
    
    # Detect acoustic heartbeat peaks
    mask_a = (t_a >= onset) & (t_a <= offset)
    dh_a_stable = dh_acoustic[mask_a]
    prom_a = np.std(dh_a_stable) * 0.4
    peaks_a, _ = find_peaks(-dh_acoustic, distance=int(0.5 * fs_a), prominence=prom_a)
    
    beat_amps_a = []
    beat_times_a = []
    half_win_a = int(0.4 * fs_a)
    for pk in peaks_a:
        t_pk = t_a[pk]
        if t_pk < onset or t_pk > offset:
            continue
        w_start = max(0, pk - half_win_a)
        w_end = min(len(audio), pk + half_win_a)
        ptp_amp = np.ptp(dh_acoustic[w_start:w_end])
        beat_amps_a.append(ptp_amp)
        beat_times_a.append(t_pk)
        
    beat_amps_a = np.array(beat_amps_a)
    beat_times_a = np.array(beat_times_a)
    
    # ── 2. EXTRACT RF HEARTBEAT ENVELOPE ──
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    N = len(i_raw)
    t = np.arange(N)/FS_RF
    
    iq = b210_iq_condition(-i_raw + 1j*q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_c = sosfiltfilt(sos_lp, iq)
    idx_def = int(20.0 * FS_RF)
    
    puw = np.unwrap(np.angle(iq_c[idx_def:]))
    dp = np.diff(puw)
    dp -= np.median(dp)
    dp = np.clip(dp, -0.5, 0.5)
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)
    
    ph_inf = np.angle(iq_c[:idx_def])
    ph_inf -= (pd.Series(ph_inf).rolling(int(FS_RF), center=True)
               .mean().bfill().ffill().values)
    ph_inf += ph_def[0] - ph_inf[-1]
    phase_clean = np.concatenate([ph_inf, ph_def])
    
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh_rf = sosfiltfilt(sos_h, phase_clean) * SCALE * 0.1
    
    stable_mask = (t >= onset) & (t <= offset)
    dh_rf_stable = dh_rf[stable_mask]
    prom_rf = np.std(dh_rf_stable) * 0.4
    peaks_rf, _ = find_peaks(-dh_rf, distance=int(0.5 * FS_RF), prominence=prom_rf)
    
    beat_amps_rf = []
    beat_times_rf = []
    half_win_rf = int(0.4 * FS_RF)
    for pk in peaks_rf:
        t_pk = t[pk]
        if t_pk < onset or t_pk > offset:
            continue
        w_start = max(0, pk - half_win_rf)
        w_end = min(N, pk + half_win_rf)
        ptp_amp = np.ptp(dh_rf[w_start:w_end])
        beat_amps_rf.append(ptp_amp)
        beat_times_rf.append(t_pk)
        
    beat_amps_rf = np.array(beat_amps_rf)
    beat_times_rf = np.array(beat_times_rf)
    
    # Find peaks of both
    if len(beat_times_a) > 0:
        # Search middle 70% of the active window to exclude edge deflation-release motion artifacts
        mid_start = onset + 0.15 * (offset - onset)
        mid_end = offset - 0.15 * (offset - onset)
        mid_mask_a = (beat_times_a >= mid_start) & (beat_times_a <= mid_end)
        if np.any(mid_mask_a):
            max_idx_a = np.where(mid_mask_a)[0][np.argmax(beat_amps_a[mid_mask_a])]
        else:
            max_idx_a = np.argmax(beat_amps_a)
        print(f"Acoustic observed peak heartbeat amplitude occurs at: {beat_times_a[max_idx_a]:.3f} s")
        
    if len(beat_times_rf) > 0:
        mid_start = onset + 0.15 * (offset - onset)
        mid_end = offset - 0.15 * (offset - onset)
        mid_mask_rf = (beat_times_rf >= mid_start) & (beat_times_rf <= mid_end)
        if np.any(mid_mask_rf):
            max_idx_rf = np.where(mid_mask_rf)[0][np.argmax(beat_amps_rf[mid_mask_rf])]
        else:
            max_idx_rf = np.argmax(beat_amps_rf)
        print(f"RF observed peak heartbeat amplitude occurs at: {beat_times_rf[max_idx_rf]:.3f} s")
        
    if len(beat_times_a) > 0 and len(beat_times_rf) > 0:
        diff_s = abs(beat_times_a[max_idx_a] - beat_times_rf[max_idx_rf])
        print(f"Difference between Acoustic and RF peak heartbeat compliance times: {diff_s:.3f} s")

def main():
    check_subject("Sub_1_Prof_kan", "Prof. Kan (Sub 1)", 6, "sthethoscope_rec06.wav")
    check_subject("Sub_2_Rajveer", "Rajveer (Sub 2)", 4, "sthethoscope_rec04.wav")

if __name__ == '__main__':
    main()
