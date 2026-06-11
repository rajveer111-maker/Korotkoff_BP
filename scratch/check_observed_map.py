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

SUBJECT_CONFIGS = [
    {
        "name": "Prof_Kan",
        "label": "Prof. Kan (Sub 1)",
        "folder": os.path.join(BASE, "Sub_1_Prof_kan"),
        "best_rec": 6,
        "steth_file": "sthethoscope_rec06.wav"
    },
    {
        "name": "Rajveer",
        "label": "Rajveer (Sub 2)",
        "folder": os.path.join(BASE, "Sub_2_Rajveer"),
        "best_rec": 4,
        "steth_file": "sthethoscope_rec04.wav"
    }
]

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

def check_subject(sub, rec_idx, df_report):
    h5_path = os.path.join(sub['folder'], f"Rec_{rec_idx}.h5")
    wav_path = os.path.join(sub['folder'], sub['steth_file'])
    
    match = df_report[(df_report['subject'] == sub['label']) & (df_report['rec'] == rec_idx)]
    if match.empty:
        print(f"No match in CSV report for {sub['label']} Rec {rec_idx}")
        return
        
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    print(f"\n==========================================")
    print(f"SUBJECT: {sub['label']} (Rec {rec_idx:02d})")
    print(f"Active Korotkoff Window: {onset:.2f} s to {offset:.2f} s (Duration: {offset-onset:.2f} s)")
    print(f"==========================================")
    
    # 1. PROCESS ACOUSTIC SOUNDS
    if os.path.exists(wav_path):
        fs_a, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float64) / 32768.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t_a = np.arange(len(audio)) / fs_a
        
        # Bandpass filter for Korotkoff sound spectrum (e.g. 50-300 Hz where clicks are prominent)
        sos_a = butter(4, [50, 300], btype='band', fs=fs_a, output='sos')
        audio_filt = sosfiltfilt(sos_a, audio)
        
        # Audio envelope
        audio_env = np.abs(hilbert(audio_filt))
        # Smooth with a 1.0s window to get the overall loudness profile
        audio_env_smooth = smooth(audio_env, int(1.0 * fs_a))
        
        # Find peak loudness between onset and offset
        mask_a = (t_a >= onset) & (t_a <= offset)
        t_a_win = t_a[mask_a]
        env_win = audio_env_smooth[mask_a]
        
        t_map_acoustic = t_a_win[np.argmax(env_win)]
        print(f"Acoustic Loudness Envelope Peaks at: {t_map_acoustic:.3f} s")
    else:
        print(f"WAV file not found: {wav_path}")
        t_map_acoustic = None
        
    # 2. PROCESS RF HEARTBEAT
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
    dh = sosfiltfilt(sos_h, phase_clean) * SCALE * 0.1
    
    # Let's find heartbeat peaks
    stable_mask = (t >= onset) & (t <= offset)
    dh_stable = dh[stable_mask]
    prom = np.std(dh_stable) * 0.4
    min_dist = int(0.5 * FS_RF)
    peaks, _ = find_peaks(-dh, distance=min_dist, prominence=prom)
    
    beat_amps = []
    beat_times = []
    half_win = int(0.4 * FS_RF)
    
    for pk in peaks:
        t_pk = t[pk]
        if t_pk < onset or t_pk > offset:
            continue
        
        w_start = max(0, pk - half_win)
        w_end = min(N, pk + half_win)
        dh_win = dh[w_start:w_end]
        ptp_amp = np.ptp(dh_win)
        
        beat_amps.append(ptp_amp)
        beat_times.append(t_pk)
        
    beat_amps = np.array(beat_amps)
    beat_times = np.array(beat_times)
    
    if len(beat_times) > 0:
        max_idx = np.argmax(beat_amps)
        print(f"RF observed peak heartbeat displacement occurs at: {beat_times[max_idx]:.3f} s (Amp = {beat_amps[max_idx]*1000:.1f} um)")
        
        # Let's look at the heartbeats around 35s to 38s
        print("Heartbeats around 35s - 38s:")
        for i in range(len(beat_times)):
            t_b = beat_times[i]
            if 34.0 <= t_b <= 38.0:
                is_max = " * [LOCAL MAX]" if (i > 0 and i < len(beat_times)-1 and beat_amps[i] > beat_amps[i-1] and beat_amps[i] > beat_amps[i+1]) else ""
                print(f"  Beat {i+1:02d}: Time = {t_b:.3f} s, Amp = {beat_amps[i]*1000:.1f} um {is_max}")

def main():
    if not os.path.exists(CSV_REPORT):
        print(f"ERROR: missing report {CSV_REPORT}")
        return
    df = pd.read_csv(CSV_REPORT)
    for sub in SUBJECT_CONFIGS:
        check_subject(sub, sub['best_rec'], df)

if __name__ == '__main__':
    main()
