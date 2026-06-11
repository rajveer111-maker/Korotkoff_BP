import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
CSV_REPORT = os.path.join(BASE, "Multi_Subject_Summary", 'cross_subject_report.csv')

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    """
    Detect the cuff maximum pressure point (inflation end / deflation onset)
    directly from the raw RF signal.
    
    During inflation, the raw RF signal has extremely high frequency/amplitude 
    fluctuations due to manual/automatic pump vibration and rapid cuff volume expansion.
    Once inflation stops and the cuff reaches its maximum pressure (the start of deflation),
    the signal becomes extremely quiet (occlusion phase baseline).
    
    This function computes the rolling energy or rolling variance of the high-pass filtered
    raw IQ signal to identify when the high-vibration inflation phase ends.
    """
    # 1. Condition IQ and bandpass filter to capture pump vibration frequencies (e.g. 5 - 100 Hz)
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    iq_hp = sosfiltfilt(sos_hp, iq)
    
    # 2. Compute rolling envelope / energy
    energy = np.abs(iq_hp)
    
    # Downsample for computational efficiency (to 100 Hz)
    ds = int(fs / 100)
    t_ds = np.arange(len(i_raw))[::ds] / fs
    energy_ds = energy[::ds]
    
    # 3. Compute rolling mean of energy over 1.0s window to smooth out individual pump strokes
    w_size = 100 # 1.0 second in 100 Hz
    energy_smooth = np.convolve(energy_ds, np.ones(w_size)/w_size, mode='same')
    
    # We restrict our search to the region before the known Korotkoff onset to avoid
    # mixing inflation vibrations with heartbeat pulses.
    max_search_sec = 25.0
    if onset_limit is not None:
        max_search_sec = min(max_search_sec, onset_limit - 1.0)
    
    search_mask = t_ds <= max_search_sec
    if not np.any(search_mask):
        return 20.0
        
    t_search = t_ds[search_mask]
    e_search = energy_smooth[search_mask]
    
    # Find the peak of inflation activity (pump vibrations)
    peak_idx = np.argmax(e_search)
    peak_val = e_search[peak_idx]
    
    # If the peak value is very low, it means there was no inflation recorded in the file 
    # (i.e. recording started after inflation was already completed).
    # In that case, the maximum pressure point is at the very beginning of the recording!
    # Let's check the ratio of maximum energy to the end-of-search energy.
    end_val = np.mean(energy_smooth[int(max_search_sec*100)-50:int(max_search_sec*100)])
    
    # Let's print out some stats for debugging
    # print(f"    Peak energy = {peak_val:.2e} at {t_search[peak_idx]:.2f}s, End energy = {end_val:.2e}")
    
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        # No significant pumping phase detected; recording started post-inflation.
        # The deflation onset is effectively at t = 0.0s!
        return 0.0
        
    # Otherwise, search after the peak activity for the point where energy drops back to baseline
    # We define baseline as the average energy after the inflation ends (e.g. from peak_idx to end)
    # The drop point is when it falls below (baseline + 0.1 * (peak_val - baseline))
    baseline = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    
    t_det = 20.0 # fallback
    for i in range(peak_idx, len(t_search)):
        # Check if it stays below threshold for at least 1.5 seconds
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
            
    return t_det

def main():
    df = pd.read_csv(CSV_REPORT)
    print("=== DYNAMIC CUFF MAXIMUM PRESSURE POINT DETECTION ===")
    
    for idx, row in df.iterrows():
        sub_name = "Sub_1_Prof_kan" if "Prof" in row['subject'] else "Sub_2_Rajveer"
        h5_path = os.path.join(BASE, sub_name, f"Rec_{row['rec']}.h5")
        if not os.path.exists(h5_path):
            continue
            
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        i_raw, q_raw = data[0], data[1]
        
        t_max_p = detect_cuff_max_pressure_point(i_raw, q_raw, onset_limit=row['rf_onset'])
        print(f"  {row['subject']} Rec {row['rec']:02d}: Korotkoff Onset = {row['rf_onset']:.2f}s | Detected Cuff Max Pressure (Deflation Onset) = {t_max_p:.2f}s")

if __name__ == '__main__':
    main()
