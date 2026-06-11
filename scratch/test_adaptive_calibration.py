import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
CSV_REPORT = os.path.join(BASE, "Multi_Subject_Summary", 'cross_subject_report.csv')

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    iq_hp = sosfiltfilt(sos_hp, iq)
    energy = np.abs(iq_hp)
    
    ds = int(fs / 100)
    t_ds = np.arange(len(i_raw))[::ds] / fs
    energy_ds = energy[::ds]
    
    w_size = 100
    energy_smooth = np.convolve(energy_ds, np.ones(w_size)/w_size, mode='same')
    
    max_search_sec = 25.0
    if onset_limit is not None:
        max_search_sec = min(max_search_sec, onset_limit - 1.0)
    
    search_mask = t_ds <= max_search_sec
    if not np.any(search_mask):
        return 8.0
        
    t_search = t_ds[search_mask]
    e_search = energy_smooth[search_mask]
    
    peak_idx = np.argmax(e_search)
    peak_val = e_search[peak_idx]
    
    end_val = np.mean(energy_smooth[max(0, int(max_search_sec*100)-50):int(max_search_sec*100)])
    
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        return 0.0
        
    baseline = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    
    t_det = 8.0
    for i in range(peak_idx, len(t_search)):
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
            
    return t_det

def main():
    df = pd.read_csv(CSV_REPORT)
    print("=== DYNAMIC ADAPTIVE CALIBRATION TEST ===")
    
    target_sbp = 125.0
    target_dbp = 75.0
    
    results = []
    
    for idx, row in df.iterrows():
        sub_name = "Sub_1_Prof_kan" if "Prof" in row['subject'] else "Sub_2_Rajveer"
        h5_path = os.path.join(BASE, sub_name, f"Rec_{row['rec']}.h5")
        if not os.path.exists(h5_path):
            continue
            
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        i_raw, q_raw = data[0], data[1]
        
        onset = row['rf_onset']
        offset = row['rf_offset']
        
        # 1. Detect dynamic t_start
        t_start = detect_cuff_max_pressure_point(i_raw, q_raw, onset_limit=onset)
        
        # 2. Dynamic beta calibration
        beta = (target_sbp - target_dbp) / (offset - onset)
        
        # 3. Dynamic P_start calibration
        P_start = target_sbp + beta * (onset - t_start)
        
        # Verify calibrated values
        cuff_onset = P_start - beta * (onset - t_start)
        cuff_offset = P_start - beta * (offset - t_start)
        
        print(f"  {row['subject']} Rec {row['rec']:02d}:")
        print(f"    Detected t_start = {t_start:.2f}s, Calibrated beta = {beta:.3f} mmHg/s, P_start = {P_start:.1f} mmHg")
        print(f"    Cuff pressure at onset: {cuff_onset:.1f} mmHg, at offset: {cuff_offset:.1f} mmHg")
        
        results.append({
            'subject': row['subject'],
            'rec': row['rec'],
            't_start': t_start,
            'beta': beta,
            'P_start': P_start
        })

if __name__ == '__main__':
    main()
