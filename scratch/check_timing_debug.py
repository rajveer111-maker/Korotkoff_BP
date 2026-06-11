import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

SUBJECT_CONFIGS = [
    {
        "name": "Prof_Kan",
        "label": "Prof. Kan (Sub 1)",
        "best_rec": 6,
    },
    {
        "name": "Rajveer",
        "label": "Rajveer (Sub 2)",
        "best_rec": 4,
    }
]

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

df = pd.read_csv(CSV_REPORT)
for sub in SUBJECT_CONFIGS:
    match = df[(df['subject'] == sub['label']) & (df['rec'] == sub['best_rec'])]
    onset = float(match.iloc[0]['rf_onset'])
    offset = float(match.iloc[0]['rf_offset'])
    
    h5_path = os.path.join(sub['folder'] if 'folder' in sub else os.path.join(BASE, sub['name'] if sub['name']=='Rajveer' else "Sub_1_Prof_kan"), f"Rec_{sub['best_rec']}.h5")
    if not os.path.exists(h5_path):
        # try another folder naming
        h5_path = os.path.join(BASE, "Sub_2_Rajveer" if sub['name']=='Rajveer' else "Sub_1_Prof_kan", f"Rec_{sub['best_rec']}.h5")
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=onset)
    t_shift = 20.0 - t_start
    print(f"Subject: {sub['label']}, best_rec: {sub['best_rec']}, onset: {onset}, offset: {offset}, t_start: {t_start}, t_shift: {t_shift}")
