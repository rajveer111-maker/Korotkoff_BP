import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k) / k, mode='same')

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

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic * qc)
    sp = np.clip(p3 / np.sqrt(p1 * p2 + 1e-20), -1, 1)
    cp = np.sqrt(max(1 - sp**2, 1e-10))
    al = np.sqrt(p2 / (p1 + 1e-20))
    i_new = ic
    q_new = (qc - ic * sp / al) / cp
    return i_new + 1j * q_new

def find_physiological_points_adaptive(t_ds_phys, env, t_start_phys=20.0):
    mask_def = t_ds_phys >= t_start_phys
    t_def = t_ds_phys[mask_def]
    env_def = env[mask_def]
    idx_map_def = np.argmax(env_def)
    t_map = t_def[idx_map_def]
    env_peak = env_def[idx_map_def]
    
    # SBP: rise to 0.55 * peak
    idx_before = np.where(env_def[:idx_map_def] <= 0.55 * env_peak)[0]
    t_sbp = t_def[idx_before[-1]] if len(idx_before) > 0 else t_start_phys + 10.0
    
    # DBP: fall to 0.70 * peak
    idx_after = np.where(env_def[idx_map_def:] <= 0.70 * env_peak)[0]
    t_dbp = t_def[idx_map_def + idx_after[0]] if len(idx_after) > 0 else t_sbp + 25.0
    return t_sbp, t_map, t_dbp

df = pd.read_csv(CSV_REPORT)
for index, row in df.iterrows():
    sub_label = row['subject']
    rec = int(row['rec'])
    if rec not in [4, 6]: continue # just check best recs
    
    onset = float(row['rf_onset'])
    offset = float(row['rf_offset'])
    
    folder_name = "Sub_1_Prof_kan" if "Sub 1" in sub_label else "Sub_2_Rajveer"
    h5_path = os.path.join(BASE, folder_name, f"Rec_{rec}.h5")
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=onset)
    t_shift = 20.0 - t_start
    t_rf = np.arange(len(i_raw)) / 10000.0
    
    # Preprocess
    idx_def = int(t_start * 10000.0) if t_start > 0.5 else int(8.0 * 10000.0)
    iq     = b210_iq_condition(-i_raw + 1j * q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=10000.0, output='sos')
    iq_c   = sosfiltfilt(sos_lp, iq)
    
    puw = np.unwrap(np.angle(iq_c[idx_def:]))
    dp  = np.diff(puw)
    dp -= np.median(dp)
    dp  = np.clip(dp, -0.5, 0.5)
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)
    ph_inf  = np.angle(iq_c[:idx_def])
    w_size = min(int(10000.0), idx_def)
    if w_size >= 10:
        ph_inf -= (pd.Series(ph_inf).rolling(w_size, center=True)
                   .mean().bfill().ffill().values)
    if len(ph_inf) > 0:
        ph_inf += ph_def[0] - ph_inf[-1]
        phase_clean = np.concatenate([ph_inf, ph_def])
    else:
        phase_clean = ph_def
        
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=10000.0, output='sos')
    dh_rf = sosfiltfilt(sos_h, phase_clean) * (333.1 / (4 * np.pi)) * 0.1
    dh_rf_env = smooth(np.abs(hilbert(dh_rf)), int(1.5 * 10000.0))
    
    n_prepend_rf = int(t_shift * 10000.0)
    t_prepend_rf = np.arange(n_prepend_rf) / 10000.0
    t_rf_phys = np.concatenate([t_prepend_rf, t_rf + t_shift])
    dh_rf_env_full = np.concatenate([np.zeros(n_prepend_rf), dh_rf_env])
    
    t_sbp_rf, t_map_rf, t_dbp_rf = find_physiological_points_adaptive(t_rf_phys[::10], dh_rf_env_full[::10])
    print(f"Subject: {sub_label}, Rec: {rec} -> ADAPTIVE POINTS: SBP={t_sbp_rf:.2f}s, MAP={t_map_rf:.2f}s, DBP={t_dbp_rf:.2f}s")
