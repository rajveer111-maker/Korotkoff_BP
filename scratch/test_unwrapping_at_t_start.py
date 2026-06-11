import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
CSV_REPORT = os.path.join(BASE, "Multi_Subject_Summary", 'cross_subject_report.csv')
FS_RF = 10000
SCALE = (299792458.0 / 0.9e9) * 1000 / (4 * np.pi)

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
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic*qc)
    sp = np.clip(p3/np.sqrt(p1*p2+1e-20), -1, 1)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    i_new = ic
    q_new = (qc - ic*sp/al) / cp
    return i_new + 1j*q_new

def main():
    df = pd.read_csv(CSV_REPORT)
    print("=== TESTING PHASE UNWRAPPING WITH ADAPTIVE T_START ===")
    
    for idx, row in df.iterrows():
        sub_name = "Sub_1_Prof_kan" if "Prof" in row['subject'] else "Sub_2_Rajveer"
        h5_path = os.path.join(BASE, sub_name, f"Rec_{row['rec']}.h5")
        if not os.path.exists(h5_path):
            continue
            
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        i_raw, q_raw = data[0], data[1]
        
        # Detect t_start
        t_start = detect_cuff_max_pressure_point(i_raw, q_raw, onset_limit=row['rf_onset'])
        
        # We need at least some samples after t_start
        if t_start <= 0.5:
            # Fallback to 8.0 if t_start is too early to do splitting
            idx_def = int(8.0 * FS_RF)
        else:
            idx_def = int(t_start * FS_RF)
            
        # Reconstruct phase
        iq = b210_iq_condition(-i_raw + 1j * q_raw)
        sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
        iq_c = sosfiltfilt(sos_lp, iq)
        
        puw = np.unwrap(np.angle(iq_c[idx_def:]))
        dp  = np.diff(puw)
        dp -= np.median(dp)
        dp  = np.clip(dp, -0.5, 0.5)
        ph_def = np.insert(np.cumsum(dp), 0, 0.0)

        ph_inf  = np.angle(iq_c[:idx_def])
        # avoid rolling window failure if idx_def is too small
        w_size = min(int(FS_RF), idx_def)
        if w_size >= 10:
            ph_inf -= (pd.Series(ph_inf).rolling(w_size, center=True)
                       .mean().bfill().ffill().values)
        if len(ph_inf) > 0:
            ph_inf += ph_def[0] - ph_inf[-1]
            phase_clean = np.concatenate([ph_inf, ph_def])
        else:
            phase_clean = ph_def
            
        sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
        dh    = sosfiltfilt(sos_h, phase_clean) * SCALE * 0.1
        
        print(f"  {row['subject']} Rec {row['rec']:02d}: Success! t_start={t_start:.2f}s, dh range = [{np.min(dh):.4f}, {np.max(dh):.4f}] mm")

if __name__ == '__main__':
    main()
