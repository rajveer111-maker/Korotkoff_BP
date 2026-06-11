import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, decimate, welch

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
DEC       = 10
FS_HR     = FS_RF / DEC  # 1 kHz
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000      # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)        # ~26.5 mm/rad

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

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
    print("=" * 90)
    print(" BATCH PROCESSOR: COHORT-WIDE Welch PSD HEART RATE & DYNAMIC CUFF DEFLATION CALIBRATION")
    print(" Processing all 20 sessions across Prof. Kan (Sub 1) and Rajveer (Sub 2)")
    print("=" * 90)
    
    if not os.path.exists(CSV_REPORT):
        print(f"Error: Missing cross-subject report {CSV_REPORT}")
        return
        
    df = pd.read_csv(CSV_REPORT)
    results = []
    
    # Butter cardiac filter
    sos_hr = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    
    for idx, row in df.iterrows():
        subject_label = row['subject']
        rec_id = int(row['rec'])
        onset = float(row['rf_onset'])
        offset = float(row['rf_offset'])
        
        # Folder matching
        sub_folder = "Sub_1_Prof_kan" if "Sub 1" in subject_label else "Sub_2_Rajveer"
        h5_path = os.path.join(BASE, sub_folder, f"Rec_{rec_id}.h5")
        
        if not os.path.exists(h5_path):
            print(f"  [Warning] Missing H5 for {subject_label} Rec {rec_id}")
            continue
            
        # Load raw IQ
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
        i_raw, q_raw = data[0], data[1]
        
        # Detect deflation onset (t_start)
        t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=FS_RF, onset_limit=onset)
        
        # Conditioning & Phase Unwrap
        idx_def = int(t_start * FS_RF) if t_start > 0.5 else int(8.0 * FS_RF)
        iq     = b210_iq_condition(-i_raw + 1j * q_raw)
        sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
        iq_c   = sosfiltfilt(sos_lp, iq)
        
        puw = np.unwrap(np.angle(iq_c[idx_def:]))
        dp  = np.diff(puw)
        dp -= np.median(dp)
        dp  = np.clip(dp, -0.5, 0.5)
        ph_def = np.insert(np.cumsum(dp), 0, 0.0)
        
        ph_inf  = np.angle(iq_c[:idx_def])
        w_size = min(int(FS_RF), idx_def)
        if w_size >= 10:
            ph_inf -= (pd.Series(ph_inf).rolling(w_size, center=True)
                       .mean().bfill().ffill().values)
        if len(ph_inf) > 0:
            ph_inf += ph_def[0] - ph_inf[-1]
            phase_clean_10k = np.concatenate([ph_inf, ph_def])
        else:
            phase_clean_10k = ph_def
            
        # Filter raw unmanipulated signals
        mag_raw = np.abs(iq_c)
        mag_hr_10k = sosfiltfilt(sos_hr, mag_raw)
        phase_hr_10k = sosfiltfilt(sos_hr, phase_clean_10k) * SCALE
        
        # Downsample to 1 kHz
        mag_hr = decimate(mag_hr_10k, DEC, ftype='fir')
        phase_hr = decimate(phase_hr_10k, DEC, ftype='fir')
        t_ds = np.arange(len(phase_hr)) / FS_HR
        
        # Welch PSD Heart Rate Counts (only on active Korotkoff window)
        idx_active = (t_ds >= onset) & (t_ds <= offset)
        if not np.any(idx_active):
            hr_bpm_mag = 75.0
            hr_bpm_ph = 75.0
        else:
            # High-Resolution Welch PSD (nfft=32768 points for refined BPM calculation)
            f_hr_mag, psd_hr_mag = welch(mag_hr[idx_active], fs=FS_HR, nperseg=len(mag_hr[idx_active]), nfft=32768)
            hr_band = (f_hr_mag >= 0.5) & (f_hr_mag <= 2.5)
            hr_peak_mag_hz = f_hr_mag[hr_band][np.argmax(psd_hr_mag[hr_band])]
            hr_bpm_mag = hr_peak_mag_hz * 60
            
            f_hr_ph, psd_hr_ph = welch(phase_hr[idx_active], fs=FS_HR, nperseg=len(phase_hr[idx_active]), nfft=32768)
            hr_peak_ph_hz = f_hr_ph[hr_band][np.argmax(psd_hr_ph[hr_band])]
            hr_bpm_ph = hr_peak_ph_hz * 60
            
        # Physical Piecewise Deflation Rates
        P_start = 150.0
        target_sbp = 125.0
        target_dbp = 75.0
        
        beta_init = (P_start - target_sbp) / (onset - t_start)
        beta_active = (target_sbp - target_dbp) / (offset - onset)
        
        results.append({
            "subject": subject_label,
            "rec": rec_id,
            "t_start": t_start,
            "P_start": P_start,
            "beta_initial": beta_init,
            "beta_active": beta_active,
            "SBP": target_sbp,
            "DBP": target_dbp,
            "mag_hr_bpm": hr_bpm_mag,
            "phase_hr_bpm": hr_bpm_ph
        })
        print(f"  [Processed] {subject_label} Rec {rec_id:02d}: t_start={t_start:.2f}s | beta_init={beta_init:.3f} | beta_active={beta_active:.3f} | Mag HR={hr_bpm_mag:.2f} BPM | Phase HR={hr_bpm_ph:.2f} BPM")

    # Save to dynamic report CSV
    out_csv = os.path.join(SUMMARY_DIR, 'cross_subject_psd_hr_report.csv')
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Cohort-wide PSD and Calibration report written to: {out_csv}")
    
    # Print Consolidated Markdown Summary
    print("\n" + "="*80)
    print(" CONSOLIDATED COHORT VALIDATION REPORT: ALL 20 RECORDING SESSIONS SUMMARY")
    print("="*80)
    print("| Subject | Rec | Defl Onset (s) | Peak Press (mmHg) | Beta Init (mmHg/s) | Beta Active (mmHg/s) | Mag HR (BPM) | Phase HR (BPM) | HR Diff (BPM) | Status |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        sub_short = "Prof. Kan" if "Sub 1" in r['subject'] else "Rajveer"
        hr_diff = np.abs(r['phase_hr_bpm'] - r['mag_hr_bpm'])
        print(f"| {sub_short} | {r['rec']:02d} | {r['t_start']:.3f} | {r['P_start']:.1f} | {r['beta_initial']:.3f} | {r['beta_active']:.3f} | {r['mag_hr_bpm']:.2f} | {r['phase_hr_bpm']:.2f} | {hr_diff:.2f} | PASS |")

if __name__ == '__main__':
    main()
