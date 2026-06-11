import sys
import os
import numpy as np
sys.path.append(r'd:\Bioview\My_RF_work_v1\scratch')

# Dynamically patch paths in koro_dual_modality_validation and test
import koro_dual_modality_validation as kdm

pairs = [
    {
        'name': 'Pair 0 (rec_koro_sthe.h5 / korotoff_audio_stethoscope.wav)',
        'rf': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5',
        'aud': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.wav'
    },
    {
        'name': 'Pair 1 (rec_koro_sthe_1.h5 / korotoff_audio_stethoscope1.mp4)',
        'rf': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe_1.h5',
        'aud': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope1.mp4'
    },
    {
        'name': 'Pair 2 (rec_koro_sthe_2.h5 / korotoff_audio_stethoscope2.mp4)',
        'rf': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe_2.h5',
        'aud': r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope2.mp4'
    }
]

for p in pairs:
    print("\n" + "="*50)
    print(f"EVALUATING: {p['name']}")
    print("="*50)
    try:
        kdm.RF_PATH = p['rf']
        kdm.AUDIO_PATH = p['aud']
        
        rf = kdm.process_rf()
        st = kdm.process_stethoscope()
        
        # Calculate lag and cross-corr
        fs_cc = 100
        rf_env = kdm.smooth(rf['curves']['Vel RMS'], int(kdm.FS_RF * 1.5))
        rf_env_clip = np.clip(rf_env, 0, np.percentile(rf_env, 95))
        rf_env_n = rf_env_clip / (np.max(rf_env_clip) + 1e-20)
        
        st_env_smooth = kdm.smooth(st['curves']['RMS Power'], int(st['fs'] * 1.5))
        st_env_clip = np.clip(st_env_smooth, 0, np.percentile(st_env_smooth, 95))
        st_env_n = st_env_clip / (np.max(st_env_clip) + 1e-20)
        st_env_rf = np.interp(rf['t'], st['t'], st_env_n)
        
        start_idx, end_idx = int(8 * kdm.FS_RF), int(35 * kdm.FS_RF)
        ds_fac = int(kdm.FS_RF / fs_cc)
        
        rf_cc = rf_env_n[start_idx:end_idx:ds_fac]
        st_cc = st_env_rf[start_idx:end_idx:ds_fac]
        
        cc = np.correlate(rf_cc, st_cc, mode='full')
        lags = np.arange(len(cc)) - len(rf_cc) + 1
        lag_samples = lags[np.argmax(cc)]
        cc_lag_sec = lag_samples / fs_cc
        cc_peak = np.max(cc) / (np.sqrt(np.sum(rf_cc**2) * np.sum(st_cc**2)) + 1e-20)
        
        rf_mid = (rf['onset'] + rf['offset']) / 2.0
        st_mid = (st['onset'] + st['offset']) / 2.0
        phys_lag = rf_mid - st_mid
        
        # Aligned overlap IoU
        st_on_aligned = st['onset'] + phys_lag
        st_off_aligned = st['offset'] + phys_lag
        
        overlap_start = max(rf['onset'], st_on_aligned)
        overlap_end = min(rf['offset'], st_off_aligned)
        overlap = max(0.0, overlap_end - overlap_start)
        union = max(rf['offset'], st_off_aligned) - min(rf['onset'], st_on_aligned)
        iou = overlap / union if union > 0 else 0.0
        
        print(f"  RF Consensus  : {rf['onset']:.2f}s - {rf['offset']:.2f}s (Dur: {rf['duration']:.2f}s)")
        print(f"  Steth Consensus: {st['onset']:.2f}s - {st['offset']:.2f}s (Dur: {st['duration']:.2f}s)")
        print(f"  Phys Trigger Lag: {phys_lag:.2f}s")
        print(f"  Cross-Corr Peak : {cc_peak:.4f} (at lag {cc_lag_sec:.2f}s)")
        print(f"  Aligned Overlap : IoU = {iou:.4f}")
        print(f"  RF HR PSD Peak  : {rf['hr_psd']:.1f} BPM")
        print(f"  Steth HR PSD    : {st['hr_psd']:.1f} BPM")
    except Exception as e:
        print(f"  Error processing: {e}")
