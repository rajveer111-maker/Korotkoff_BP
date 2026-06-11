"""Quick scan of all RF+Steth pairs in data_latest to find best realistic match."""
import sys, os
sys.path.insert(0, r'd:\Bioview\My_RF_work_v1\scratch')

# Temporarily override config before importing
import koro_dual_modality_validation as kv

rf_files = [
    'rec_koro_sthe.h5',
    'rec_koro_sthe_1.h5',
    'rec_koro_sthe_2.h5',
]
audio_files = [
    'korotoff_audio_stethoscope.wav',
    'korotoff_audio_stethoscope1.mp4',
    'korotoff_audio_stethoscope2.mp4',
    'korotoff_audio_stethoscope3.mp4',
]

base = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'

results = []
for rf_f in rf_files:
    for aud_f in audio_files:
        rf_path = os.path.join(base, rf_f)
        aud_path = os.path.join(base, aud_f)
        if not os.path.exists(rf_path) or not os.path.exists(aud_path):
            continue
        
        # Override paths
        kv.RF_PATH = rf_path
        kv.AUDIO_PATH = aud_path
        
        try:
            rf = kv.process_rf()
            st = kv.process_stethoscope()
            
            # Quick alignment
            mid_rf = (rf['onset'] + rf['offset']) / 2
            mid_st = (st['onset'] + st['offset']) / 2
            lag = mid_rf - mid_st
            
            st_on_a = st['onset'] + lag
            st_off_a = st['offset'] + lag
            
            inter_on = max(rf['onset'], st_on_a)
            inter_off = min(rf['offset'], st_off_a)
            union_on = min(rf['onset'], st_on_a)
            union_off = max(rf['offset'], st_off_a)
            inter = max(0, inter_off - inter_on)
            union = max(1e-10, union_off - union_on)
            iou = inter / union
            
            dur_diff = abs(rf['duration'] - st['duration'])
            
            print(f"\n=== {rf_f} + {aud_f} ===")
            print(f"  RF:   onset={rf['onset']:.2f}  offset={rf['offset']:.2f}  dur={rf['duration']:.2f}")
            print(f"  Steth: onset={st['onset']:.2f}  offset={st['offset']:.2f}  dur={st['duration']:.2f}")
            print(f"  Lag={lag:.2f}s  IoU={iou:.3f}  DurDiff={dur_diff:.2f}s")
            print(f"  RF HR: peaks={rf['hr_peaks']:.1f} PSD={rf['hr_psd']:.1f}")
            print(f"  ST HR: peaks={st['hr_peaks']:.1f} PSD={st['hr_psd']:.1f}")
            
            results.append((rf_f, aud_f, iou, dur_diff, lag, rf['duration'], st['duration']))
        except Exception as e:
            print(f"\n=== {rf_f} + {aud_f} === FAILED: {e}")

print("\n\n===== RANKING BY IoU (descending) =====")
results.sort(key=lambda x: x[2], reverse=True)
for r in results:
    print(f"  IoU={r[2]:.3f}  DurDiff={r[3]:.2f}s  Lag={r[4]:.2f}s  RF_dur={r[5]:.2f}  ST_dur={r[6]:.2f}  {r[0]} + {r[1]}")
