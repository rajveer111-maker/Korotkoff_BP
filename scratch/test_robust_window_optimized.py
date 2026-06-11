import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from koro_parallel_features import load_stethoscope, find_robust_stethoscope_window

audio_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.mp4'

def main():
    print("Loading stethoscope...")
    t_aud, aud_raw, koro_aud, fs_aud = load_stethoscope(audio_path)
    if t_aud is None:
        print("Audio not found")
        return
    print(f"Stethoscope loaded in {t_aud[-1]:.2f}s, fs={fs_aud}")
    
    print("Detecting stethoscope window...")
    steth_on, steth_off = find_robust_stethoscope_window(koro_aud, t_aud, fs_aud)
    print(f"\n[Consensus Stethoscope Window]: {steth_on:.2f}s - {steth_off:.2f}s (duration: {steth_off-steth_on:.2f}s)")

if __name__ == '__main__':
    main()
