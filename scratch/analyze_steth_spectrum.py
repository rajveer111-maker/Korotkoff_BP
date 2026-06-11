import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from koro_parallel_features import load_stethoscope

audio_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.mp4'

def main():
    t_aud, aud_raw, _, fs_aud = load_stethoscope(audio_path)
    if t_aud is None:
        print("Audio not found")
        return
    
    # Define segments
    # 1. True Korotkoff window (20s to 25s)
    idx_koro = (t_aud >= 20.0) & (t_aud <= 25.0)
    # 2. Cuff deflation noise window (30s to 45s)
    idx_cuff = (t_aud >= 30.0) & (t_aud <= 45.0)
    
    sig_koro = aud_raw[idx_koro]
    sig_cuff = aud_raw[idx_cuff]
    
    f_koro, p_koro = welch(sig_koro, fs=fs_aud, nperseg=4096)
    f_cuff, p_cuff = welch(sig_cuff, fs=fs_aud, nperseg=4096)
    
    # Print power in different bands
    bands = [[20, 200], [40, 150], [50, 150], [80, 200], [100, 250], [100, 500]]
    print("Band Power Comparison (Koro vs Cuff Noise):")
    for b in bands:
        m_k = (f_koro >= b[0]) & (f_koro <= b[1])
        m_c = (f_cuff >= b[0]) & (f_cuff <= b[1])
        pow_k = np.sum(p_koro[m_k])
        pow_c = np.sum(p_cuff[m_c])
        ratio = pow_k / (pow_c + 1e-20)
        print(f"Band {b[0]}-{b[1]} Hz: Koro Power = {pow_k:.2e}, Cuff Power = {pow_c:.2e}, Ratio (Koro/Cuff) = {ratio:.2f}")

if __name__ == '__main__':
    main()
