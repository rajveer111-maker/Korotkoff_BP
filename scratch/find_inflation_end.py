import h5py
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"

def analyze_file(sub_name, rec_num):
    h5_path = os.path.join(BASE, sub_name, f"Rec_{rec_num}.h5")
    if not os.path.exists(h5_path):
        return
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    t = np.arange(len(i_raw)) / 10000.0
    
    # Condition IQ
    ic = i_raw - i_raw.mean()
    qc = q_raw - q_raw.mean()
    iq = ic + 1j*qc
    
    # Lowpass filter the IQ signal to smooth high freq noise
    sos_lp = butter(4, 20.0, btype='low', fs=10000.0, output='sos')
    iq_c = sosfiltfilt(sos_lp, iq)
    
    # Calculate phase derivative (dphi) and its absolute cumulative sum or absolute derivative
    phase = np.angle(iq_c)
    dphi = np.abs(np.diff(np.unwrap(phase)))
    
    # Downsample for faster analysis
    ds = 100
    t_ds = t[:-1:ds]
    dphi_ds = dphi[::ds]
    
    # Let's look at the rolling standard deviation or mean of dphi
    # During inflation, there are rapid phase wraps, meaning dphi is large.
    # When inflation stops (maximum cuff pressure point), dphi becomes very small and stable.
    # Let's find the point where the rolling standard deviation of dphi (over e.g. 1.0s) 
    # drops dramatically, or let's inspect the values!
    w_size = int(1.0 * 10000 / ds) # 1.0 second window
    dphi_std = [np.std(dphi_ds[max(0, i-w_size):i+1]) for i in range(len(dphi_ds))]
    dphi_std = np.array(dphi_std)
    
    # The inflation end is where the high-activity phase ends.
    # Let's find the maximum value of dphi_std in the first 25s, and then find where it drops below a threshold.
    # Specifically, after the peak of inflation activity (which is usually in the first 10-15s),
    # the signal activity drops to a baseline.
    max_idx = np.argmax(dphi_std)
    # Search after the peak activity for the first time it drops below 10% of the peak activity and stays there.
    threshold = 0.05 * dphi_std[max_idx]
    
    t_onset_detected = 20.0 # fallback
    for i in range(max_idx, len(dphi_std)):
        if np.all(dphi_std[i:i+int(2.0*10000/ds)] < threshold):
            t_onset_detected = t_ds[i]
            break
            
    print(f"{sub_name} Rec {rec_num:02d}: Max Activity at {t_ds[max_idx]:.2f}s, Detected Inflation End / Deflation Onset = {t_onset_detected:.2f}s")

print("--- Analyzing Subject 1 (Prof Kan) ---")
for r in range(1, 11):
    analyze_file("Sub_1_Prof_kan", r)

print("\n--- Analyzing Subject 2 (Rajveer) ---")
for r in range(1, 11):
    analyze_file("Sub_2_Rajveer", r)
