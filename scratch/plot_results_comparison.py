import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, decimate
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\processing_comparison.png'

def run_comparison():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_raw, q_raw = data[0, :], data[1, :]
    time = np.arange(len(i_raw)) / fs
    
    # --- METHOD A: RECOMMENDED (Centering + Unwrapping + Detrending) ---
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq_c = i_c + 1j * q_c
    phase_clean = np.unwrap(np.angle(iq_c))
    phase_clean_detrend = detrend(phase_clean)
    
    # --- METHOD B: BROKEN (No Centering + Unwrapping) ---
    iq_raw = i_raw + 1j * q_raw
    phase_broken = np.unwrap(np.angle(iq_raw))
    
    # PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top Left: Clean Phase
    axes[0, 0].plot(time, phase_clean_detrend, color='blue')
    axes[0, 0].set_title('METHOD A: With IQ Centering (CLEAN)')
    axes[0, 0].set_ylabel('Phase (radians)')
    axes[0, 0].grid(True)
    
    # Top Right: Broken Phase
    axes[0, 1].plot(time, phase_broken, color='red')
    axes[0, 1].set_title('METHOD B: No IQ Centering (BROKEN DRIFT)')
    axes[0, 1].set_ylabel('Phase (radians)')
    axes[0, 1].grid(True)
    
    # Bottom Left: Clean Zoom
    axes[1, 0].plot(time[:10000], phase_clean_detrend[:10000], color='blue')
    axes[1, 0].set_title('Zoom: Heartbeats are visible (0.2 rad peaks)')
    axes[1, 0].grid(True)
    
    # Bottom Right: Broken Zoom
    axes[1, 1].plot(time[:10000], phase_broken[:10000], color='red')
    axes[1, 1].set_title('Zoom: Heartbeats are hidden by giant drift')
    axes[1, 1].grid(True)
    
    plt.suptitle('Why we MUST use IQ Centering\n(Comparison of rec_koro11_1.h5)', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Comparison report saved to: {output_img}")

if __name__ == '__main__':
    run_comparison()
