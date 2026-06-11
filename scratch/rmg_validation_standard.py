import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_chest_new.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\rmg_validation_normalized.png'

def normalize(x):
    """Normalize signal to range [-1, 1] for visual comparison."""
    x_centered = x - np.mean(x)
    max_val = np.max(np.abs(x_centered))
    return x_centered / max_val if max_val > 0 else x_centered

def run_rmg_normalized():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    i_raw, q_raw = data[0, :], data[1, :]
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    time = np.arange(len(i_raw)) / fs
    
    # 1. RMG DUAL EXTRACTION
    ncs_ph = np.unwrap(np.angle(iq))
    ncs_am = np.abs(iq)
    
    # 2. FILTERING FOR VITALS
    # Breathing (0.1 - 0.5 Hz)
    sos_br = signal.butter(4, [0.1, 0.5], btype='band', fs=fs, output='sos')
    br_wave = signal.sosfiltfilt(sos_br, ncs_ph)
    
    # Heart Rate (0.8 - 3.0 Hz)
    sos_hr = signal.butter(4, [0.8, 3.0], btype='band', fs=fs, output='sos')
    hr_wave = signal.sosfiltfilt(sos_hr, ncs_ph)
    
    # 3. NORMALIZATION (The Requested Method)
    ncs_am_norm = normalize(ncs_am)
    ncs_ph_norm = normalize(ncs_ph)
    br_norm = normalize(br_wave)
    hr_norm = normalize(hr_wave)

    # PLOTTING
    fig = plt.figure(figsize=(18, 24))
    
    # Row 1: NCS_am (Magnitude)
    plt.subplot(4, 1, 1)
    plt.plot(time, ncs_am_norm, color='green')
    plt.title('RMG Metric: NCS_am (Normalized Magnitude)'); plt.ylabel('Norm (a.u.)'); plt.grid(True)
    
    # Row 2: NCS_ph (Phase)
    plt.subplot(4, 1, 2)
    plt.plot(time, ncs_ph_norm, color='red')
    plt.title('RMG Metric: NCS_ph (Normalized Phase)'); plt.ylabel('Norm (a.u.)'); plt.grid(True)
    
    # Row 3: Filtered Breathing
    plt.subplot(4, 1, 3)
    plt.plot(time, br_norm, color='royalblue')
    plt.title('Filtered Respiration (Normalized)'); plt.ylabel('Norm (a.u.)'); plt.grid(True)
    
    # Row 4: Filtered Heartbeat
    plt.subplot(4, 1, 4)
    plt.plot(time, hr_norm, color='firebrick')
    plt.title('Filtered Cardiac Pulse (Normalized)'); plt.ylabel('Norm (a.u.)'); plt.grid(True)
    
    plt.suptitle(f'Normalized RMG Validation: {os.path.basename(file_path)}\nScaling: [-1 to +1] for all panels', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Normalized validation plot saved to: {output_img}")

if __name__ == '__main__':
    run_rmg_normalized()
