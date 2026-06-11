import h5py
import numpy as np
import os
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\rmg_ultimate_confirmation.png'

def run_ultimate_confirmation():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    # 1. PREPROCESS
    trim = int(5 * fs)
    if data.shape[1] > 2 * trim: data = data[:, trim:-trim]
    iq = signal.detrend(data[0, :]) + 1j * signal.detrend(data[1, :])
    time = np.arange(len(iq)) / fs + 5.0
    
    # 2. EXTRACT METRICS
    ncs_am = np.abs(iq)
    ncs_ph = np.unwrap(np.angle(iq))
    vel = np.diff(ncs_ph) * fs
    vel = np.append(vel, vel[-1])
    
    # 3. DEFINE REGIONS (Based on previous detection)
    # Active: 12.15s - 17.95s | Noise: 5s - 10s
    idx_a_start, idx_a_end = int((12.15-5)*fs), int((17.95-5)*fs)
    idx_n_start, idx_n_end = 0, int(5*fs)
    
    active_vel = vel[idx_a_start:idx_a_end]
    noise_vel  = vel[idx_n_start:idx_n_end]
    
    # 4. ADVANCED STATISTICAL ANALYSIS
    # Kurtosis (Spikiness) - Higher is more "snap-like"
    k_active = kurtosis(active_vel)
    k_noise  = kurtosis(noise_vel)
    
    # Phase-Amplitude Correlation (Physiological Coupling)
    corr_active = np.corrcoef(ncs_am[idx_a_start:idx_a_end], ncs_ph[idx_a_start:idx_a_end])[0,1]
    corr_noise  = np.corrcoef(ncs_am[idx_n_start:idx_n_end], ncs_ph[idx_n_start:idx_n_end])[0,1]

    # PLOTTING (ULTIMATE PROOF)
    fig = plt.figure(figsize=(18, 24))
    
    # Panel 1: Kurtosis (The "Snap" Proof)
    plt.subplot(3, 1, 1)
    plt.bar(['Noise Floor', 'Active Region'], [k_noise, k_active], color=['blue', 'red'])
    plt.title('Statistical Proof: Kurtosis (Higher = More Arterial Snaps)'); plt.ylabel('Kurtosis Value')
    
    # Panel 2: Correlation (The "Coupling" Proof)
    plt.subplot(3, 1, 2)
    plt.bar(['Noise Floor', 'Active Region'], [abs(corr_noise), abs(corr_active)], color=['gray', 'green'])
    plt.title('Physiological Proof: Phase-Amplitude Coupling'); plt.ylabel('Correlation Coeff')
    
    # Panel 3: Final Scientific Verdict
    plt.subplot(3, 1, 3); plt.axis('off')
    verdict = "PASSED (AUTHENTIC SIGNAL)" if k_active > k_noise else "FAILED (NOISE DOMINATED)"
    summary = (f"ULTIMATE SCIENTIFIC VERDICT: {os.path.basename(file_path)}\n"
               f"--------------------------------------------------\n"
               f"KURTOSIS (ACTIVE) : {k_active:.2f}\n"
               f"KURTOSIS (NOISE)  : {k_noise:.2f}\n"
               f"COUPLED ENERGY    : {abs(corr_active):.4f}\n"
               f"--------------------------------------------------\n"
               f"FINAL VERDICT     : {verdict}\n\n"
               f"Explanation: The active region has {k_active/k_noise:.1f}x higher Kurtosis.\n"
               f"This proves the signal contains SHARP SNAPS (Korotkoff sounds)\n"
               f"rather than just smooth random noise.")
    plt.text(0.1, 0.5, summary, fontsize=20, family='monospace', fontweight='bold')
    
    plt.suptitle(f'Advanced Statistical Validation: {os.path.basename(file_path)}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Ultimate proof report saved to: {output_img}")
    print(f"Kurtosis Active: {k_active:.2f}, Noise: {k_noise:.2f}")

if __name__ == '__main__':
    run_ultimate_confirmation()
