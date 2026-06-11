import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, find_peaks

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
FS = 10000

def main():
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
    ir, qr = data[0, :], data[1, :]
    t_rf = np.arange(len(ir)) / FS
    
    i_c, q_c = ir - np.mean(ir), qr - np.mean(qr)
    iq = i_c + 1j * q_c
    
    sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    
    idx_deflation = int(20.0 * FS)
    phase_unwrap_def = np.unwrap(np.angle(iq_clean[idx_deflation:]))
    dphi_def = np.diff(phase_unwrap_def)
    carrier_offset = np.median(dphi_def)
    dphi_clean_def = dphi_def - carrier_offset
    dphi_clean_def = np.clip(dphi_clean_def, -0.5, 0.5)
    phase_clean_def = np.insert(np.cumsum(dphi_clean_def), 0, 0)
    
    t_idx = np.arange(len(phase_clean_def))
    t_norm = (t_idx - np.mean(t_idx)) / (np.max(t_idx) - np.min(t_idx) + 1e-9)
    poly_def = np.polyfit(t_norm, phase_clean_def, 2)
    phase_clean_def = phase_clean_def - np.polyval(poly_def, t_norm)
    
    phase_clean_inf = np.angle(iq_clean[:idx_deflation])
    phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(FS*1.0), center=True).mean().bfill().ffill().values
    shift = phase_clean_def[0] - phase_clean_inf[-1]
    phase_clean_inf = phase_clean_inf + shift
    
    phase_clean = np.zeros(len(iq))
    phase_clean[:idx_deflation] = phase_clean_inf
    phase_clean[idx_deflation:] = phase_clean_def
    
    LAMBDA_MM = (299792458 / 0.9e9) * 1000
    SCALE = LAMBDA_MM / (4 * np.pi)
    
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)
    vel_koro = np.append(np.diff(phase_koro * SCALE * 1000) * FS / 1000, 0)
    
    # Let's check the peaks in vel_koro from 20s to 50s
    # Smooth RF energy
    ph_energy = pd.Series(vel_koro).pow(2).rolling(window=int(FS*0.3), center=True).mean().fillna(0).values
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2.0), center=True).mean().fillna(0).values
    
    print("=" * 60)
    print("RF VELOCITY & ENERGY PROFILING")
    print("=" * 60)
    print("Time (s) | Energy Envelope Value")
    print("-" * 35)
    for t_val in np.arange(20.0, 50.0, 2.0):
        idx = int(t_val * FS)
        if idx < len(sm_energy):
            print(f"  {t_val:5.1f}s | {sm_energy[idx]:.4f}")
            
    # Find peaks in the smoothed energy envelope to see where the energy bursts are!
    peaks_e, props_e = find_peaks(sm_energy, distance=int(FS*3.0), prominence=np.max(sm_energy)*0.05)
    print("\nPeaks in RF Energy Envelope:")
    for p in peaks_e:
        print(f"  Time = {t_rf[p]:.2f}s, Envelope = {sm_energy[p]:.4f}")

if __name__ == '__main__':
    main()
