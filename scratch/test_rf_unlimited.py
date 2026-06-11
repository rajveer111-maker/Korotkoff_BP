import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
FS = 10000

def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w, center=True).mean().fillna(0).values)

def main():
    print("Loading RF signal...")
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0, :], data[1, :]
    t_rf = np.arange(len(i_raw)) / FS
    
    print("Applying Decoupled Phase Reconstruction...")
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
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
    
    # Filter RF Korotkoff
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)
    disp_koro = phase_koro * SCALE * 1000
    vel_koro = np.append(np.diff(disp_koro) * FS / 1000, 0)
    
    # Detect RF Window with UNLIMITED search space (after 20s to end of file)
    ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
    
    # Let's search after 20s deflation onset
    vs = int(22.0 * FS)  # Start search at 22s to bypass the valve transient!
    ve = int(len(sm_energy) - 5 * FS)
    
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    rf_on = t_rf[max(si, int(22.0*FS))]
    rf_off = t_rf[min(ei, int((t_rf[-1]-5)*FS))]
    
    print(f"Dynamically detected RF Window (no 40s cap): {rf_on:.2f}s - {rf_off:.2f}s (dur: {rf_off - rf_on:.2f}s)")

if __name__ == '__main__':
    main()
