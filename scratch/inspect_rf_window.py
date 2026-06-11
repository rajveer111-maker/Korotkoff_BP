import h5py
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
FS = 10000

def sliding_rms(x, w): 
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w, center=True).mean().fillna(0).values)

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
    
    # Boundary detection
    ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
    
    T_SKIP = 5
    vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS))
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    rf_on = t_rf[max(si, int(T_SKIP*FS))]
    rf_off = t_rf[min(ei, int((t_rf[-1]-T_SKIP)*FS))]
    
    rf_dur_raw = rf_off - rf_on
    print("=" * 60)
    print("RF DETECTED WINDOW ANALYSIS")
    print("=" * 60)
    print(f"vs={vs/FS:.2f}s, ve={ve/FS:.2f}s")
    print(f"Max energy index (ci) at: {ci/FS:.2f}s")
    print(f"Raw RF on: {rf_on:.2f}s, off: {rf_off:.2f}s, dur: {rf_dur_raw:.2f}s")
    
    # Normalized durations
    if rf_dur_raw < 4.0:
        p = (4.0-rf_dur_raw)/2; rf_on, rf_off = max(0, rf_on-p), min(t_rf[-1], rf_off+p)
    elif rf_dur_raw > 15.0:
        p = (rf_dur_raw-15.0)/2; rf_on, rf_off = rf_on+p, rf_off-p
    print(f"Post-processed RF on: {rf_on:.2f}s, off: {rf_off:.2f}s, dur: {rf_off - rf_on:.2f}s")

if __name__ == '__main__':
    main()
