import h5py
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
FS = 10000
FC = -100.714

def lowpass_filter_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def analyze_phase_fluctuations(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t_full = np.arange(len(iq_raw)) / FS
    
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t_full)
    iq_baseband = lowpass_filter_sos(iq_shifted, 15.0, FS, order=2)
    
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    idx_crop = (t_full >= 3.0) & (t_full <= 8.0)
    t_crop = t_full[idx_crop]
    phase_crop = phase[idx_crop]
    
    # Linear detrending (1st-order poly)
    p = np.polyfit(t_crop, phase_crop, 1)
    phase_detrended = phase_crop - np.polyval(p, t_crop)
    
    std_rad = np.std(phase_detrended)
    p2p_rad = np.max(phase_detrended) - np.min(phase_detrended)
    
    # Convert to physical displacement in micrometers
    SCALE_FACTOR = 333333.3 / (4 * np.pi)
    std_um = std_rad * SCALE_FACTOR
    p2p_um = p2p_rad * SCALE_FACTOR
    
    return std_rad, p2p_rad, std_um, p2p_um

std_b_rad, p2p_b_rad, std_b_um, p2p_b_um = analyze_phase_fluctuations(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))
std_t_rad, p2p_t_rad, std_t_um, p2p_t_um = analyze_phase_fluctuations(os.path.join(ultra_dir, 'ultra_rftable2.h5'))

print("Body 2:")
print(f"  Phase RMS: {std_b_rad*1000:.3f} mrad, P2P: {p2p_b_rad*1000:.3f} mrad")
print(f"  Displacement RMS: {std_b_um:.3f} um, P2P: {p2p_b_um:.3f} um")
print("Table 2:")
print(f"  Phase RMS: {std_t_rad*1000:.3f} mrad, P2P: {p2p_t_rad*1000:.3f} mrad")
print(f"  Displacement RMS: {std_t_um:.3f} um, P2P: {p2p_t_um:.3f} um")
