import h5py
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
FS = 10000

all_files = [
    ('Table 1', 'ultra_rftable1.h5', 5.0, 15.0),
    ('Body 1', 'ultra_rfbody01.h5', 28.0, 38.0),
    ('Table 2', 'ultra_rftable2.h5', 2.0, 9.0),
    ('Body 2', 'ultra_rfbody1.h5', 2.0, 9.0),
    ('Table 3', 'ultra_rftable3.h5', 2.0, 8.0),
    ('Body 3', 'ultra_rfbody2.h5', 38.0, 41.0),
    ('Table 4', 'ultra_rftable4.h5', 2.0, 8.0),
    ('Body 4', 'ultra_rfbody3.h5', 2.0, 9.0),
]

def lowpass_filter_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def analyze_modulation(filepath, t_start, t_end):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, int(t_start*FS):int(t_end*FS)]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS + t_start
    N = len(iq_raw)
    
    # --- 1. Find dominant carrier peak ---
    fft_vals = np.fft.fftshift(np.fft.fft(iq_raw))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/FS))
    power = np.abs(fft_vals)**2
    
    # Exclude DC
    non_dc = np.abs(fft_freqs) > 1.0
    freqs_ndc = fft_freqs[non_dc]
    power_ndc = power[non_dc]
    
    peak_idx = np.argmax(power_ndc)
    FC_local = freqs_ndc[peak_idx]
    peak_val = power_ndc[peak_idx]
    
    # --- 2. Calculate Peak Bandwidth at -20 dB ---
    psd_db = 10 * np.log10(power / peak_val + 1e-12)
    orig_peak_idx = np.where(fft_freqs == FC_local)[0][0]
    
    # Left & Right crossings at -20 dB
    left_idx = orig_peak_idx
    while left_idx > 0 and psd_db[left_idx] > -20:
        left_idx -= 1
    right_idx = orig_peak_idx
    while right_idx < len(fft_freqs)-1 and psd_db[right_idx] > -20:
        right_idx += 1
    bw_20 = fft_freqs[right_idx] - fft_freqs[left_idx]
    
    # --- 3. Extract Phase Displacement (DDC) ---
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC_local * t)
    iq_baseband = lowpass_filter_sos(iq_shifted, 10.0, FS)
    
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Detrend phase with 2nd-order poly
    p = np.polyfit(t, phase, 2)
    phase_detrended = phase - np.polyval(p, t)
    disp_um = phase_detrended * 10000
    
    # Standard deviation of phase displacement (measure of micro-motion energy)
    disp_std = np.std(disp_um)
    
    # --- 4. Integrated Power in Heartbeat Band (0.8 - 2.5 Hz) ---
    f_welch, p_welch = welch(disp_um, fs=FS, nperseg=int(N/2))
    hb_band = (f_welch >= 0.8) & (f_welch <= 2.5)
    hb_power = np.trapz(p_welch[hb_band], f_welch[hb_band])
    
    return FC_local, bw_20, disp_std, hb_power

print(f"{'File':<12} | {'Carrier (Hz)':<12} | {'BW @-20dB (Hz)':<15} | {'Disp Std (µm)':<15} | {'HB Band Power (µm²)':<20}")
print("-" * 80)
for name, filename, start_t, end_t in all_files:
    fc, bw20, d_std, hb_p = analyze_modulation(os.path.join(ultra_dir, filename), start_t, end_t)
    print(f"{name:<12} | {fc:<12.2f} | {bw20:<15.4f} | {d_std:<15.2f} | {hb_p:<20.2f}")
