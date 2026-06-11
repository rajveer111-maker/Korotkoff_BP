import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
FS = 10000

all_files = [
    ('Table 1', 'ultra_rftable1.h5'),
    ('Body 1', 'ultra_rfbody01.h5'),
    ('Table 2', 'ultra_rftable2.h5'),
    ('Body 2', 'ultra_rfbody1.h5'),
    ('Table 3', 'ultra_rftable3.h5'),
    ('Body 3', 'ultra_rfbody2.h5'),
    ('Table 4', 'ultra_rftable4.h5'),
    ('Body 4', 'ultra_rfbody3.h5'),
]

def get_top_peak(signal, fs):
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    power = np.abs(fft_vals)**2
    # Exclude near-DC (< 1 Hz)
    non_dc = np.abs(fft_freqs) > 1.0
    if not np.any(non_dc):
        return 0, 0
    freqs_ndc = fft_freqs[non_dc]
    power_ndc = power[non_dc]
    max_idx = np.argmax(power_ndc)
    return freqs_ndc[max_idx], 10 * np.log10(power_ndc[max_idx])

print(f"{'File':<15}{'Active Peak Freq':<20}{'Active Power (dB)':<20}{'Quiet Peak Freq':<20}{'Quiet Power (dB)':<20}")
for name, filename in all_files:
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    iq_raw = data[0, :] + 1j * data[1, :]
    
    # Define segments based on file
    # For Body 1, Active starts at 27s
    if name == 'Body 1':
        active_sig = iq_raw[28 * FS : 45 * FS]
        quiet_sig = iq_raw[2 * FS : 25 * FS]
    elif name == 'Table 1':
        active_sig = iq_raw[5 * FS : 25 * FS]
        quiet_sig = iq_raw[0 * FS : 5 * FS]
    else:
        active_sig = iq_raw[2 * FS : 9 * FS]
        quiet_sig = iq_raw[12 * FS : 30 * FS]
        
    act_f, act_p = get_top_peak(active_sig, FS)
    q_f, q_p = get_top_peak(quiet_sig, FS)
    print(f"{name:<15}{act_f:<20.2f}{act_p:<20.2f}{q_f:<20.2f}{q_p:<20.2f}")
