import h5py
import os
import numpy as np

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
FS = 10000

all_files = [
    ('Table 1', 'ultra_rftable1.h5', 5, 25),
    ('Body 1', 'ultra_rfbody01.h5', 28, 45),
    ('Table 2', 'ultra_rftable2.h5', 2, 9),
    ('Body 2', 'ultra_rfbody1.h5', 2, 9),
    ('Table 3', 'ultra_rftable3.h5', 2, 9),
    ('Body 3', 'ultra_rfbody2.h5', 2, 9),
    ('Table 4', 'ultra_rftable4.h5', 2, 9),
    ('Body 4', 'ultra_rfbody3.h5', 2, 9),
]

def get_peak_near_freq(freq_target, freqs, power, tolerance=5.0):
    idx = np.abs(np.abs(freqs) - freq_target) <= tolerance
    if not np.any(idx):
        return 0.0, -999.0
    sub_freqs = freqs[idx]
    sub_power = power[idx]
    max_idx = np.argmax(sub_power)
    return sub_freqs[max_idx], 10 * np.log10(sub_power[max_idx])

print(f"{'File':<12}{'1f0 (100.7Hz)':<18}{'2f0 (201.4Hz)':<18}{'3f0 (302.1Hz)':<18}{'4f0 (402.9Hz)':<18}{'5f0 (503.6Hz)':<18}")
for name, filename, t_start, t_end in all_files:
    filepath = os.path.join(ultra_dir, filename)
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    iq_raw = data[0, :] + 1j * data[1, :]
    
    active_sig = iq_raw[int(t_start * FS) : int(t_end * FS)]
    
    # Compute FFT
    N = len(active_sig)
    fft_vals = np.fft.fft(active_sig)
    fft_freqs = np.fft.fftfreq(N, 1/FS)
    power = np.abs(fft_vals)**2
    
    res = []
    for h in [1, 2, 3, 4, 5]:
        target_f = h * 100.714
        f_peak, p_db = get_peak_near_freq(target_f, fft_freqs, power)
        res.append(f"{f_peak:.1f} Hz ({p_db:.1f} dB)")
        
    print(f"{name:<12}{res[0]:<18}{res[1]:<18}{res[2]:<18}{res[3]:<18}{res[4]:<18}")
