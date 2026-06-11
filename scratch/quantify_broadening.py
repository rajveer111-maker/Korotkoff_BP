import h5py
import os
import numpy as np

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

def analyze_peak_bandwidth(filepath, start_t, end_t, name):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:, int(start_t*FS):int(end_t*FS)]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    N = len(iq_raw)
    fft_vals = np.fft.fft(iq_raw)
    fft_freqs = np.fft.fftfreq(N, 1/FS)
    
    fft_vals = np.fft.fftshift(fft_vals)
    fft_freqs = np.fft.fftshift(fft_freqs)
    
    power = np.abs(fft_vals)**2
    # Exclude DC
    non_dc = np.abs(fft_freqs) > 1.0
    freqs_ndc = fft_freqs[non_dc]
    power_ndc = power[non_dc]
    
    # Find peak
    peak_idx = np.argmax(power_ndc)
    peak_freq = freqs_ndc[peak_idx]
    peak_val = power_ndc[peak_idx]
    
    # Normalized PSD
    psd_norm = power / peak_val
    psd_db = 10 * np.log10(psd_norm + 1e-12)
    
    # Find index of peak in the shifted array
    orig_peak_idx = np.where(fft_freqs == peak_freq)[0][0]
    
    # Calculate bandwidth at -30 dB and -40 dB
    # We look for the first crossings on left and right of the peak
    def get_bandwidth(db_thresh):
        # Left crossing
        left_idx = orig_peak_idx
        while left_idx > 0 and psd_db[left_idx] > db_thresh:
            left_idx -= 1
        # Right crossing
        right_idx = orig_peak_idx
        while right_idx < len(fft_freqs)-1 and psd_db[right_idx] > db_thresh:
            right_idx += 1
        
        bw = fft_freqs[right_idx] - fft_freqs[left_idx]
        return bw

    bw_30 = get_bandwidth(-30)
    bw_40 = get_bandwidth(-40)
    return peak_freq, bw_30, bw_40

print(f"{'File':<15}{'Peak Freq (Hz)':<20}{'BW @ -30dB (Hz)':<20}{'BW @ -40dB (Hz)':<20}")
for name, filename, start_t, end_t in all_files:
    f_peak, bw30, bw40 = analyze_peak_bandwidth(os.path.join(ultra_dir, filename), start_t, end_t, name)
    print(f"{name:<15}{f_peak:<20.2f}{bw30:<20.4f}{bw40:<20.4f}")
