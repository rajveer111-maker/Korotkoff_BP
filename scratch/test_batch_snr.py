import h5py
import numpy as np
import glob
import os
from scipy.signal import butter, sosfiltfilt

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
SUBJECTS = {
    'Sub_1': os.path.join(BASE, 'Sub_1_Prof_kan'),
    'Sub_2': os.path.join(BASE, 'Sub_2_Rajveer'),
}
FS = 10000

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

# Let's test all files
for subj, sdir in SUBJECTS.items():
    print(f"\nSubject: {subj}")
    h5_files = sorted(glob.glob(os.path.join(sdir, 'Rec_*.h5')))
    for h5f in h5_files[:3]: # check first 3 files
        rname = os.path.basename(h5f)
        with h5py.File(h5f, 'r') as f:
            data = f['data'][:]
        i_raw, q_raw = -data[0,:], data[1,:]
        
        xc, yc, R = fit_circle(i_raw, q_raw)
        i_c, q_c = i_raw - xc, q_raw - yc
        iq_c = i_c + 1j * q_c
        
        dphi = np.angle(iq_c[1:] * np.conj(iq_c[:-1]))
        hist, bins = np.histogram(dphi, bins=512)
        co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
        dphi_c = dphi - co
        iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
        clip = max(3 * iqr, 0.01)
        dphi_c = np.clip(dphi_c, -clip, clip)
        phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
        ph = signal.detrend(phase, type='linear') if 'signal' in globals() else phase - np.polyval(np.polyfit(np.arange(len(phase)), phase, 1), np.arange(len(phase)))
        
        t = np.arange(len(ph)) / FS
        SCALE = (299792458.0 / 0.9e9) * 1000 / (4 * np.pi)
        
        # Filter 15-50 Hz
        sos = butter(4, [15, 50], btype='band', fs=FS, output='sos')
        pk = sosfiltfilt(sos, ph)
        vk = np.append(np.diff(pk) * FS, 0) * SCALE
        
        # Adaptive deflation and windows
        k_on, k_off = 24.0, 41.5
        t_start_base = t[-1] - 7.0
        t_end_base = t[-1] - 2.0
        
        rms_k = np.sqrt(np.mean(vk[(t >= k_on) & (t <= k_off)]**2))
        rms_b = np.sqrt(np.mean(vk[(t >= t_start_base) & (t <= t_end_base)]**2))
        print(f"  {rname}: Koro RMS={rms_k:.3f} mm/s, Quiet RMS={rms_b:.3f} mm/s, SNR={rms_k/rms_b:.2f}x")
