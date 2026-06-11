import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, decimate
import os

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000
DEC = 10
FS = FS_RF // DEC

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def env_smooth(x, w_sec, fs):
    k = max(1, int(w_sec * fs))
    return np.convolve(np.abs(hilbert(x)), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def get_map(h5_file, name):
    rf_path = os.path.join(BASE, h5_file)
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0, :], rf_data[1, :]
    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    sos_lp = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    rf_pulse = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
    t_rf = np.arange(len(rf_pulse)) / FS
    rf_env = env_smooth(rf_pulse, 1.5, FS)
    
    # Restrict search between 24 and 43 seconds (deflation period)
    mask = (t_rf >= 24.0) & (t_rf <= 43.0)
    t_map = t_rf[mask][np.argmax(rf_env[mask])]
    print(f"{name} MAP peak time: {t_map:.2f} s")

get_map('Sub_1_Prof_kan/Rec_6.h5', 'Sub 1')
get_map('Sub_2_Rajveer/Rec_4.h5', 'Sub 2')
