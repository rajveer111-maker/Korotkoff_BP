import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, decimate

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k) / k, mode='same')

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    iq_hp = sosfiltfilt(sos_hp, iq)
    energy = np.abs(iq_hp)
    ds = int(fs / 100)
    t_ds = np.arange(len(i_raw))[::ds] / fs
    energy_ds = energy[::ds]
    w_size = 100
    energy_smooth = np.convolve(energy_ds, np.ones(w_size)/w_size, mode='same')
    max_search_sec = 25.0
    if onset_limit is not None:
        max_search_sec = min(max_search_sec, onset_limit - 1.0)
    search_mask = t_ds <= max_search_sec
    if not np.any(search_mask):
        return 8.0
    t_search = t_ds[search_mask]
    e_search = energy_smooth[search_mask]
    peak_idx = np.argmax(e_search)
    peak_val = e_search[peak_idx]
    end_val = np.mean(energy_smooth[max(0, int(max_search_sec*100)-50):int(max_search_sec*100)])
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        return 0.0
    baseline = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    t_det = 8.0
    for i in range(peak_idx, len(t_search)):
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
    return t_det

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic * qc)
    sp = np.clip(p3 / np.sqrt(p1 * p2 + 1e-20), -1, 1)
    cp = np.sqrt(max(1 - sp**2, 1e-10))
    al = np.sqrt(p2 / (p1 + 1e-20))
    i_new = ic
    q_new = (qc - ic * sp / al) / cp
    return i_new + 1j * q_new

with h5py.File(os.path.join(BASE, "Sub_1_Prof_kan", "Rec_6.h5"), 'r') as f:
    data = f['data'][:]
i_raw, q_raw = data[0], data[1]

t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=27.75)
t_shift = 20.0 - t_start
t_rf = np.arange(len(i_raw)) / 10000.0

iq = b210_iq_condition(-i_raw + 1j * q_raw)
sos_lp = butter(4, 50.0, btype='low', fs=10000.0, output='sos')
iq_c = sosfiltfilt(sos_lp, iq)

# Cardiac band
sos_hr = butter(4, [0.4, 3.0], btype='band', fs=10000.0, output='sos')
mag_hr_10k = sosfiltfilt(sos_hr, np.abs(iq_c))
mag_hr = decimate(mag_hr_10k, 10, ftype='fir')
mag_hr_env = smooth(np.abs(hilbert(mag_hr)), int(1.5 * 1000.0))

t_ds = np.arange(len(mag_hr)) / 1000.0
t_ds_phys = t_ds + t_shift

# Let's search the true deflation period, e.g., 20.0s to 50.0s physical time
mask_deflation = (t_ds_phys >= 20.0) & (t_ds_phys <= 50.0)
t_def = t_ds_phys[mask_deflation]
env_def = mag_hr_env[mask_deflation]

# Find all peaks in the envelope
from scipy.signal import find_peaks
peaks, properties = find_peaks(env_def, prominence=0.01, distance=1000)
print("Deflation envelope peaks between 20s and 50s:")
for p in peaks:
    print(f"Peak at t = {t_def[p]:.2f}s, value = {env_def[p]:.4f}")

# Peak of the whole envelope
idx_max = np.argmax(env_def)
print(f"Absolute max between 20s and 50s: t = {t_def[idx_max]:.2f}s, value = {env_def[idx_max]:.4f}")
