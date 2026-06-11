import h5py
import numpy as np
from scipy.signal import decimate, detrend, butter, sosfiltfilt

# Load data
H5_FILE = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5'
f = h5py.File(H5_FILE, 'r')
data = np.array(f['data'])
I, Q = data[0, :], data[1, :]

# Decimated Demodulation Pipeline
iq = -I + 1j*Q
ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()

# IQ Balance Correction
p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
sp = p3 / np.sqrt(p1*p2+1e-20)
cp = np.sqrt(max(1-sp**2, 1e-10))
al = np.sqrt(p2/(p1+1e-20))
qc = (qc - sp*ic) / (al*cp + 1e-15)
iq_c = ic + 1j*qc

# Phase Difference
dphi = np.angle(iq_c[1:] * np.conj(iq_c[:-1]))
h, b = np.histogram(dphi, 512)
co = b[np.argmax(h)] + (b[1]-b[0])/2
dc = dphi - co

# ----------------------------------------------------
# PRE-PROCESSING CORRECTION: ULTRA-TIGHT PHYSIOLOGICAL CLIPPING
# ----------------------------------------------------
# 0.0002 rad corresponds to a peak contraction velocity of 53 mm/s, which is the 
# absolute clinical maximum for chest and wall contractions!
dc_clipped = np.clip(dc, -0.0002, 0.0002)

# Integrate phase differences
phase_rad = detrend(np.insert(np.cumsum(dc_clipped), 0, 0.0))

# Physical Scale Conversion (mm)
FC_HZ = 0.9e9
C_LIGHT = 299792458
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)
phase_mm = phase_rad * SCALE

# Downsample to 1 kHz
phase_ds = decimate(phase_mm, 10, ftype='fir')
t_ds = np.arange(len(phase_ds)) / 1000.0
mask = (t_ds >= 5.0) & (t_ds <= (t_ds[-1] - 5.0))

# Detrend before filtering
phase_detrended_ds = detrend(phase_ds)

# Highpass filter (0.5 Hz)
sos_hp = butter(4, 0.5/(0.5*1000), btype='high', output='sos')
phase_clean = sosfiltfilt(sos_hp, phase_detrended_ds)

# Cardiac (0.8 - 3.0 Hz bandpass)
sos_hr = butter(4, [0.8/(0.5*1000), 3.0/(0.5*1000)], btype='band', output='sos')
phase_hr = sosfiltfilt(sos_hr, phase_clean)

# Korotkoff displacement (10 - 49 Hz bandpass)
sos_k = butter(4, [10/(0.5*1000), 49/(0.5*1000)], btype='band', output='sos')
phase_koro_disp = sosfiltfilt(sos_k, phase_clean)

print("\n==================================================")
print("PRE-PROCESSED DISPLACEMENT METRICS (Ultra-Tight 0.0002 Clip)")
print("==================================================")
print(f"1) Overall Chest Displacement (0.5 Hz HP):")
print(f"   Peak-to-Peak : {np.ptp(phase_clean[mask])*1000:.3f} um ({np.ptp(phase_clean[mask]):.6f} mm)")
print(f"   RMS Amplitude: {np.std(phase_clean[mask])*1000:.3f} um ({np.std(phase_clean[mask]):.6f} mm)")
print("\n2) Cardiac Heartbeat Displacement (0.8 - 3.0 Hz):")
print(f"   Peak-to-Peak : {np.ptp(phase_hr[mask])*1000:.3f} um ({np.ptp(phase_hr[mask]):.6f} mm)")
print(f"   RMS Amplitude: {np.std(phase_hr[mask])*1000:.3f} um ({np.std(phase_hr[mask]):.6f} mm)")
print("\n3) Korotkoff Vibration Displacement (10 - 49 Hz):")
print(f"   Peak-to-Peak : {np.ptp(phase_koro_disp[mask])*1000:.3f} um ({np.ptp(phase_koro_disp[mask]):.6f} mm)")
print(f"   RMS Amplitude: {np.std(phase_koro_disp[mask])*1000:.3f} um ({np.std(phase_koro_disp[mask]):.6f} mm)")
print("==================================================\n")
