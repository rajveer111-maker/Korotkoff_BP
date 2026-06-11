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

# Ultra-tight physiological clipping to block clock phase slips
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

# Preprocessed (0.5 Hz highpass)
sos_hp = butter(4, 0.5/(0.5*1000), btype='high', output='sos')
phase_clean = sosfiltfilt(sos_hp, detrend(phase_ds))

# Korotkoff phase velocity
sos_k = butter(4, [10/(0.5*1000), 49/(0.5*1000)], btype='band', output='sos')
phase_koro_disp = sosfiltfilt(sos_k, phase_clean)
phase_koro = np.append(np.diff(phase_koro_disp) * 1000.0, 0.0) # mm/s velocity

# Apply 5-second noise mask
stable_mask = (t_ds >= 5.0) & (t_ds <= (t_ds[-1] - 5.0))
phase_koro_masked = np.where(stable_mask, phase_koro, 0.0)

# SLIDING 10-SECOND EPOCH ANALYSIS
epoch_len = int(10.0 * 1000) # 10 seconds at 1 kHz
step = int(0.25 * 1000)     # 0.25s slide step

print("\n==================================================")
print("TESTING BAYESIAN PHYSIOLOGICAL PRIOR EPOCHING")
print("==================================================")

best_score, best_on = -1, 0

for s in range(0, len(phase_koro_masked) - epoch_len, step):
    e = s + epoch_len
    t_start = t_ds[s]
    t_mid = t_start + 5.0 # Midpoint of the 10s epoch
    
    epoch_sig = phase_koro_masked[s:e]
    rms = np.sqrt(np.mean(epoch_sig**2))
    
    # Apply physiological Gaussian prior centered at 24.0s (diastolic/systolic window midpoint)
    # with a standard deviation of 8.0s (spanning the clinical measurement range)
    prior = np.exp(-0.5 * ((t_mid - 24.0) / 8.0)**2)
    score = rms * prior
    
    if score > best_score:
        best_score = score
        best_on = t_start

print(f"Bayesian Prior Epoching Window:")
print(f"   Best Window: {best_on:.2f} s to {best_on+10.0:.2f} s (Score = {best_score:.6f})")
print("==================================================\n")
