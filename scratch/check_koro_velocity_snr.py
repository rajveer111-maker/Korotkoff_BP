import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5'
FS_RF      = 10_000
FC_HZ      = 0.9e9
SCALE      = ((299792458 / FC_HZ) * 1000) / (4 * np.pi)

def apply_iq(i, q):
    return -i + 1j * q

def iq_condition(iq):
    ic, qc = iq.real - iq.real.mean(), iq.imag - iq.imag.mean()
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc = (qc - sp*ic) / (al*cp + 1e-15)
    return ic + 1j*qc

def robust_phase(iq):
    dphi = np.angle(iq[1:]*np.conj(iq[:-1]))
    h, b = np.histogram(dphi, 512)
    co = b[np.argmax(h)] + (b[1]-b[0])/2
    dc = dphi - co
    iqr = np.percentile(dc,75)-np.percentile(dc,25)
    dc = np.clip(dc, -max(3*iqr,0.017), max(3*iqr,0.017))
    return signal.detrend(np.insert(np.cumsum(dc),0,0.0))

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

ir, qr = data[0,:], data[1,:]
N = len(ir)
t = np.arange(N)/FS_RF

iq = iq_condition(apply_iq(ir, qr))
phase = robust_phase(iq)

# Koro bandpass
sos_k = butter(4,[10,49],btype='band',fs=FS_RF,output='sos')
pk = sosfiltfilt(sos_k, phase)
vk = np.append(np.diff(pk)*FS_RF, 0)*SCALE

# Detected RF Window in koro_rf_vs_stethoscope.py:
# RF Window: 18.50s - 29.00s
# Steth Window: 16.25s - 26.75s
# Let's check RMS in active region (18.5s - 29.0s) vs noise region (5s - 10s)
idx_active_start = int(18.5 * FS_RF)
idx_active_end   = int(29.0 * FS_RF)
idx_noise_start  = int(5.0 * FS_RF)
idx_noise_end    = int(10.0 * FS_RF)

active_vel = vk[idx_active_start:idx_active_end]
noise_vel  = vk[idx_noise_start:idx_noise_end]

rms_active = np.sqrt(np.mean(active_vel**2))
rms_noise  = np.sqrt(np.mean(noise_vel**2))
snr = 20 * np.log10(rms_active / rms_noise)

print(f"Velocity Range: [{np.min(vk):.6f}, {np.max(vk):.6f}] mm/s")
print(f"Active RMS: {rms_active:.6f} mm/s")
print(f"Noise RMS: {rms_noise:.6f} mm/s")
print(f"SNR: {snr:.2f} dB")
