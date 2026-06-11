import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch

RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5'
FS_RF      = 10_000
FC_HZ      = 0.9e9

C = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

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
with h5py.File(RF_PATH,'r') as f: data=f['data'][:]
ir, qr = data[0,:], data[1,:]
iq = iq_condition(apply_iq(ir, qr))
phase = robust_phase(iq)

# HR displacement
sos_h = butter(4,[0.5,3.0],btype='band',fs=FS_RF,output='sos')
disp_hr_rf = sosfiltfilt(sos_h, phase)*SCALE

# Heart Rate Velocity
vel_hr_rf = np.diff(disp_hr_rf) * FS_RF

# Calculate PSD on Velocity
f_rf, p_rf = welch(vel_hr_rf, fs=FS_RF, nperseg=min(len(vel_hr_rf), int(FS_RF * 15)))

# Find local maxima in 0.5 - 3.0 Hz range
mask = (f_rf >= 0.5) & (f_rf <= 3.0)
f_range = f_rf[mask]
p_range = p_rf[mask]

peaks, _ = signal.find_peaks(p_range, distance=5)
# Sort peaks by amplitude
sorted_peaks = sorted(peaks, key=lambda x: p_range[x], reverse=True)

print("Top 5 PSD peaks in RF Velocity (0.5 to 3.0 Hz):")
for i, p_idx in enumerate(sorted_peaks[:5]):
    freq = f_range[p_idx]
    bpm = freq * 60.0
    amp = p_range[p_idx]
    print(f"Peak {i+1}: {bpm:.1f} BPM (Freq: {freq:.3f} Hz, Power: {amp:.2e})")
