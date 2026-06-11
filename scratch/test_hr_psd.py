import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch

RF_PATH    = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope1.mp4'
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

# Heart Rate Velocity (derivative of displacement)
vel_hr_rf = np.diff(disp_hr_rf) * FS_RF

# Calculate PSD on Velocity
f_rf, p_rf = welch(vel_hr_rf, fs=FS_RF, nperseg=min(len(vel_hr_rf), int(FS_RF * 10)))

# Test different search bands
for low_f in [0.5, 0.7, 0.75, 0.8, 0.9, 1.0]:
    mask = (f_rf >= low_f) & (f_rf <= 3.0)
    hz = f_rf[mask][np.argmax(p_rf[mask])]
    print(f"RF Velocity PSD (low_f={low_f} Hz): {hz * 60.0:.1f} BPM")
