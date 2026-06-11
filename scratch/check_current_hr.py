import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

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

# HR on RF
t_stable = disp_hr_rf[int(10*FS_RF):int(20*FS_RF)]
pth = np.std(t_stable)*0.8
t_rf = np.arange(len(disp_hr_rf))/FS_RF
peaks_rf, _ = signal.find_peaks(-disp_hr_rf, distance=int(FS_RF*0.5), prominence=pth)
if len(peaks_rf)>1:
    iv = np.diff(t_rf[peaks_rf]); viv = iv[(iv>0.4)&(iv<1.5)]
    hr_rf_bpm = 60.0/np.median(viv) if len(viv)>0 else 0
else: hr_rf_bpm = 0

print(f"Time-Domain RF Peaks HR: {hr_rf_bpm:.1f} BPM, peaks found: {len(peaks_rf)}")
