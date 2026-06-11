import h5py
import numpy as np
from scipy.signal import decimate, detrend, butter, sosfiltfilt

def test_file(filepath):
    f = h5py.File(filepath, 'r')
    data = np.array(f['data'])
    I, Q = data[0, :], data[1, :]

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

    # Scale Conversion
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

    # Search window: strictly between 5.0s and rec_dur - 15.0s
    rec_dur = t_ds[-1]
    ss = int(5.0 * 1000)
    se = int((rec_dur - 15.0) * 1000)
    
    target_dur = 10.0
    ws = int(target_dur * 1000)
    
    best_score, best_on = -1, 0
    # Slide sample-by-sample for maximum sub-millisecond precision
    for s in range(ss, se - ws):
        e = s + ws
        if e > se:
            break
        epoch_sig = phase_koro[s:e]
        rms = np.sqrt(np.mean(epoch_sig**2) + 1e-20)
        
        # Simple, robust sliding RMS without any rigid Gaussian coordinate constraints
        if rms > best_score:
            best_score = rms
            best_on = t_ds[s]
            
    print(f"File: {filepath.split('/')[-1] or filepath.split('\\\\')[-1]}")
    print(f"   Best Window: {best_on:.2f} s to {best_on+10.0:.2f} s")

print("\n==================================================")
test_file(r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe.h5')
test_file(r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe_1.h5')
print("==================================================\n")
