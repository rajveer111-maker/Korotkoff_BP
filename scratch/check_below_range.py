import h5py, numpy as np, os
from scipy import signal

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_below.h5'

def robust_phase(iq):
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    bin_w          = bins[1] - bins[0]
    carrier_offset = bins[np.argmax(hist)] + bin_w / 2
    dphi_c         = dphi - carrier_offset
    iqr_val  = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip_val = max(3.0 * iqr_val, 0.017)   # floor ~ 1 Hz at FS=10kHz
    dphi_c   = np.clip(dphi_c, -clip_val, clip_val)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    phase = signal.detrend(phase, type='linear')
    return phase, carrier_offset

def apply_iq_mode(i_raw, q_raw, mode):
    modes = {
        'I+jQ' : i_raw  + 1j * q_raw,
        'Q+jI' : q_raw  + 1j * i_raw,
        'I-jQ' : i_raw  - 1j * q_raw,
        '-I+jQ': -i_raw + 1j * q_raw,
    }
    return modes[mode]

def b210_iq_condition(iq_raw):
    i_c = iq_raw.real - iq_raw.real.mean()
    q_c = iq_raw.imag - iq_raw.imag.mean()
    p1 = np.mean(i_c**2)
    p2 = np.mean(q_c**2)
    p3 = np.mean(i_c * q_c)
    sin_phi = p3 / np.sqrt(p1 * p2 + 1e-20)
    cos_phi = np.sqrt(max(1.0 - sin_phi**2, 1e-10))
    alpha   = np.sqrt(p2 / (p1 + 1e-20))
    q_corr = (q_c - sin_phi * i_c) / (alpha * cos_phi + 1e-15)
    return i_c + 1j * q_corr

print("Loading RF (below)...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
iq_raw = apply_iq_mode(i_raw, q_raw, '-I+jQ')
iq = b210_iq_condition(iq_raw)

phase_clean, _ = robust_phase(iq)
print(f"rec_koro_below.h5 phase range: [{np.min(phase_clean):.2f}, {np.max(phase_clean):.2f}] rad, span: {np.max(phase_clean)-np.min(phase_clean):.2f} rad")
