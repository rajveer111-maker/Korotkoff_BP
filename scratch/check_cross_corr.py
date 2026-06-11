"""
Verify lag compensation between Stethoscope and RF signals.
Finds the cross-correlation peak to align signals.
"""
import h5py, os, numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, find_peaks, detrend, correlate
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC1 = 10; DEC2 = 10
FS_100 = 100
FC    = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000.0) / (4.0 * np.pi)

SESSIONS = [
    dict(sub_dir='Sub_1_Prof_kan', rec=6, k_on=27.53, k_off=43.33),
    dict(sub_dir='Sub_2_Rajveer', rec=4, k_on=27.38, k_off=42.00),
]

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

for s in SESSIONS:
    # Load RF
    rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    with h5py.File(rp, 'r') as f: raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc); ic -= xc; qc -= yc

    phi = robust_phase(ic, qc) * SCALE
    mag_raw = np.sqrt(ic**2 + qc**2)

    phi_100 = decimate(decimate(phi, DEC1, ftype='fir'), DEC2, ftype='fir')
    mag_100 = decimate(decimate(mag_raw, DEC1, ftype='fir'), DEC2, ftype='fir')

    phi_hr = bpf(phi_100, 1.0, 2.5, FS_100)
    mag_hr = bpf(mag_100, 1.0, 2.5, FS_100)

    # Load Stethoscope
    wp = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")
    fs_a, aud = wavfile.read(wp)
    aud = aud.astype(np.float64) / 32768.0
    if aud.ndim > 1: aud = aud.mean(1)

    st_bp = bpf(aud, 30, 1000, fs_a)
    st_env_a = np.abs(hilbert(st_bp))

    t_rf = np.arange(len(phi_100)) / FS_100
    t_a = np.arange(len(st_env_a)) / fs_a
    st_env_100 = np.interp(t_rf, t_a, st_env_a)

    st_hr = bpf(st_env_100, 1.0, 2.5, FS_100)

    # Extract Korotkoff window ONLY
    mask = (t_rf >= s['k_on']) & (t_rf <= s['k_off'])
    t_k = t_rf[mask]
    st_k = st_hr[mask]
    phi_k = phi_hr[mask]
    mag_k = mag_hr[mask]

    # Normalize locally inside the Korotkoff window
    st_k = st_k / np.max(np.abs(st_k))
    phi_k = phi_k / np.max(np.abs(phi_k))
    mag_k = mag_k / np.max(np.abs(mag_k))

    # Cross-correlate to find lag
    corr_phi = correlate(phi_k, st_k, mode='same')
    lags = np.arange(-len(st_k)//2, len(st_k)//2)
    best_lag_phi = lags[np.argmax(np.abs(corr_phi))] / FS_100

    corr_mag = correlate(mag_k, st_k, mode='same')
    best_lag_mag = lags[np.argmax(np.abs(corr_mag))] / FS_100

    print(f"\n{s['sub_dir']} (Rec {s['rec']}):")
    print(f"  Optimal lag (Phase vs Steth): {best_lag_phi*1000:.1f} ms")
    print(f"  Optimal lag (Mag vs Steth):   {best_lag_mag*1000:.1f} ms")
