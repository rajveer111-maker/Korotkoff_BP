"""
Test script to verify peak detection directly on the high-frequency TKEO envelopes
of Stethoscope, RF Phase, and RF Magnitude.
"""
import h5py, os, numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, find_peaks, detrend
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
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

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    return np.convolve(x, np.ones(int(w*fs))/int(w*fs), mode='same')

for s in SESSIONS:
    # Load RF
    rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
    with h5py.File(rp, 'r') as f: raw = f['data'][:]
    ic, qc = -raw[0,:], raw[1,:]
    xc, yc = fit_circle(ic, qc); ic -= xc; qc -= yc

    # RF Phase Velocity (30-200 Hz)
    phi = robust_phase(ic, qc)
    vel_hi = np.append(np.diff(bpf(phi, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
    t_rf = np.arange(len(vel_dec)) / FS

    # RF Magnitude |IQ| (30-200 Hz)
    mag_raw = np.sqrt(ic**2 + qc**2)
    mag_k = bpf(mag_raw, 30, 200, FS_RF)
    mag_dec = decimate(smooth(tkeo(mag_k), 0.15, FS_RF), DEC, ftype='fir')

    # Load Stethoscope
    wp = os.path.join(BASE, s['sub_dir'], f"sthethoscope_rec{s['rec']:02d}.wav")
    fs_a, aud = wavfile.read(wp)
    aud = aud.astype(np.float64) / 32768.0
    if aud.ndim > 1: aud = aud.mean(1)

    st_bp = bpf(aud, 30, 1000, fs_a)
    st_env_a = smooth(tkeo(bpf(np.abs(hilbert(st_bp)), 20, 200, fs_a)), 0.15, fs_a)
    st_fine = np.interp(t_rf, np.arange(len(st_env_a))/fs_a, st_env_a)

    # Detect peaks locally in the Korotkoff window
    mask = (t_rf >= s['k_on']) & (t_rf <= s['k_off'])
    
    st_k = st_fine.copy(); st_k[~mask] = 0
    vel_k = vel_dec.copy(); vel_k[~mask] = 0
    mag_k = mag_dec.copy(); mag_k[~mask] = 0

    st_k = st_k / np.max(st_k)
    vel_k = vel_k / np.max(vel_k)
    mag_k = mag_k / np.max(mag_k)

    min_dist = int(FS * 0.45)
    
    # Try different prominence thresholds to see if we get a solid number of beats
    pks_st, _  = find_peaks(st_k,  distance=min_dist, prominence=0.15)
    pks_vel, _ = find_peaks(vel_k, distance=min_dist, prominence=0.15)
    pks_mag, _ = find_peaks(mag_k, distance=min_dist, prominence=0.15)

    print(f"\n{s['sub_dir']} (Rec {s['rec']}):")
    print(f"  Steth peaks: {len(pks_st)}")
    print(f"  Phase peaks: {len(pks_vel)}")
    print(f"  Mag peaks:   {len(pks_mag)}")
