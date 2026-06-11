"""
Test script to compare Phase-based and Mag-based envelopes against Steth
across all 6 sessions in the dashboard.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, hilbert, decimate, iirnotch, filtfilt, fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    k = max(1, int(w * fs))
    return fftconvolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    return xc, yc

def robust_phase(ic, qc):
    iq = ic + 1j*qc
    dp = np.angle(iq[1:] * np.conj(iq[:-1]))
    h, bins = np.histogram(dp, bins=512)
    co = bins[np.argmax(h)] + (bins[1]-bins[0])/2
    dp -= co
    iqr = np.percentile(dp, 75) - np.percentile(dp, 25)
    dp = np.clip(dp, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dp), 0, 0.0), type='linear')

def process_and_plot(sub_dir, rec_idx, ax):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Phase Env
    phi = robust_phase(i_c, q_c)
    phi_clean = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    vel_hi = np.append(np.diff(bpf(phi_clean, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_dec = decimate(smooth(tkeo(vel_hi), 0.15, FS_RF), DEC, ftype='fir')
    vel_env = smooth(np.maximum(vel_dec, 0), 1.5, FS)
    t_rf = np.arange(len(vel_env)) / FS
    
    # Mag Env
    mag_raw = np.sqrt(i_c**2 + q_c**2)
    mag_clean = nf(nf(nf(mag_raw, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    mag_hi = bpf(mag_clean, 30, 200, FS_RF)
    mag_dec = decimate(smooth(tkeo(mag_hi), 0.15, FS_RF), DEC, ftype='fir')
    mag_env = smooth(np.maximum(mag_dec, 0), 1.5, FS)
    
    # Audio
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    st_bp = bpf(audio, 30, 1000, fs_a)
    st_hilb = np.abs(hilbert(st_bp))
    st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
    st_wide_a = smooth(tkeo(st_koro), 1.5, fs_a)
    steth_env = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)
    
    mask = (t_rf >= 22.0) & (t_rf <= 45.0)
    
    def get_clean_norm(env):
        base = np.percentile(env[mask], 5)
        env_clean = np.maximum(env - base, 0)
        return env_clean / (np.max(env_clean[mask]) + 1e-12)
        
    s_norm = get_clean_norm(steth_env)
    p_norm = get_clean_norm(vel_env)
    m_norm = get_clean_norm(mag_env)
    
    ax.plot(t_rf, s_norm, color='#2980B9', label='Steth')
    ax.plot(t_rf, p_norm, color='#C0392B', label='Phase (Red)')
    ax.plot(t_rf, m_norm, color='#9B59B6', label='Mag (Purple)', ls='--')
    ax.set_title(f"{sub_dir} Rec {rec_idx}")
    ax.set_xlim([22, 45])
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3)

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
process_and_plot('Sub_1_Prof_kan', 1, axs[0, 0])
process_and_plot('Sub_1_Prof_kan', 3, axs[1, 0])
process_and_plot('Sub_1_Prof_kan', 6, axs[2, 0])

process_and_plot('Sub_2_Rajveer', 2, axs[0, 1])
process_and_plot('Sub_2_Rajveer', 4, axs[1, 1])
process_and_plot('Sub_2_Rajveer', 6, axs[2, 1])

axs[0, 0].legend()
plt.tight_layout()
plt.savefig(r'C:\Users\rajve\.gemini\antigravity\brain\455975f3-9b17-4899-a08c-147aeebc3fe5\artifacts\compare_phase_mag_all.png', dpi=150)
print("Done comparing all!")
