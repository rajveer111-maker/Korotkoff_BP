"""
Test script to plot low-frequency heartbeat modulation (0.9 to 2.5 Hz)
for a 10-second zoomed-in window [30, 40] seconds, comparing Steth and RF.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt
from scipy.io import wavfile
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FS_100 = 100
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

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

def plot_hb(sub_dir, rec_idx, ax):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Low-freq phase (0.9 to 2.5 Hz)
    phi = robust_phase(i_c, q_c)
    phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
    t_100 = np.arange(len(phi_100)) / FS_100
    phi_hr = bpf(phi_100, 0.9, 2.5, FS_100)
    
    # Audio env BPF at 100 Hz
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    st_bp = bpf(audio, 30, 1000, fs_a)
    st_hilb = np.abs(signal.hilbert(st_bp))
    st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
    st_hr = bpf(st_env_100, 0.9, 2.5, FS_100)
    
    # Zoom window [30, 40]
    mask_z = (t_100 >= 30.0) & (t_100 <= 40.0)
    t_z = t_100[mask_z]
    s_z = st_hr[mask_z] / np.max(np.abs(st_hr[mask_z]))
    p_z = -phi_hr[mask_z] / np.max(np.abs(phi_hr[mask_z]))  # Inverted phase
    
    ax.plot(t_z, s_z, color='#2980B9', label='Steth Envelope Modulation', lw=2)
    ax.plot(t_z, p_z, color='#C0392B', label='RF Phase (Inverted)', lw=2)
    ax.set_xlim([30, 40])
    ax.set_ylim([-1.2, 1.2])
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{sub_dir} Rec {rec_idx}")

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
plot_hb('Sub_1_Prof_kan', 1, axs[0, 0])
plot_hb('Sub_1_Prof_kan', 3, axs[1, 0])
plot_hb('Sub_1_Prof_kan', 6, axs[2, 0])

plot_hb('Sub_2_Rajveer', 2, axs[0, 1])
plot_hb('Sub_2_Rajveer', 4, axs[1, 1])
plot_hb('Sub_2_Rajveer', 6, axs[2, 1])

axs[0, 0].legend()
plt.tight_layout()
plt.savefig(r'C:\Users\rajve\.gemini\antigravity\brain\455975f3-9b17-4899-a08c-147aeebc3fe5\artifacts\test_heartbeat_zoom.png', dpi=150)
print("Done plotting heartbeat zoom!")
