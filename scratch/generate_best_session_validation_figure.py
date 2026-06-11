"""
Generate high-fidelity academic validation figure for the 1 best session (Subject 1, Rec 6)
Layout: 1 row x 3 columns (Horizontal Layout, perfect for academic papers)
- Panel A: Broad Korotkoff Envelopes (22s to 45s)
- Panel B: High-zoom Beat-by-Beat Waveform Alignment (32s to 36s) (Shows detailed pulse morphology)
- Panel C: Instantaneous Heart Rate Tracking (30s to 39s)
Saved in 300 DPI with publication-ready white background.
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = os.path.join(BASE, 'best_session_validation_300dpi.png')
FS_RF = 10000; DEC = 10; FS = 1000; FS_100 = 100
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

# Color Scheme for academic publication
CP = '#C0392B'   # Crimson Red - Near-field RF (USRP)
CS = '#2980B9'   # Ocean Blue - Stethoscope Ground Truth

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def lpf(x, fc, fs, order=2):
    sos = butter(order, fc, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def nf(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

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

def process_session(sub_dir, rec_idx):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
    if not os.path.exists(rf_path) or not os.path.exists(wav_path):
        return None
        
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # 1. Broad Envelopes Processing
    phi = robust_phase(i_c, q_c)
    phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    
    vel_hi = np.append(np.diff(bpf(phi, 30, 200, FS_RF)) * FS_RF, 0.0) * SCALE
    vel_dec = decimate(lpf(tkeo(vel_hi), 1.0, FS_RF), DEC, ftype='fir')
    rf_env = lpf(np.maximum(vel_dec, 0), 1.0, FS)
    t_rf = np.arange(len(rf_env)) / FS
    
    # Audio processing
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1: audio = audio.mean(axis=1)
    st_bp = bpf(audio, 30, 1000, fs_a)
    st_hilb = np.abs(signal.hilbert(st_bp))
    st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)
    st_wide_a = lpf(tkeo(st_koro), 1.0, fs_a)
    steth_env = np.interp(t_rf, np.arange(len(st_wide_a))/fs_a, st_wide_a)
    
    # 2. Heartbeat Waveform Zoom Processing
    phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
    t_100 = np.arange(len(phi_100)) / FS_100
    st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
    
    phi_hr = bpf(phi_100, 1.0, 1.8, FS_100)
    mag_hr = bpf(np.sqrt(i_c**2 + q_c**2), 1.0, 1.8, FS_RF)
    mag_hr_100 = decimate(decimate(mag_hr, 10, ftype='fir'), 10, ftype='fir')
    st_hr = bpf(st_env_100, 1.0, 1.8, FS_100)
    
    # Local normalization for broad envelopes in [22, 45]s
    mask_env = (t_rf >= 22.0) & (t_rf <= 45.0)
    r_base = np.percentile(rf_env[mask_env], 5)
    rf_clean = np.maximum(rf_env - r_base, 0)
    r_norm = rf_clean / (np.max(rf_clean[mask_env]) + 1e-12)
    
    s_base = np.percentile(steth_env[mask_env], 5)
    steth_clean = np.maximum(steth_env - s_base, 0)
    s_norm = steth_clean / (np.max(steth_clean[mask_env]) + 1e-12)
    
    # Dynamic Modality Selection in stable heartbeat window [30, 39]s
    mask_w = (t_100 >= 30.0) & (t_100 <= 39.0)
    t_w = t_100[mask_w]
    s_hr_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
    
    min_dist = int(FS_100 * 0.55)
    s_peaks, _ = find_peaks(s_hr_local, distance=min_dist, prominence=0.15)
    s_times = t_w[s_peaks]
    
    best_score = -9999
    best_p_times = None
    best_modality = None
    best_sign = 1.0
    best_sig_w = None
    best_p_peaks = None
    
    for mod_name, sig_hr in [('Phase', phi_hr), ('Magnitude', mag_hr_100)]:
        for sign in [1.0, -1.0]:
            sig_w = sign * (sig_hr[mask_w] / np.max(np.abs(sig_hr[mask_w])))
            pks, _ = find_peaks(sig_w, distance=min_dist, prominence=0.15)
            p_times = t_w[pks]
            
            matched = 0
            for st in s_times:
                if len(p_times) > 0 and np.min(np.abs(p_times - st)) < 0.4:
                    matched += 1
            
            score = matched - 0.5 * np.abs(len(pks) - len(s_peaks))
            if score > best_score:
                best_score = score
                best_p_times = p_times
                best_modality = mod_name
                best_sign = sign
                best_sig_w = sig_w
                best_p_peaks = pks
                
    # Instantaneous HR values
    s_hr_vals = 60.0 / np.diff(s_times) if len(s_times) > 1 else np.array([])
    s_hr_t = (s_times[:-1] + s_times[1:]) / 2.0
    
    p_hr_vals = 60.0 / np.diff(best_p_times) if len(best_p_times) > 1 else np.array([])
    p_hr_t = (best_p_times[:-1] + best_p_times[1:]) / 2.0
    
    return {
        't_rf': t_rf, 'r_norm': r_norm, 's_norm': s_norm,
        't_w': t_w, 's_hr_w': s_hr_local, 'p_hr_w': best_sig_w,
        's_times': s_times, 'p_times': best_p_times,
        's_hr_t': s_hr_t, 's_hr_vals': s_hr_vals,
        'p_hr_t': p_hr_t, 'p_hr_vals': p_hr_vals,
        't_100': t_100, 's_peaks': s_peaks, 'p_peaks': best_p_peaks,
        'modality': best_modality, 'sign': best_sign
    }

# Process the single best session: Subject 1, Rec 6
res = process_session('Sub_1_Prof_kan', 6)

# Setup publication style plotting parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8.5,
    'lines.linewidth': 1.8,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor='white')

# 1. Broad Korotkoff Envelopes
ax1 = axes[0]
ax1.plot(res['t_rf'], res['s_norm'], color=CS, label='Stethoscope')
ax1.plot(res['t_rf'], res['r_norm'], color=CP, label=f'RF ({res["modality"]})')
ax1.set_xlim([22, 45])
ax1.set_ylim([0, 1.15])
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title("A. Korotkoff Envelope Dynamics", loc='left')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Normalized Envelope")
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

# 2. Zoomed-in beat-by-beat waveforms (High Zoom: 32.0 to 36.0s to clearly show pulse displacement changes)
ax2 = axes[1]
mask_zoom = (res['t_w'] >= 32.0) & (res['t_w'] <= 36.0)
t_z = res['t_w'][mask_zoom]
s_z = res['s_hr_w'][mask_zoom]
p_z = res['p_hr_w'][mask_zoom]

# Get indices within the zoom window for markers
s_p_zoom = [t for t in res['s_times'] if 32.0 <= t <= 36.0]
p_p_zoom = [t for t in res['p_times'] if 32.0 <= t <= 36.0]

ax2.plot(t_z, s_z, color=CS, label='Stethoscope Envelope')
ax2.plot(t_z, p_z, color=CP, label=f'RF ({res["modality"]})')
ax2.scatter(s_p_zoom, [np.interp(t, t_z, s_z) for t in s_p_zoom], color=CS, s=40, zorder=5, label='Stethoscope Peaks')
ax2.scatter(p_p_zoom, [np.interp(t, t_z, p_z) for t in p_p_zoom], color=CP, marker='s', s=45, zorder=5, label='RF Peaks')

ax2.set_xlim([32.0, 36.0])
ax2.set_ylim([-1.1, 1.1])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("B. High-Zoom Beat Alignment", loc='left')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Normalized Amplitude")
ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

# 3. Instantaneous Heart Rate Match
ax3 = axes[2]
ax3.plot(res['s_hr_t'], res['s_hr_vals'], 'o-', color=CS, ms=5, label='Stethoscope HR')
ax3.plot(res['p_hr_t'], res['p_hr_vals'], 's-', color=CP, ms=5, label='RF HR')
ax3.set_xlim([30, 39])
ax3.set_ylim([65, 95])
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_title("C. Beat-by-Beat HR Match", loc='left')
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Heart Rate (BPM)")
ax3.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"DONE -> {OUT}")
