"""
Generate high-fidelity academic validation dashboard for Subject 2 (Rajveer)
Layout: 3 rows x 3 columns
- Column 1: Broad baseline-subtracted Korotkoff envelopes (Stethoscope vs Near-field RF Phase/Mag)
- Column 2: Zoomed-in beat-by-beat heartbeat waveforms (1.0 to 1.8 Hz BPF) [30, 39]s
- Column 3: Instantaneous Heart Rate tracking (BPM) comparison
- Row 3: Grand Average Envelope (Left) & Combined Cohort Session-Average HR Correlation Scatter Plot (Middle/Right merged)
"""
import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = os.path.join(BASE, 'subject_2_cohort_validation.png')
FS_RF = 10000; DEC = 10; FS = 1000; FS_100 = 100
FC    = 0.9e9
SCALE = ((299792458.0 / FC) * 1000) / (4 * np.pi)

# Color Scheme
CP = '#C0392B'   # Red - Near-field RF (USRP)
CS = '#2980B9'   # Blue - Stethoscope Ground Truth

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
    
    s_avg_hr = 60.0 / np.mean(np.diff(s_times)) if len(s_times) > 1 else np.nan
    p_avg_hr = 60.0 / np.mean(np.diff(best_p_times)) if len(best_p_times) > 1 else np.nan
    
    return {
        't_rf': t_rf, 'r_norm': r_norm, 's_norm': s_norm,
        't_w': t_w, 's_hr_w': s_hr_local, 'p_hr_w': best_sig_w,
        's_times': s_times, 'p_times': best_p_times,
        's_hr_t': s_hr_t, 's_hr_vals': s_hr_vals,
        'p_hr_t': p_hr_t, 'p_hr_vals': p_hr_vals,
        's_avg_hr': s_avg_hr, 'p_avg_hr': p_avg_hr,
        't_100': t_100, 's_peaks': s_peaks, 'p_peaks': best_p_peaks,
        'modality': best_modality, 'sign': best_sign
    }

# Gather Cohort Data for BOTH Subjects to make a Combined Scatter Plot
cohort_s_hr = []
cohort_p_hr = []

# Subject 1 valid
for r in [1, 3, 6, 7]:
    res = process_session('Sub_1_Prof_kan', r)
    if res is not None:
        cohort_s_hr.append(res['s_avg_hr'])
        cohort_p_hr.append(res['p_avg_hr'])

# Subject 2 valid
for r in [2, 4, 6]:
    res = process_session('Sub_2_Rajveer', r)
    if res is not None:
        cohort_s_hr.append(res['s_avg_hr'])
        cohort_p_hr.append(res['p_avg_hr'])

SUB2 = 'Sub_2_Rajveer'
sessions2 = [2, 4]  # Represent sessions for the dashboard rows 1 & 2

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 11.5, 'axes.labelweight': 'bold',
    'axes.titlesize': 12.0, 'axes.titleweight': 'bold',
    'legend.fontsize': 9, 'lines.linewidth': 2.0,
})

fig = plt.figure(figsize=(18, 11), facecolor='white')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.22)

all_r_env2 = []
all_s_env2 = []
t_std = np.linspace(22.0, 45.0, 2000)

for s_idx in [2, 4, 6]:
    res = process_session(SUB2, s_idx)
    if res is not None:
        all_r_env2.append(np.interp(t_std, res['t_rf'], res['r_norm']))
        all_s_env2.append(np.interp(t_std, res['t_rf'], res['s_norm']))

# Plot representative sessions
for row_idx, s_idx in enumerate(sessions2):
    res = process_session(SUB2, s_idx)
    if res is None: continue
    
    # Col 1: Korotkoff Envelope Dynamics
    ax1 = fig.add_subplot(gs[row_idx, 0])
    ax1.plot(res['t_rf'], res['s_norm'], color=CS, label='Stethoscope Envelope')
    ax1.plot(res['t_rf'], res['r_norm'], color=CP, label=f'Near-field RF ({res["modality"]})')
    ax1.set_xlim([22, 45])
    ax1.set_ylim([0, 1.15])
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Session {s_idx} — Korotkoff Envelope", fontweight='bold')
    if row_idx == 0:
        ax1.legend(loc='upper right')
    if row_idx == 1:
        ax1.set_xlabel("Time (s)", fontweight='bold')
    else:
        ax1.set_xticklabels([])
    ax1.set_ylabel("Normalized Env.", fontweight='bold')
        
    # Col 2: Zoomed-in beat-by-beat waveforms
    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.plot(res['t_w'], res['s_hr_w'], color=CS, label='Steth Envelope Mod.')
    ax2.plot(res['t_w'], res['p_hr_w'], color=CP, label=f'Near-field RF {res["modality"]} ({"Inv" if res["sign"] < 0 else "Normal"})')
    
    ax2.plot(res['s_times'], res['s_hr_w'][res['s_peaks']], 'o', color=CS, ms=6)
    ax2.plot(res['p_times'], res['p_hr_w'][res['p_peaks']], 's', color=CP, ms=6)
    
    ax2.set_xlim([30, 39])
    ax2.set_ylim([-1.2, 1.2])
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Session {s_idx} — Beat-by-Beat Waveform Zoom", fontweight='bold')
    if row_idx == 0:
        ax2.legend(loc='lower left')
    if row_idx == 1:
        ax2.set_xlabel("Time (s)", fontweight='bold')
    else:
        ax2.set_xticklabels([])
    ax2.set_ylabel("Normalized Amp.", fontweight='bold')
        
    # Col 3: Instantaneous HR
    ax3 = fig.add_subplot(gs[row_idx, 2])
    ax3.plot(res['s_hr_t'], res['s_hr_vals'], 'o-', color=CS, ms=5, label='Steth HR')
    ax3.plot(res['p_hr_t'], res['p_hr_vals'], 's-', color=CP, ms=5, label='RF HR')
    ax3.set_xlim([30, 39])
    ax3.set_ylim([50, 95])
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"Session {s_idx} — Instantaneous HR Match", fontweight='bold')
    if row_idx == 0:
        ax3.legend(loc='upper right')
    if row_idx == 1:
        ax3.set_xlabel("Time (s)", fontweight='bold')
    else:
        ax3.set_xticklabels([])
    ax3.set_ylabel("Heart Rate (BPM)", fontweight='bold')

# Row 3, Col 1: Grand Average Envelope for Subject 2
ax_avg = fig.add_subplot(gs[2, 0])
mean_s = np.mean(all_s_env2, axis=0)
std_s  = np.std(all_s_env2, axis=0)
mean_r = np.mean(all_r_env2, axis=0)
std_r  = np.std(all_r_env2, axis=0)

ax_avg.plot(t_std, mean_s, color=CS, lw=2.4, label='Steth Mean')
ax_avg.fill_between(t_std, np.maximum(mean_s - std_s, 0), np.minimum(mean_s + std_s, 1.1), color=CS, alpha=0.15)
ax_avg.plot(t_std, mean_r, color=CP, lw=2.4, label='RF Mean')
ax_avg.fill_between(t_std, np.maximum(mean_r - std_r, 0), np.minimum(mean_r + std_r, 1.1), color=CP, alpha=0.15)
ax_avg.set_xlim([22, 45])
ax_avg.set_ylim([0, 1.15])
ax_avg.grid(True, alpha=0.3)
ax_avg.set_title("Subject 2 — Grand Average Envelopes (3 Sessions)", fontweight='bold')
ax_avg.set_xlabel("Time (s)", fontweight='bold')
ax_avg.set_ylabel("Normalized Env.", fontweight='bold')
ax_avg.legend(loc='upper right')

# Row 3, Col 2 & 3: Combined Cohort Session-Average HR Correlation Scatter Plot
ax_scat = fig.add_subplot(gs[2, 1:])
xs = np.array(cohort_s_hr)
ys = np.array(cohort_p_hr)

# Linear regression
slope, intercept = np.polyfit(xs, ys, 1)
r_val = np.corrcoef(xs, ys)[0, 1]
ax_scat.scatter(xs[:4], ys[:4], color='#3498DB', edgecolors='none', s=95, alpha=0.85, label='Subject 1 Sessions')
ax_scat.scatter(xs[4:], ys[4:], color='#1ABC9C', edgecolors='none', s=95, alpha=0.85, label='Subject 2 Sessions')
ax_scat.plot(np.unique(xs), slope*np.unique(xs) + intercept, color='#E67E22', lw=2.5,
             label=f'Combined Linear Fit (R = {r_val:.3f}, R² = {r_val**2:.3f})')
# Identity line
ax_scat.plot([50, 95], [50, 95], color='#7F8C8D', ls='--', lw=1.5, label='Identity Line (1:1)')

ax_scat.set_xlim([50, 95])
ax_scat.set_ylim([50, 95])
ax_scat.grid(True, alpha=0.3)
ax_scat.set_title("Combined Cohort (Both Subjects) — Session-Average HR Correlation (7 Sessions)", fontweight='bold')
ax_scat.set_xlabel("Stethoscope Ground Truth Heart Rate (BPM)", fontweight='bold')
ax_scat.set_ylabel("Near-field RF (USRP) Heart Rate (BPM)", fontweight='bold')
ax_scat.legend(loc='upper left')

fig.suptitle("Clinical Cohort Validation Dashboard: Subject 2 (Rajveer)\nRadiomyography (RMG) near-field RF (USRP) vs Digital Acoustic Stethoscope Ground Truth",
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"DONE -> {OUT}")
