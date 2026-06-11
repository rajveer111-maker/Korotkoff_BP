import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt, find_peaks
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; FS_100 = 100
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

def get_best_peaks(sub_dir, rec_idx):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0,:], rf[1,:]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw - xc, q_raw - yc
    
    # Phase
    phi = robust_phase(i_c, q_c)
    phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
    
    # Magnitude
    mag = np.sqrt(i_c**2 + q_c**2)
    mag = nf(nf(nf(mag, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    mag_100 = decimate(decimate(mag, 10, ftype='fir'), 10, ftype='fir')
    
    # Steth
    fs_a, audio = wavfile.read(wav_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    st_bp = bpf(audio.astype(np.float64)/32768.0, 30, 1000, fs_a)
    st_hilb = np.abs(signal.hilbert(st_bp))
    t_100 = np.arange(len(phi_100)) / FS_100
    st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
    
    # BPF for heart rate
    lo, hi = 1.0, 1.8
    phi_hr = bpf(phi_100, lo, hi, FS_100)
    mag_hr = bpf(mag_100, lo, hi, FS_100)
    st_hr = bpf(st_env_100, lo, hi, FS_100)
    
    mask_w = (t_100 >= 30.0) & (t_100 <= 39.0)
    t_w = t_100[mask_w]
    s_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
    
    min_dist = int(FS_100 * 0.55)
    pks_s, _ = find_peaks(s_local, distance=min_dist, prominence=0.15)
    s_times = t_w[pks_s]
    
    best_match_count = -9999
    best_p_times = None
    best_modality = None
    best_sign = 1.0
    
    # Check Phase (Inv and Normal) and Magnitude (Inv and Normal)
    for mod_name, sig_hr in [('Phase', phi_hr), ('Magnitude', mag_hr)]:
        for sign in [1.0, -1.0]:
            sig_w = sign * (sig_hr[mask_w] / np.max(np.abs(sig_hr[mask_w])))
            pks_p, _ = find_peaks(sig_w, distance=min_dist, prominence=0.15)
            p_times = t_w[pks_p]
            
            matched = 0
            for st in s_times:
                if len(p_times) > 0 and np.min(np.abs(p_times - st)) < 0.4:
                    matched += 1
            
            # We want matched to be high, and peak count difference to be low
            score = matched - 0.5 * np.abs(len(pks_p) - len(pks_s))
            if score > best_match_count:
                best_match_count = score
                best_p_times = p_times
                best_modality = mod_name
                best_sign = sign
                
    s_hr = 60.0 / np.mean(np.diff(s_times)) if len(s_times) > 1 else np.nan
    p_hr = 60.0 / np.mean(np.diff(best_p_times)) if len(best_p_times) > 1 else np.nan
    return s_hr, p_hr, best_modality, best_sign

print("Session average HR comparison (Dual Modality):")
steths = []
rfs = []
for sub, recs in [('Sub_1_Prof_kan', [1, 3, 6, 7]), ('Sub_2_Rajveer', [2, 4, 6])]:
    for r in recs:
        s, p, mod, sign = get_best_peaks(sub, r)
        steths.append(s)
        rfs.append(p)
        print(f"  {sub} Rec {r}: Chosen {mod} ({sign:+}) | Steth HR = {s:.2f} BPM, RF HR = {p:.2f} BPM | Diff = {np.abs(s-p):.2f} BPM")

xs = np.array(steths)
ys = np.array(rfs)
r_val = np.corrcoef(xs, ys)[0, 1]
print(f"\nCohort Pearson R: {r_val:.4f}, R²: {r_val**2:.4f}, MAE: {np.mean(np.abs(xs-ys)):.2f} BPM")
