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

def evaluate(sub_dir, rec_idx, lo_cutoff):
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
    
    phi = robust_phase(i_c, q_c)
    phi = nf(nf(nf(phi, 64, FS_RF), 100.6, FS_RF), 50, FS_RF)
    phi_100 = decimate(decimate(phi * SCALE, 10, ftype='fir'), 10, ftype='fir')
    t_100 = np.arange(len(phi_100)) / FS_100
    
    fs_a, audio = wavfile.read(wav_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    st_bp = bpf(audio.astype(np.float64)/32768.0, 30, 1000, fs_a)
    st_hilb = np.abs(signal.hilbert(st_bp))
    st_env_100 = np.interp(t_100, np.arange(len(st_hilb))/fs_a, st_hilb)
    
    phi_hr = bpf(phi_100, lo_cutoff, 2.5, FS_100)
    st_hr = bpf(st_env_100, lo_cutoff, 2.5, FS_100)
    
    mask_w = (t_100 >= 30.0) & (t_100 <= 39.0)
    t_w = t_100[mask_w]
    s_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
    
    # Let's find best sign
    best_match = -1
    best_sign = 1.0
    best_s_pks, best_p_pks = None, None
    
    for sign in [1.0, -1.0]:
        p_local = sign * (phi_hr[mask_w] / np.max(np.abs(phi_hr[mask_w])))
        
        min_dist = int(FS_100 * 0.5)
        pks_s, _ = find_peaks(s_local, distance=min_dist, prominence=0.2)
        pks_p, _ = find_peaks(p_local, distance=min_dist, prominence=0.2)
        
        s_times = t_w[pks_s]
        p_times = t_w[pks_p]
        
        matched = 0
        for st in s_times:
            if len(p_times) > 0 and np.min(np.abs(p_times - st)) < 0.4:
                matched += 1
        if matched > best_match:
            best_match = matched
            best_sign = sign
            best_s_pks = s_times
            best_p_pks = p_times
            
    return len(best_s_pks), len(best_p_pks), best_match, best_sign

print("Comparing lo=0.9 Hz vs lo=1.1 Hz vs lo=1.2 Hz:")
for lo in [0.9, 1.1, 1.2]:
    print(f"\n--- Low Cutoff: {lo:.1f} Hz ---")
    print("Subject 1:")
    for r in [1, 2, 3, 6, 7, 8, 9]:
        res = evaluate('Sub_1_Prof_kan', r, lo)
        if res:
            print(f"  Rec {r}: Steth {res[0]} | RF {res[1]} | Matched {res[2]} (sign: {res[3]:+})")
    print("Subject 2:")
    for r in [2, 4, 6]:
        res = evaluate('Sub_2_Rajveer', r, lo)
        if res:
            print(f"  Rec {r}: Steth {res[0]} | RF {res[1]} | Matched {res[2]} (sign: {res[3]:+})")
