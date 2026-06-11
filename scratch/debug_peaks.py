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

def debug_session(sub_dir, rec_idx, lo):
    rf_path = os.path.join(BASE, sub_dir, f'Rec_{rec_idx}.h5')
    wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')
    if not os.path.exists(wav_path):
        wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx}.wav')
        
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
    
    phi_hr = bpf(phi_100, lo, 2.5, FS_100)
    st_hr = bpf(st_env_100, lo, 2.5, FS_100)
    
    mask_w = (t_100 >= 30.0) & (t_100 <= 39.0)
    t_w = t_100[mask_w]
    s_local = st_hr[mask_w] / np.max(np.abs(st_hr[mask_w]))
    
    best_match = -1
    best_s_pks, best_p_pks = None, None
    best_sign = 1.0
    
    for sign in [1.0, -1.0]:
        p_local = sign * (phi_hr[mask_w] / np.max(np.abs(phi_hr[mask_w])))
        min_dist = int(FS_100 * 0.50)
        pks_s, _ = find_peaks(s_local, distance=min_dist, prominence=0.20)
        pks_p, _ = find_peaks(p_local, distance=min_dist, prominence=0.20)
        
        s_times = t_w[pks_s]
        p_times = t_w[pks_p]
        
        matched = 0
        for st in s_times:
            if len(p_times) > 0 and np.min(np.abs(p_times - st)) < 0.4:
                matched += 1
        if matched > best_match:
            best_match = matched
            best_s_pks = s_times
            best_p_pks = p_times
            best_sign = sign
            
    print(f"\n=================== {sub_dir} Rec {rec_idx} (sign: {best_sign:+.1f}) ===================")
    print("Steth peaks:", best_s_pks)
    print("RF peaks:   ", best_p_pks)
    s_bpms = 60.0 / np.diff(best_s_pks)
    p_bpms = 60.0 / np.diff(best_p_pks)
    print("Steth bpms: ", np.round(s_bpms, 1))
    print("RF bpms:    ", np.round(p_bpms, 1))
    
    s_mids = (best_s_pks[:-1] + best_s_pks[1:]) / 2.0
    p_mids = (best_p_pks[:-1] + best_p_pks[1:]) / 2.0
    for i, sm in enumerate(s_mids):
        diffs = np.abs(p_mids - sm)
        if len(diffs) == 0: continue
        idx = np.argmin(diffs)
        if diffs[idx] < 0.4:
            print(f"  Match: Steth Mid {sm:.2f} (HR {s_bpms[i]:.1f}) <-> RF Mid {p_mids[idx]:.2f} (HR {p_bpms[idx]:.1f}) | Diff: {diffs[idx]:.2f}s | Err: {np.abs(s_bpms[i] - p_bpms[idx]):.1f} BPM")

debug_session('Sub_1_Prof_kan', 1, 1.2)
debug_session('Sub_1_Prof_kan', 3, 1.2)
debug_session('Sub_1_Prof_kan', 6, 1.2)
