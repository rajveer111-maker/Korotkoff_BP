import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, filtfilt, iirnotch, fftconvolve
from scipy.io import wavfile as wav

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
FS_RF = 10000; DEC = 10; FS = 1000
FC = 0.9e9
LAMBDA_MM = (299_792_458.0 / FC) * 1000.0
SCALE     = LAMBDA_MM / (4.0 * np.pi)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res[0], res[1], res[2]
    xc = -a / 2
    yc = -b / 2
    R = np.sqrt(xc**2 + yc**2 - c)
    return xc, yc, R

def iq_condition_circle(i_raw, q_raw):
    xc, yc, R = fit_circle(i_raw, q_raw)
    return i_raw - xc, q_raw - yc, xc, yc, R

def robust_phase(i_c, q_c):
    iq = i_c + 1j * q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi_c = dphi - co
    iqr = np.percentile(dphi_c, 75) - np.percentile(dphi_c, 25)
    clip = max(3 * iqr, 0.01)
    dphi_c = np.clip(dphi_c, -clip, clip)
    phase = np.insert(np.cumsum(dphi_c), 0, 0.0)
    return signal.detrend(phase, type='linear')

def notch(x, f0, fs, Q=30):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

SUB1_RF  = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
SUB1_WAV = os.path.join(BASE, 'Sub_1_Prof_kan', 'sthethoscope_rec06.wav')
SUB2_RF  = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
SUB2_WAV = os.path.join(BASE, 'Sub_2_Rajveer', 'sthethoscope_rec04.wav')

def test_hr(rf_path, wav_path, defl_onset, notches, lag):
    with h5py.File(rf_path, 'r') as f:
        rf_data = f['data'][:]
    i_raw, q_raw = -rf_data[0,:], rf_data[1,:]
    t_rf = np.arange(len(i_raw)) / FS_RF
    i_c, q_c, _, _, _ = iq_condition_circle(i_raw, q_raw)
    phi = robust_phase(i_c, q_c)
    for freq in notches:
        phi = notch(phi, freq, FS_RF)
    
    t_start_clean = defl_onset + 3.0
    
    sos_dh = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh = sosfiltfilt(sos_dh, phi) * SCALE
    dh[t_rf < t_start_clean] = 0.0
    
    fs_aud, audio_stereo = wav.read(wav_path)
    audio = audio_stereo[:, 0].astype(np.float32)
    ds_factor = 4
    audio_ds = signal.decimate(audio, ds_factor)
    fs_aud_ds = fs_aud // ds_factor
    t_aud = (np.arange(len(audio_ds)) / fs_aud_ds) + lag
    
    sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
    ka = sosfiltfilt(sos_aud, audio_ds)
    ka[t_aud < t_start_clean] = 0.0
    
    audio_env = np.abs(signal.hilbert(ka))
    sos_hr_a = butter(4, [0.4, 3.0], btype='band', fs=fs_aud_ds, output='sos')
    dh_acoustic = sosfiltfilt(sos_hr_a, audio_env)
    dh_acoustic[t_aud < t_start_clean] = 0.0
    
    # Calculate heart rate (HR) from RF displacement on stable window (defl_onset to end)
    mask_hr_rf = (t_rf >= defl_onset) & (t_rf <= t_rf[-1])
    t_stable_rf = t_rf[mask_hr_rf]
    dh_stable_rf = dh[mask_hr_rf]
    min_dist_rf = int(FS_RF * 0.5)
    peaks_rf, _ = signal.find_peaks(dh_stable_rf, distance=min_dist_rf, prominence=np.std(dh_stable_rf)*0.5)
    hr_rf = 60.0 / np.mean(np.diff(t_stable_rf[peaks_rf])) if len(peaks_rf) > 1 else 0.0

    # Calculate heart rate (HR) from Steth displacement on stable window (defl_onset to end)
    mask_hr_st = (t_aud >= defl_onset) & (t_aud <= t_aud[-1])
    t_stable_st = t_aud[mask_hr_st]
    dh_stable_st = dh_acoustic[mask_hr_st]
    min_dist_st = int(fs_aud_ds * 0.5)
    peaks_st, _ = signal.find_peaks(dh_stable_st, distance=min_dist_st, prominence=np.std(dh_stable_st)*0.5)
    hr_st = 60.0 / np.mean(np.diff(t_stable_st[peaks_st])) if len(peaks_st) > 1 else 0.0

    print(f"defl_onset = {defl_onset} s | RF HR = {hr_rf:.2f} BPM, Steth HR = {hr_st:.2f} BPM")
    return hr_rf, hr_st

print("Subject 1:")
test_hr(SUB1_RF, SUB1_WAV, defl_onset=18.0, notches=[100.71, 201.43, 302.14, 402.86], lag=1.7083)

print("Subject 2:")
test_hr(SUB2_RF, SUB2_WAV, defl_onset=18.6, notches=[50.0, 64.0, 100.6, 201.2], lag=2.6042)
