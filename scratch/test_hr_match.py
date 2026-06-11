import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, welch
import h5py, os

SUB1_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_6.h5"
SUB1_WAV = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\sthethoscope_rec06.wav"

SUB2_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\Rec_4.h5"
SUB2_WAV = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\sthethoscope_rec04.wav"

SCALE = (299792458.0 / 0.9e9 * 1000.0) / (4.0 * np.pi)

def test_hr(rf_path, wav_path, k_on, k_off, lag):
    # RF
    with h5py.File(rf_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = -data[0, :], data[1, :]
    # Circle fit
    A = np.column_stack([i_raw, q_raw, np.ones_like(i_raw)])
    B = -(i_raw**2 + q_raw**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = -res[0]/2, -res[1]/2
    i_c = i_raw - xc
    q_c = q_raw - yc
    
    # Demod
    iq = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi -= co
    iqr = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    phi = signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')
    
    FS_RF = 10000
    t_rf = np.arange(len(phi)) / FS_RF
    
    # RF Displacement
    sos_dh = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh = sosfiltfilt(sos_dh, phi) * SCALE
    
    # Audio
    fs_aud, audio_stereo = wav.read(wav_path)
    audio = audio_stereo[:, 0].astype(np.float32)
    ds_factor = 4
    audio_ds = signal.decimate(audio, ds_factor)
    fs_aud_ds = fs_aud // ds_factor
    t_aud = (np.arange(len(audio_ds)) / fs_aud_ds) + lag
    
    # Acoustic filter and envelope
    sos_aud = butter(4, [50, 1000], btype='band', fs=fs_aud_ds, output='sos')
    ka = sosfiltfilt(sos_aud, audio_ds)
    audio_env = np.abs(signal.hilbert(ka))
    sos_hr_a = butter(4, [0.4, 3.0], btype='band', fs=fs_aud_ds, output='sos')
    dh_acoustic = sosfiltfilt(sos_hr_a, audio_env)
    
    # Compute HR inside [k_on, k_off]
    mask_rf = (t_rf >= k_on) & (t_rf <= k_off)
    t_stable_rf = t_rf[mask_rf]
    dh_stable_rf = dh[mask_rf]
    min_dist_rf = int(FS_RF * 0.5)
    peaks_rf, _ = signal.find_peaks(dh_stable_rf, distance=min_dist_rf, prominence=np.std(dh_stable_rf)*0.3)
    hr_rf = 60.0 / np.mean(np.diff(t_stable_rf[peaks_rf])) if len(peaks_rf) > 1 else 0.0
    
    mask_st = (t_aud >= k_on) & (t_aud <= k_off)
    t_stable_st = t_aud[mask_st]
    dh_stable_st = dh_acoustic[mask_st]
    min_dist_st = int(fs_aud_ds * 0.5)
    peaks_st, _ = signal.find_peaks(dh_stable_st, distance=min_dist_st, prominence=np.std(dh_stable_st)*0.3)
    hr_st = 60.0 / np.mean(np.diff(t_stable_st[peaks_st])) if len(peaks_st) > 1 else 0.0
    
    print(f"k_on={k_on}, k_off={k_off}")
    print(f"  RF peaks detected: {len(peaks_rf)} -> HR: {hr_rf:.1f} BPM")
    print(f"  Steth peaks detected: {len(peaks_st)} -> HR: {hr_st:.1f} BPM")

print("Subject 1:")
test_hr(SUB1_RF, SUB1_WAV, k_on=27.53, k_off=43.33, lag=1.7083)

print("Subject 2:")
test_hr(SUB2_RF, SUB2_WAV, k_on=27.38, k_off=42.00, lag=2.6042)
