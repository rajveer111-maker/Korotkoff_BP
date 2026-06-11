import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert, butter, sosfiltfilt
from koro_parallel_features import load_stethoscope

rf_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
audio_path = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\korotoff_audio_stethoscope.mp4'

FS = 10000

def sliding_rms(x, w): return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)
def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

# Load RF
with h5py.File(rf_path, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = data[0,:], data[1,:]
time = np.arange(len(i_raw)) / FS

# Process RF Phase (User's high-fidelity decoupled phase)
i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
iq = i_c + 1j * q_c
sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
iq_clean = signal.sosfiltfilt(sos_lp, iq)

idx_deflation = int(20.0 * FS)
phase_unwrap_def = np.unwrap(np.angle(iq_clean[idx_deflation:]))
dphi_def = np.diff(phase_unwrap_def)
carrier_offset = np.median(dphi_def)
dphi_clean_def = dphi_def - carrier_offset
dphi_clean_def = np.clip(dphi_clean_def, -0.5, 0.5)
phase_clean_def = np.insert(np.cumsum(dphi_clean_def), 0, 0)

t_idx = np.arange(len(phase_clean_def))
t_norm = (t_idx - np.mean(t_idx)) / (np.max(t_idx) - np.min(t_idx) + 1e-9)
poly_def = np.polyfit(t_norm, phase_clean_def, 2)
phase_clean_def = phase_clean_def - np.polyval(poly_def, t_norm)

phase_clean_inf = np.angle(iq_clean[:idx_deflation])
phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(FS*1.0), center=True).mean().bfill().ffill().values
shift = phase_clean_def[0] - phase_clean_inf[-1]
phase_clean_inf = phase_clean_inf + shift

phase_clean = np.zeros(len(iq))
phase_clean[:idx_deflation] = phase_clean_inf
phase_clean[idx_deflation:] = phase_clean_def

LAMBDA_MM = (299792458 / 0.9e9) * 1000
SCALE = LAMBDA_MM / (4 * np.pi)

sos_hr = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')

phase_hr = signal.sosfiltfilt(sos_hr, phase_clean)
phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)

disp_hr = phase_hr * SCALE * 1000
disp_koro = phase_koro * SCALE * 1000
vel_koro = np.append(np.diff(disp_koro) * FS / 1000, 0)

# Koro window on RF
ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
T_SKIP = 5
vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS))
ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
eth = np.max(sm_energy[vs:ve]) * 0.08
si, ei = ci, ci
while si > 0 and sm_energy[si] > eth: si -= 1
while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
on_s, off_s = time[max(si, int(T_SKIP*FS))], time[min(ei, int((time[-1]-T_SKIP)*FS))]
print(f"Detected RF Koro Window: {on_s:.2f}s - {off_s:.2f}s, dur: {off_s-on_s:.2f}s")

# Load Audio
t_aud, aud_raw, koro_aud, fs_aud = load_stethoscope(audio_path)
print(f"Audio Loaded: dur={t_aud[-1]:.2f}s, fs={fs_aud}")

# Find audio peaks to see Korotkoff window
aud_env = np.abs(hilbert(koro_aud))
aud_env_smoothed = np.convolve(aud_env, np.ones(int(fs_aud * 0.5))/(fs_aud * 0.5), mode='same')
from scipy.signal import find_peaks
peaks, _ = find_peaks(aud_env_smoothed, distance=int(fs_aud * 1.0), prominence=np.max(aud_env_smoothed)*0.03)
print("Audio Peaks:")
for p in peaks:
    print(f"  Peak at {t_aud[p]:.2f}s, value: {aud_env_smoothed[p]:.5f}")
