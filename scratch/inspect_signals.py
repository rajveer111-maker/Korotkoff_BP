import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.signal import hilbert
import pandas as pd
from koro_parallel_features import load_rf, load_stethoscope, SESSIONS

rep_session = [s for s in SESSIONS if s['session_name'] == 'Sub_1_Prof_kan_Session_1'][0]
t_rf, vel_koro, disp_hr, phase, fs_rf = load_rf(rep_session['rf'])
t_aud, aud_raw, koro_aud, fs_aud = load_stethoscope(rep_session['audio'])

# Calculate smoothed envelopes
def sliding_rms(x, w):
    return np.sqrt(pd.Series(x).pow(2).rolling(w, center=True).mean().fillna(0).values)

rf_env = np.abs(hilbert(vel_koro))
rf_env_smoothed = np.convolve(rf_env, np.ones(int(fs_rf * 0.5))/(fs_rf * 0.5), mode='same')

aud_env = np.abs(hilbert(koro_aud))
aud_env_smoothed = np.convolve(aud_env, np.ones(int(fs_aud * 0.5))/(fs_aud * 0.5), mode='same')

print("RF Env Max at index:", np.argmax(rf_env_smoothed), "time:", t_rf[np.argmax(rf_env_smoothed)])
print("Audio Env Max at index:", np.argmax(aud_env_smoothed), "time:", t_aud[np.argmax(aud_env_smoothed)])

# Find peaks on Audio Env
from scipy.signal import find_peaks
peaks, _ = find_peaks(aud_env_smoothed, distance=int(fs_aud * 1.0), prominence=np.max(aud_env_smoothed)*0.05)
print("Audio Env peaks at times:")
for p in peaks:
    print(f"  Peak at {t_aud[p]:.2f}s, value: {aud_env_smoothed[p]:.5f}")
