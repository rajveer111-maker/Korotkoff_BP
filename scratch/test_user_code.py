import h5py, numpy as np, os, pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

FILE_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\rec_koro_sthe.h5'
FS = 10000
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\scratch\test_user_koro_rec_koro_sthe.png'

def sliding_rms(x, w): return np.sqrt(pd.Series(x).pow(2).rolling(window=w).mean().fillna(0).values)
def sliding_kurtosis(x, w): return pd.Series(x).rolling(window=w).kurt().fillna(0).values
def calc_tkeo(x):
    t = np.zeros_like(x); t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]; return t

def run():
    print(f"Loading H5 file: {FILE_PATH}")
    if not os.path.exists(FILE_PATH): print("File not found"); return
    with h5py.File(FILE_PATH, 'r') as f: data = f['data'][:]
    i_raw, q_raw = data[0,:], data[1,:]
    time = np.arange(len(i_raw)) / FS
    N = len(i_raw)

    # === SIGNAL PROCESSING ===
    # Center the IF circle
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    # 50 Hz low-pass filter to remove USRP high-frequency white noise before unwrapping
    sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    mag_raw = np.abs(iq)

    # Phase processing: Decoupled Phase Reconstruction to isolate inflation and deflation
    idx_deflation = int(20.0 * FS)
    
    # 1. Process the deflation period (t >= 20s) with absolute high-fidelity precision
    phase_unwrap_def = np.unwrap(np.angle(iq_clean[idx_deflation:]))
    dphi_def = np.diff(phase_unwrap_def)
    carrier_offset = np.median(dphi_def)  # rad/sample
    dphi_clean_def = dphi_def - carrier_offset
    dphi_clean_def = np.clip(dphi_clean_def, -0.5, 0.5)
    phase_clean_def = np.insert(np.cumsum(dphi_clean_def), 0, 0)
    # Detrend using a 2nd-order polynomial to remove residual warm-up drift (use normalized index for absolute numerical precision)
    t_idx = np.arange(len(phase_clean_def))
    t_norm = (t_idx - np.mean(t_idx)) / (np.max(t_idx) - np.min(t_idx) + 1e-9)
    poly_def = np.polyfit(t_norm, phase_clean_def, 2)
    phase_clean_def = phase_clean_def - np.polyval(poly_def, t_norm)
    
    # 2. Process the inflation period (0-20s) with raw wrapped phase (perfect noise, no steps!)
    phase_clean_inf = np.angle(iq_clean[:idx_deflation])
    # Zero-center the inflation noise using a 1-second rolling mean
    phase_clean_inf = phase_clean_inf - pd.Series(phase_clean_inf).rolling(window=int(FS*1.0), center=True).mean().bfill().ffill().values
    
    # Shift phase_clean_inf so that its end aligns perfectly with the start of phase_clean_def
    shift = phase_clean_def[0] - phase_clean_inf[-1]
    phase_clean_inf = phase_clean_inf + shift
    
    # 3. Combine both into a single continuous, step-free phase trace
    phase_clean = np.zeros(len(iq))
    phase_clean[:idx_deflation] = phase_clean_inf
    phase_clean[idx_deflation:] = phase_clean_def

    # Physical conversion
    LAMBDA_MM = (299792458 / 0.9e9) * 1000  # 333.10 mm
    SCALE = LAMBDA_MM / (4 * np.pi)          # 26.51 mm/rad

    # 50 Hz notch
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag = signal.filtfilt(b50, a50, mag_raw)

    # ---- DERIVATION CHAIN ----
    sos_hr = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')

    phase_hr = signal.sosfiltfilt(sos_hr, phase_clean)     # rad
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)  # rad

    # Phase -> Displacement: converted to micrometers (um)
    disp_hr = phase_hr * SCALE * 1000      # um
    disp_koro = phase_koro * SCALE * 1000   # um

    # Displacement -> Velocity (mm/s)
    # Since disp is in um, velocity in mm/s = d(disp_um)/dt / 1000
    vel_hr = np.append(np.diff(disp_hr) * FS / 1000, 0)      # mm/s
    vel_koro = np.append(np.diff(disp_koro) * FS / 1000, 0)   # mm/s

    # Other signals
    hr_mag = signal.sosfiltfilt(sos_hr, mag)

    # Koro window detection
    vel_tkeo = calc_tkeo(vel_koro)
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
    dur = off_s - on_s
    if dur < 4.0:
        p = (4.0-dur)/2; on_s, off_s = max(0, on_s-p), min(time[-1], off_s+p)
    elif dur > 15.0:
        p = (dur-15.0)/2; on_s, off_s = on_s+p, off_s-p
    dur = off_s - on_s

    print(f"{'='*55}")
    print(f"  Koro Window  : {on_s:.2f}s - {off_s:.2f}s ({dur:.1f}s)")
    print(f"{'='*55}")

if __name__ == '__main__':
    run()
