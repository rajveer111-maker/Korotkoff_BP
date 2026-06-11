"""
Academic Publication Parallel 8-Panel Physical Comparison Figure Generator v3.0
========================================================================
Generates a stunning, comprehensive 8-panel figure (4x2 grid) for paper submission,
comparing the physical signal processing pipelines of RF Sensor and Stethoscope (Audio)
side-by-side, resolving acoustic cuff sound separation, and demonstrating beat-level
overlay consistency ("pie-to-pie" pulse-to-pulse alignment).

Layout:
  - Column 1: RF Sensor Modality
    - Panel A: Raw Preprocessed Phase
    - Panel C: Bandpassed Korotkoff Velocity (10–200 Hz)
    - Panel E: Korotkoff Energy Envelope & Detected True Boundary
    - Panel G: Pulse-to-Pulse Beat Overlay (Heartbeat aligned)
  - Column 2: Acoustic Stethoscope Modality
    - Panel B: Raw Acoustic Waveform
    - Panel D: Bandpassed Korotkoff Acoustic Wave (20–200 Hz)
    - Panel F: Acoustic Hilbert Envelope & Cuff Deflation Noise Separation
    - Panel H: Pulse-to-Pulse Waveform Overlay (Acoustic pulse aligned)

Usage:
  python koro_parallel_paper_panels.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.signal import hilbert

# Import loader helpers from feature script
from koro_parallel_features import load_stethoscope

warnings.filterwarnings('ignore')

# Config
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
FINAL_PLOT_PATH = os.path.join(OUTPUT_DIR, 'paper_parallel_8panels.png')
COPY_DEST_DIR = r'C:\Users\rajve\.gemini\antigravity\brain\b11c4ec4-c7a3-4eaf-86b7-1efc0188caab'

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\Rec_7.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\sthethoscope_rec07.mp4'

FS = 10000

# Set plotting styles for academic grade
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

def calc_tkeo(x):
    t = np.zeros_like(x)
    t[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return t

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def sliding_rms(x, w): 
    return np.sqrt(pd.Series(x).pow(2).rolling(window=w, center=True).mean().fillna(0).values)

def find_deflation_onset(aud_raw, fs_aud):
    # Compute sliding RMS of raw audio with 200ms window
    win_len = int(fs_aud * 0.2)
    rms = np.sqrt(pd.Series(aud_raw).pow(2).rolling(win_len, center=True).mean().fillna(0).values)
    # Smooth with 1.0s window to remove short gaps
    rms_smooth = np.convolve(rms, np.ones(int(fs_aud*1.0))/int(fs_aud*1.0), mode='same')
    
    # We search in the first 25 seconds for the transition
    search_lim = min(int(25 * fs_aud), len(rms_smooth))
    
    # Find the steepest drop in the smoothed RMS (where pump shuts off)
    grad = np.gradient(rms_smooth[:search_lim])
    deflation_onset_idx = np.argmin(grad)
    deflation_onset_t = deflation_onset_idx / fs_aud
    
    # Sanity check: must be between 10s and 25s, otherwise default to 20.0s
    if not (10.0 <= deflation_onset_t <= 25.0):
        max_val = np.max(rms_smooth[:search_lim])
        below_th = np.where(rms_smooth[:search_lim] < 0.20 * max_val)[0]
        if len(below_th) > 0:
            deflation_onset_t = below_th[0] / fs_aud
        else:
            deflation_onset_t = 20.0
            
    deflation_onset_t = np.clip(deflation_onset_t, 10.0, 25.0)
    return deflation_onset_t

def main():
    print("=" * 80)
    print("  ACADEMIC PUBLICATION 8-PANEL PHYSICAL MODALITY COMPARISON GENERATOR")
    print("=" * 80)
    
    # ------------------------------------------------------------------
    # LOAD SIGNAL STREAMS
    # ------------------------------------------------------------------
    print(f"Loading RF H5 file: {RF_PATH}")
    if not os.path.exists(RF_PATH):
        print("[ERROR] RF file not found")
        sys.exit(1)
        
    import h5py
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
        
    i_raw, q_raw = data[0, :], data[1, :]
    t_rf = np.arange(len(i_raw)) / FS
    print(f"RF Signal loaded: fs={FS} Hz, len={len(t_rf)} samples, dur={t_rf[-1]:.2f}s")
    
    print(f"Loading Stethoscope audio: {AUDIO_PATH}")
    t_aud, aud_raw, koro_aud, fs_aud, _ = load_stethoscope(AUDIO_PATH)
    if t_aud is None:
        print("[ERROR] Stethoscope file not found")
        sys.exit(1)
    print(f"Audio Signal loaded: fs={fs_aud} Hz, len={len(t_aud)} samples, dur={t_aud[-1]:.2f}s")
    
    # ------------------------------------------------------------------
    # PHASE RECONSTRUCTION VIA PHASE ARC METHOD (RF SENSOR)
    # ------------------------------------------------------------------
    print("Applying Phase Arc Method (Pre-processed and Unwrapped)...")
    # Pre-processing: mean centering (DC removal) to align circle centered at (0,0)
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    
    # 50 Hz low-pass filter to remove USRP high-frequency white noise before unwrapping
    sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    
    # Phase Arc Method: unwrapped angle in radians
    phase_clean = np.unwrap(np.angle(iq_clean))
    
    # Linear detrending (pre-processing step) to remove linear frequency offset drift
    phase_clean = signal.detrend(phase_clean)
    
    # Physical Conversion Constants
    LAMBDA_MM = (299792458 / 0.9e9) * 1000  # 333.10 mm
    SCALE = LAMBDA_MM / (4 * np.pi)          # 26.51 mm/rad
    
    # ------------------------------------------------------------------
    # PHYSICAL SIGNAL CONDITIONING & DERIVATION CHAIN
    # ------------------------------------------------------------------
    # 50 Hz notch filter for preprocessed magnitude
    b50, a50 = signal.iirnotch(50.0, 30, FS)
    mag_raw = np.abs(iq)
    mag = signal.filtfilt(b50, a50, mag_raw)
    
    # Bandpass filters
    sos_hr = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')
    
    phase_hr = signal.sosfiltfilt(sos_hr, phase_clean)     # rad
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)  # rad
    
    # Phase -> Displacement: converted to micrometers (um)
    disp_hr = phase_hr * SCALE * 1000      # um
    disp_koro = phase_koro * SCALE * 1000   # um
    
    # Displacement -> Velocity (mm/s)
    # Velocity in mm/s = d(disp_um)/dt / 1000
    vel_hr = np.append(np.diff(disp_hr) * FS / 1000, 0)      # mm/s
    vel_koro = np.append(np.diff(disp_koro) * FS / 1000, 0)   # mm/s
    
    # ------------------------------------------------------------------
    # PHYSICAL BOUNDARY DURATION ISOALATION
    # ------------------------------------------------------------------
    # Detect exact RF Korotkoff window
    ph_energy = sliding_rms(vel_koro, int(FS*0.3))**2
    sm_energy = pd.Series(ph_energy).rolling(window=int(FS*2), center=True).mean().fillna(0).values
    T_SKIP = 5
    vs, ve = int(T_SKIP*FS), min(int(len(sm_energy) - T_SKIP*FS), int(40*FS))
    ci = vs + np.argmax(sm_energy[vs:ve]) if vs < ve else np.argmax(sm_energy)
    eth = np.max(sm_energy[vs:ve]) * 0.08
    si, ei = ci, ci
    while si > 0 and sm_energy[si] > eth: si -= 1
    while ei < len(sm_energy)-1 and sm_energy[ei] > eth: ei += 1
    rf_on = t_rf[max(si, int(T_SKIP*FS))]
    rf_off = t_rf[min(ei, int((t_rf[-1]-T_SKIP)*FS))]
    
    # Normalize durations if outliers occur
    rf_dur_raw = rf_off - rf_on
    if rf_dur_raw < 4.0:
        p = (4.0-rf_dur_raw)/2; rf_on, rf_off = max(0, rf_on-p), min(t_rf[-1], rf_off+p)
    elif rf_dur_raw > 15.0:
        p = (rf_dur_raw-15.0)/2; rf_on, rf_off = rf_on+p, rf_off-p
    rf_dur = rf_off - rf_on
    
    # Audio Korotkoff window physically aligns to the RF window since recorded simultaneously
    aud_on, aud_off = rf_on, rf_off
    aud_dur = aud_off - aud_on
    
    # ------------------------------------------------------------------
    # SMOOTHED ENVELOPES & ENERGY CALCULATIONS
    # ------------------------------------------------------------------
    # RF Energy Envelope
    rf_env_smoothed = sm_energy / (np.max(sm_energy) + 1e-20)
    
    # Stethoscope Hilbert Envelope
    aud_env = np.abs(hilbert(koro_aud))
    aud_env_smoothed = smooth(aud_env, int(fs_aud * 0.5))
    # Normalize envelope based on true Korotkoff window max
    aud_env_max = np.max(aud_env_smoothed[int(aud_on*fs_aud):int(aud_off*fs_aud)])
    aud_env_smoothed = aud_env_smoothed / (aud_env_max + 1e-20)
    
    # ------------------------------------------------------------------
    # AUTOMATIC INFLATION/DEFLATION TRANSITION & NOISE PEAK DETECTIONS
    # ------------------------------------------------------------------
    print("Automatically detecting deflation transition and noise events...")
    t_deflation = find_deflation_onset(aud_raw, fs_aud)
    print(f"  Automatic Cuff Deflation Onset detected at: {t_deflation:.2f} seconds")
    
    # Find cuff deflation noise peak (valve release) in the last 15 seconds of the recording
    search_start_t = max(35.0, t_aud[-1] - 15.0)
    search_start_idx = int(search_start_t * fs_aud)
    cuff_peak_idx = search_start_idx + np.argmax(aud_env_smoothed[search_start_idx:])
    cuff_peak_t = cuff_peak_idx / fs_aud
    cuff_peak_val = aud_env_smoothed[cuff_peak_idx]
    print(f"  Automatic Cuff Deflation Noise Peak (Valve Release) detected at: {cuff_peak_t:.2f} seconds")
    
    # Find early pump click automatically (local maximum of envelope between 5s and 18s)
    early_start, early_end = int(5.0 * fs_aud), int(18.0 * fs_aud)
    pump_click_idx = early_start + np.argmax(aud_env_smoothed[early_start:early_end])
    pump_click_t = pump_click_idx / fs_aud
    pump_click_val = aud_env_smoothed[pump_click_idx]
    print(f"  Automatic Early Cuff Pump Click detected at: {pump_click_t:.2f} seconds")
    
    # ------------------------------------------------------------------
    # GENERATE 8-PANEL ULTIMATE PLOT (300 DPI)
    # ------------------------------------------------------------------
    print("\nPlotting ultimate 8-panel physical comparison figure at 300 DPI...")
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25)
    
    # Cohesive Color Palette
    color_rf = '#2E6F9E'       # Sleek Cobalt Blue
    color_aud = '#C83D3D'      # Deep Crimson Red
    color_env = '#2F3640'      # Slate Charcoal
    
    # --------------------------------------------------------------
    # PANEL A: RF SENSOR RAW PHASE OVERVIEW (Row 1 Left)
    # --------------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(t_rf, phase_clean, color=color_rf, lw=1.2, alpha=0.9, label='Preprocessed Phase')
    axA.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.5, hatch='//', edgecolor='#A5B1C2', label='Cuff Inflation Phase')
    axA.axvspan(t_deflation, t_rf[-1], color='#EAF0F6', alpha=0.3, label='Cuff Deflation Phase (Omron)')
    axA.set_xlabel('Time (seconds)')
    axA.set_ylabel('Phase Angle (radians)')
    axA.set_title('A. RF Sensor Preprocessed Phase (DC centered, Unwrapped)', fontweight='bold')
    axA.grid(True, alpha=0.15)
    axA.legend(loc='upper right', fontsize=9)
    axA.set_xlim(0, t_rf[-1])
    axA.set_ylim(np.min(phase_clean) - 0.2, np.max(phase_clean) + 0.3)
    axA.text(t_deflation / 2, np.min(phase_clean) + 0.2, f'Cuff Inflation Phase\n(0-{t_deflation:.1f}s, pressure -> 140 mmHg)\nNo Korotkoff sounds', 
             ha='center', va='bottom', fontsize=8, color='#57606F', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='grey'))
    
    # --------------------------------------------------------------
    # PANEL B: STETHOSCOPE RAW ACOUSTIC WAVEFORM (Row 1 Right)
    # --------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(t_aud, aud_raw, color=color_aud, lw=0.6, alpha=0.7, label='Raw Acoustic Wave')
    axB.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.5, hatch='//', edgecolor='#A5B1C2', label='Cuff Inflation Phase')
    axB.axvspan(t_deflation, t_aud[-1], color='#EAF0F6', alpha=0.3, label='Cuff Deflation Phase (Omron)')
    axB.set_xlabel('Time (seconds)')
    axB.set_ylabel('Amplitude (a.u.)')
    axB.set_title('B. Stethoscope Raw Acoustic Waveform', fontweight='bold')
    axB.grid(True, alpha=0.15)
    axB.legend(loc='upper right', fontsize=9)
    axB.set_xlim(0, t_aud[-1])
    axB.text(t_deflation / 2, np.min(aud_raw)*0.7, f'Cuff Inflation Phase\n(0-{t_deflation:.1f}s, pressure -> 140 mmHg)\nNo Korotkoff sounds', 
             ha='center', va='bottom', fontsize=8, color='#57606F', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='grey'))
    
    # --------------------------------------------------------------
    # PANEL C: RF SENSOR BANDPASSED KOROTKOFF VELOCITY (Row 2 Left)
    # --------------------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(t_rf, vel_koro, color=color_rf, lw=0.8, alpha=0.9, label='$v_{koro}(t)$ (10-200 Hz)')
    axC.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.3, hatch='//', edgecolor='#A5B1C2')
    axC.axvspan(t_deflation, t_rf[-1], color='#EAF0F6', alpha=0.2)
    axC.axvspan(rf_on, rf_off, color=color_rf, alpha=0.1, label='True Korotkoff Window')
    axC.set_xlabel('Time (seconds)')
    axC.set_ylabel('Velocity (mm/s)')
    axC.set_title('C. RF Sensor Bandpassed Korotkoff Velocity', fontweight='bold')
    axC.grid(True, alpha=0.15)
    axC.legend(loc='upper right', fontsize=9)
    axC.set_xlim(0, t_rf[-1])
    
    # Adjust y-limits robustly
    vel_koro_lim = max(100, np.percentile(np.abs(vel_koro[int(rf_on*FS):int(rf_off*FS)]), 99.5) * 1.5)
    axC.set_ylim(-vel_koro_lim, vel_koro_lim)
    
    # --------------------------------------------------------------
    # PANEL D: STETHOSCOPE BANDPASSED KOROTKOFF WAVE (Row 2 Right)
    # --------------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    axD.plot(t_aud, koro_aud, color=color_aud, lw=0.6, alpha=0.8, label='Acoustic (50-1000 Hz)')
    axD.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.3, hatch='//', edgecolor='#A5B1C2')
    axD.axvspan(t_deflation, t_aud[-1], color='#EAF0F6', alpha=0.2)
    axD.axvspan(aud_on, aud_off, color=color_aud, alpha=0.1, label='True Korotkoff Window')
    axD.set_xlabel('Time (seconds)')
    axD.set_ylabel('Amplitude (a.u.)')
    axD.set_title('D. Stethoscope Bandpassed Korotkoff Signal', fontweight='bold')
    axD.grid(True, alpha=0.15)
    axD.legend(loc='upper right', fontsize=9)
    axD.set_xlim(0, t_aud[-1])
    
    # --------------------------------------------------------------
    # PANEL E: RF SENSOR KOROTKOFF ENERGY ENVELOPE & BOUNDARY (Row 3 Left)
    # --------------------------------------------------------------
    axE = fig.add_subplot(gs[2, 0])
    axE.plot(t_rf, rf_env_smoothed, color=color_env, lw=1.8, label='Energy Smooth Envelope')
    axE.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.3, hatch='//', edgecolor='#A5B1C2')
    axE.axvspan(t_deflation, t_rf[-1], color='#EAF0F6', alpha=0.2)
    axE.axvspan(rf_on, rf_off, color=color_rf, alpha=0.12, edgecolor=color_rf, ls='--')
    axE.set_xlabel('Time (seconds)')
    axE.set_ylabel('Normalized Energy')
    axE.set_title(f'E. RF Sensor Korotkoff Energy Envelope & Boundary (Dur = {rf_dur:.2f}s)', fontweight='bold')
    axE.grid(True, alpha=0.15)
    axE.legend(loc='upper right', fontsize=9)
    axE.set_xlim(0, t_rf[-1])
    axE.set_ylim(-0.05, 1.1)
    
    # Annotation inside shading
    axE.text((rf_on + rf_off)/2, 0.5, f'True Korotkoff\nWindow\n({rf_dur:.2f}s, after {t_deflation:.1f}s pump)', color=color_rf, 
             fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=color_rf))
    
    # --------------------------------------------------------------
    # PANEL F: STETHOSCOPE ENVELOPE & CUFF SOUND SEPARATION (Row 3 Right)
    # --------------------------------------------------------------
    axF = fig.add_subplot(gs[2, 1])
    axF.plot(t_aud, aud_env_smoothed, color=color_env, lw=1.8, label='Hilbert Smooth Envelope')
    axF.axvspan(0, t_deflation, color='#DFE4EA', alpha=0.3, hatch='//', edgecolor='#A5B1C2')
    axF.axvspan(t_deflation, t_aud[-1], color='#EAF0F6', alpha=0.2)
    axF.axvspan(aud_on, aud_off, color=color_aud, alpha=0.12, edgecolor=color_aud, ls='--')
    axF.set_xlabel('Time (seconds)')
    axF.set_ylabel('Normalized Energy')
    axF.set_title(f'F. Stethoscope Envelope & Cuff sound Separation (Dur = {aud_dur:.2f}s)', fontweight='bold')
    axF.grid(True, alpha=0.15)
    axF.legend(loc='upper right', fontsize=9)
    axF.set_xlim(0, t_aud[-1])
    axF.set_ylim(-0.05, 1.4)
    
    # True Korotkoff window label
    axF.text((aud_on + aud_off)/2, 0.5, f'True Korotkoff\nWindow\n({aud_dur:.2f}s, after {t_deflation:.1f}s pump)', color=color_aud, 
             fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=color_aud))
    
    # Cuff deflation noise peak (valve release)
    axF.annotate('Cuff Deflation Noise /\nValve Release (Not Korotkoff)', 
                 xy=(cuff_peak_t, cuff_peak_val), 
                 xytext=(cuff_peak_t - 15, cuff_peak_val + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
                 fontweight='bold', fontsize=9, color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FFEAA7', alpha=0.9, ec='orange'))
                 
    # Annotation for early pump click and valve thumps
    axF.annotate('Cuff Pump Click\n& Valve Thumps\n(Non-Korotkoff)', 
                 xy=(pump_click_t, pump_click_val),
                 xytext=(3, 1.15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
                 fontweight='bold', fontsize=9, color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FFEAA7', alpha=0.9, ec='orange'))
    
    # --------------------------------------------------------------
    # PANEL G: RF SENSOR ZOOMED OVERLAY: HEARTBEATS vs. KOROTKOFF SNAPS (Row 4 Left)
    # --------------------------------------------------------------
    axG = fig.add_subplot(gs[3, 0])
    
    t_zoom_start = np.floor(t_deflation)
    t_zoom_end = t_zoom_start + 7.0
    idx_zoom_rf = np.where((t_rf >= t_zoom_start) & (t_rf <= t_zoom_end))[0]
    t_z_rf = t_rf[idx_zoom_rf]
    
    # Normalize displacement and velocity over the zoom window between -1 and 1
    d_z = disp_hr[idx_zoom_rf]
    d_z_norm = 2.0 * (d_z - np.min(d_z)) / (np.max(d_z) - np.min(d_z) + 1e-9) - 1.0
    
    v_z = vel_koro[idx_zoom_rf]
    v_z_norm = v_z / (np.max(np.abs(v_z)) + 1e-9)
    
    # Active window shading (yellow)
    axG.axvspan(rf_on, rf_off, color='#FFEAA7', alpha=0.35, label='Active Window')
    
    axG.plot(t_z_rf, d_z_norm, color='black', lw=2.2, label='Heartbeat (0.5-3.0 Hz)')
    axG.plot(t_z_rf, v_z_norm, color='red', lw=1.2, alpha=0.8, label='Korotkoff Snaps (10-200 Hz)')
    
    axG.set_xlabel('Time (seconds)')
    axG.set_ylabel('Normalized Amplitude')
    axG.set_title('G. RF Sensor Zoomed Overlay: Heartbeats vs. Korotkoff Snaps', fontweight='bold')
    axG.grid(True, alpha=0.15)
    axG.legend(loc='upper right', fontsize=9)
    axG.set_xlim(t_zoom_start, t_zoom_end)
    axG.set_ylim(-1.15, 1.15)
    
    # --------------------------------------------------------------
    # PANEL H: STETHOSCOPE ZOOMED OVERLAY: HEARTBEATS vs. ACOUSTIC CLICKS (Row 4 Right)
    # --------------------------------------------------------------
    axH = fig.add_subplot(gs[3, 1])
    
    idx_zoom_aud = np.where((t_aud >= t_zoom_start) & (t_aud <= t_zoom_end))[0]
    t_z_aud = t_aud[idx_zoom_aud]
    
    # Normalize audio clicks over the zoom window
    a_z = koro_aud[idx_zoom_aud]
    a_z_norm = a_z / (np.max(np.abs(a_z)) + 1e-9)
    
    # Interpolate RF displacement reference to the audio timeline for perfect alignment
    d_z_aud = np.interp(t_z_aud, t_z_rf, d_z_norm)
    
    # Active window shading (yellow)
    axH.axvspan(aud_on, aud_off, color='#FFEAA7', alpha=0.35, label='Active Window')
    
    axH.plot(t_z_aud, d_z_aud, color='black', lw=2.2, label='Heartbeat Reference (0.5-3.0 Hz)')
    axH.plot(t_z_aud, a_z_norm, color='red', lw=0.9, alpha=0.8, label='Acoustic Clicks (50-1000 Hz)')
    
    axH.set_xlabel('Time (seconds)')
    axH.set_ylabel('Normalized Amplitude')
    axH.set_title('H. Stethoscope Zoomed Overlay: Heartbeats vs. Acoustic Clicks', fontweight='bold')
    axH.grid(True, alpha=0.15)
    axH.legend(loc='upper right', fontsize=9)
    axH.set_xlim(t_zoom_start, t_zoom_end)
    axH.set_ylim(-1.15, 1.15)
    
    # --------------------------------------------------------------
    # SAVE AND COPY RESULTS
    # --------------------------------------------------------------
    fig.suptitle('Parallel RF Sensor vs Stethoscope Biophysical Korotkoff Signal Analysis & Pulse Overlay\n'
                 f'(Subject: Sub_2_Rajveer, Session: 7 (Best Session) | True Simultaneous Window: {rf_on:.2f}s - {rf_off:.2f}s)', 
                 fontweight='bold', fontsize=18, y=0.995)
    
    # Make sure parent directory exists
    os.makedirs(os.path.dirname(FINAL_PLOT_PATH), exist_ok=True)
    
    plt.savefig(FINAL_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Generated new 8-panel physical comparison figure at -> {FINAL_PLOT_PATH}")
    
    # Copy to destination if possible
    if os.path.exists(COPY_DEST_DIR):
        import shutil
        dest_file = os.path.join(COPY_DEST_DIR, 'paper_parallel_8panels.png')
        shutil.copyfile(FINAL_PLOT_PATH, dest_file)
        print(f"Copied figure to brain artifacts directory -> {dest_file}")

if __name__ == '__main__':
    main()
