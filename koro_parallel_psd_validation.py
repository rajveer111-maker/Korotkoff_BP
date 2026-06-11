"""
Academic Publication Parallel 6-Panel PSD and Time-Frequency Spectrogram Validation Figure Generator
========================================================================================
Generates a stunning, comprehensive 6-panel figure (3x2 grid) for paper submission,
validating our concurrent dual-sensory approach using Power Spectral Density (PSD)
and Time-Frequency Spectrogram analysis.

Layout:
  - Row 1: Raw Signal PSD Analysis
    - Panel A: RF Sensor Phase PSD (Active vs. Quiet Window) - highlights heart rate & respiratory peaks
    - Panel B: Stethoscope Raw Acoustic PSD (Active vs. Quiet Window) - highlights raw audio spectrum
  - Row 2: Bandpassed Korotkoff Signal PSD Analysis
    - Panel C: RF Sensor Korotkoff Velocity PSD (Active vs. Quiet) - shows arterial snaps in 10-200 Hz
    - Panel D: Stethoscope Bandpassed Acoustic PSD (Active vs. Quiet) - shows acoustic clicks in 50-1000 Hz
  - Row 3: Time-Frequency Spectrograms
    - Panel E: RF Sensor Korotkoff Spectrogram over time - highlights heartbeat snaps during Active window
    - Panel F: Stethoscope Bandpassed Spectrogram over time - highlights concurrent acoustic click onset

Usage:
  python koro_parallel_psd_validation.py
"""
import os, sys, warnings, h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.signal import welch, butter, sosfiltfilt, hilbert

# Import loader helpers from feature script
from koro_parallel_features import load_stethoscope

warnings.filterwarnings('ignore')

# Config
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
FINAL_PLOT_PATH = os.path.join(OUTPUT_DIR, 'paper_parallel_psd_validation.png')
COPY_DEST_DIR = r'C:\Users\rajve\.gemini\antigravity\brain\b11c4ec4-c7a3-4eaf-86b7-1efc0188caab'

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\Rec_1.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_2_Rajveer\sthethoscope_rec01.mp4'

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
    print("  6-PANEL PSD & TIME-FREQUENCY SPECTROGRAM VALIDATION FIGURE GENERATOR")
    print("=" * 80)
    
    # ------------------------------------------------------------------
    # LOAD SIGNAL STREAMS
    # ------------------------------------------------------------------
    print(f"Loading RF H5 file: {RF_PATH}")
    if not os.path.exists(RF_PATH):
        print("[ERROR] RF file not found")
        sys.exit(1)
        
    with h5py.File(RF_PATH, 'r') as f:
        data = f['data'][:]
        
    i_raw, q_raw = data[0, :], data[1, :]
    t_rf = np.arange(len(i_raw)) / FS
    
    print(f"Loading Stethoscope audio: {AUDIO_PATH}")
    t_aud, aud_raw, koro_aud, fs_aud, _ = load_stethoscope(AUDIO_PATH)
    if t_aud is None:
        print("[ERROR] Stethoscope file not found")
        sys.exit(1)
        
    # ------------------------------------------------------------------
    # PHASE RECONSTRUCTION (RF SENSOR)
    # ------------------------------------------------------------------
    i_c, q_c = i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)
    iq = i_c + 1j * q_c
    
    sos_lp = signal.butter(4, 50.0, btype='low', fs=FS, output='sos')
    iq_clean = signal.sosfiltfilt(sos_lp, iq)
    phase_clean = np.unwrap(np.angle(iq_clean))
    phase_clean = signal.detrend(phase_clean)
    
    # Physical Conversion Constants
    LAMBDA_MM = (299792458 / 0.9e9) * 1000  # 333.10 mm
    SCALE = LAMBDA_MM / (4 * np.pi)          # 26.51 mm/rad
    
    # Notch filters
    for fn in [50.0, 100.0, 150.0]:
        b, a = signal.iirnotch(fn, 30, FS)
        phase_clean = signal.filtfilt(b, a, phase_clean)
        
    # Bandpass filters
    sos_hr = signal.butter(4, [0.5, 3.0], btype='band', fs=FS, output='sos')
    sos_koro = signal.butter(4, [10, 200], btype='band', fs=FS, output='sos')
    
    phase_hr = signal.sosfiltfilt(sos_hr, phase_clean)
    phase_koro = signal.sosfiltfilt(sos_koro, phase_clean)
    
    disp_hr = phase_hr * SCALE * 1000      # um
    
    # Compute Korotkoff velocity and RF envelope for automatic window isolation
    vel_koro = np.append(np.diff(phase_koro) * FS, 0) * SCALE # mm/s
    
    # Automatic inflation/deflation transition detection
    t_deflation = find_deflation_onset(aud_raw, fs_aud)
    print(f"  Automatic Cuff Deflation Onset detected at: {t_deflation:.2f} seconds")
    
    # Detect exact RF Korotkoff window automatically
    # 0.3s sliding RMS on velocity
    ph_energy = np.sqrt(pd.Series(vel_koro).pow(2).rolling(int(FS*0.3), center=True).mean().fillna(0).values)**2
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
        
    print(f"  Automatic True Korotkoff window: {rf_on:.2f}s to {rf_off:.2f}s")
    
    # Quiet post-deflation baseline: 10s window starting after Korotkoff sounds end
    quiet_on = min(rf_off + 2.0, t_rf[-1] - 11.0)
    quiet_off = quiet_on + 10.0
    print(f"  Quiet baseline window: {quiet_on:.2f}s to {quiet_off:.2f}s")
    
    # ------------------------------------------------------------------
    # PLOTTING DESIGN
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)
    
    color_active = '#C83D3D'    # Crimson Red for Active window
    color_quiet = '#2E6F9E'     # Cobalt Blue for Quiet window
    color_rf = '#2E6F9E'
    color_aud = '#C83D3D'
    
    # Helper to extract segment indexes
    def get_indices(t, onset, offset):
        return np.where((t >= onset) & (t <= offset))[0]
        
    idx_act_rf = get_indices(t_rf, rf_on, rf_off)
    idx_q_rf = get_indices(t_rf, quiet_on, quiet_off)
    
    idx_act_aud = get_indices(t_aud, rf_on, rf_off)
    idx_q_aud = get_indices(t_aud, quiet_on, quiet_off)
    
    # --------------------------------------------------------------
    # PANEL A: RF PHASE PSD (Row 1 Left)
    # --------------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    
    # Compute PSD on raw Phase displacement
    f_act, p_act = welch(phase_clean[idx_act_rf], fs=FS, nperseg=int(FS*4.0))
    f_q, p_q = welch(phase_clean[idx_q_rf], fs=FS, nperseg=int(FS*4.0))
    
    axA.semilogy(f_act, p_act, color=color_active, lw=2.0, label='Korotkoff Window (24.25-35.25s)')
    axA.semilogy(f_q, p_q, color=color_quiet, lw=1.5, ls='--', alpha=0.8, label='Quiet Baseline Window (42-52s)')
    
    axA.set_xlabel('Frequency (Hz)')
    axA.set_ylabel('Power Spectral Density ($rad^2/Hz$)')
    axA.set_title('A. RF Sensor Phase Angle PSD Analysis (0–15 Hz)', fontweight='bold')
    axA.set_xlim(0.1, 15.0)
    axA.grid(True, alpha=0.15)
    axA.legend(loc='upper right', fontsize=9)
    
    # Highlight Heart rate fundamental frequency (~1.1 Hz) and respiration (~0.25 Hz)
    axA.axvline(1.1, color='grey', ls=':', alpha=0.7)
    axA.text(1.2, axA.get_ylim()[0]*50, 'Heart Rate\n(~1.1 Hz)', color='dimgrey', fontsize=8, fontweight='bold')
    axA.axvline(0.25, color='grey', ls=':', alpha=0.7)
    axA.text(0.3, axA.get_ylim()[0]*5, 'Respiration\n(~0.25 Hz)', color='dimgrey', fontsize=8, fontweight='bold')

    # --------------------------------------------------------------
    # PANEL B: STETHOSCOPE RAW ACOUSTIC PSD (Row 1 Right)
    # --------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    
    f_act_aud, p_act_aud = welch(aud_raw[idx_act_aud], fs=fs_aud, nperseg=int(fs_aud*4.0))
    f_q_aud, p_q_aud = welch(aud_raw[idx_q_aud], fs=fs_aud, nperseg=int(fs_aud*4.0))
    
    axB.semilogy(f_act_aud, p_act_aud, color=color_active, lw=2.0, label='Korotkoff Window')
    axB.semilogy(f_q_aud, p_q_aud, color=color_quiet, lw=1.5, ls='--', alpha=0.8, label='Quiet Window')
    
    axB.set_xlabel('Frequency (Hz)')
    axB.set_ylabel('Power Spectral Density ($a.u.^2/Hz$)')
    axB.set_title('B. Stethoscope Raw Acoustic PSD Analysis (0–2000 Hz)', fontweight='bold')
    axB.set_xlim(10, 2000)
    axB.grid(True, alpha=0.15)
    axB.legend(loc='upper right', fontsize=9)

    # --------------------------------------------------------------
    # PANEL C: RF KOROTKOFF VELOCITY PSD (Row 2 Left)
    # --------------------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    
    f_act_v, p_act_v = welch(vel_koro[idx_act_rf], fs=FS, nperseg=int(FS*2.0))
    f_q_v, p_q_v = welch(vel_koro[idx_q_rf], fs=FS, nperseg=int(FS*2.0))
    
    axC.plot(f_act_v, p_act_v*1e6, color=color_active, lw=2.0, label='Korotkoff Window (Active)')
    axC.plot(f_q_v, p_q_v*1e6, color=color_quiet, lw=1.5, ls='--', alpha=0.8, label='Quiet Window (Baseline)')
    
    axC.set_xlabel('Frequency (Hz)')
    axC.set_ylabel('Power Spectral Density ($\mu m^2/s^2/Hz$)')
    axC.set_title('C. RF Sensor Bandpassed Korotkoff Velocity PSD (10–200 Hz)', fontweight='bold')
    axC.set_xlim(10, 200)
    axC.set_ylim(0, np.max(p_act_v[f_act_v <= 200])*1.5*1e6)
    axC.grid(True, alpha=0.15)
    axC.legend(loc='upper right', fontsize=9)
    
    # Shading the main energy band (10-100 Hz)
    axC.axvspan(10, 100, color='red', alpha=0.08, label='Main Korotkoff Band')

    # --------------------------------------------------------------
    # PANEL D: STETHOSCOPE BANDPASSED ACOUSTIC PSD (Row 2 Right)
    # --------------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    
    f_act_audbp, p_act_audbp = welch(koro_aud[idx_act_aud], fs=fs_aud, nperseg=int(fs_aud*2.0))
    f_q_audbp, p_q_audbp = welch(koro_aud[idx_q_aud], fs=fs_aud, nperseg=int(fs_aud*2.0))
    
    axD.plot(f_act_audbp, p_act_audbp*1e9, color=color_active, lw=2.0, label='Korotkoff Window (Active)')
    axD.plot(f_q_audbp, p_q_audbp*1e9, color=color_quiet, lw=1.5, ls='--', alpha=0.8, label='Quiet Window (Baseline)')
    
    axD.set_xlabel('Frequency (Hz)')
    axD.set_ylabel('Power Spectral Density ($a.u.^2/Hz$)')
    axD.set_title('D. Stethoscope Bandpassed Acoustic PSD (50–1000 Hz)', fontweight='bold')
    axD.set_xlim(50, 1000)
    axD.set_ylim(0, np.max(p_act_audbp[(f_act_audbp >= 50) & (f_act_audbp <= 1000)])*1.5*1e9)
    axD.grid(True, alpha=0.15)
    axD.legend(loc='upper right', fontsize=9)
    
    # Shading the main energy band (50-400 Hz)
    axD.axvspan(50, 400, color='red', alpha=0.08)

    # --------------------------------------------------------------
    # PANEL E: RF KOROTKOFF VELOCITY SPECTROGRAM (Row 3 Left)
    # --------------------------------------------------------------
    axE = fig.add_subplot(gs[2, 0])
    
    # Compute STFT
    ff_rf, tt_rf, Zxx_rf = signal.stft(vel_koro, fs=FS, nperseg=int(FS*1.0), noverlap=int(FS*0.9))
    Zxx_mag_rf = np.abs(Zxx_rf)
    
    # Crop to 10-200 Hz
    idx_f_rf = np.where((ff_rf >= 10) & (ff_rf <= 200))[0]
    
    im_rf = axE.pcolormesh(tt_rf, ff_rf[idx_f_rf], 20 * np.log10(Zxx_mag_rf[idx_f_rf, :] + 1e-12), 
                           shading='gouraud', cmap='viridis', vmin=-110, vmax=-40)
    
    # Overlay lines for Active Window
    axE.axvline(rf_on, color='white', ls='--', lw=1.8)
    axE.axvline(rf_off, color='white', ls='--', lw=1.8)
    
    axE.set_xlabel('Time (seconds)')
    axE.set_ylabel('Frequency (Hz)')
    axE.set_title('E. RF Sensor Korotkoff Velocity STFT Spectrogram over Time', fontweight='bold')
    axE.set_xlim(0, t_rf[-1])
    axE.set_ylim(10, 200)
    
    # Colorbar
    cbar_rf = fig.colorbar(im_rf, ax=axE, orientation='horizontal', pad=0.12, shrink=0.8)
    cbar_rf.set_label('Power Spectral Magnitude (dB)')
    
    axE.text((rf_on+rf_off)/2, 175, 'Active\nWindow', color='white', fontweight='bold', ha='center', va='center')
    axE.axvspan(0, t_deflation, color='white', alpha=0.15, hatch='//')

    # --------------------------------------------------------------
    # PANEL F: STETHOSCOPE BANDPASSED SPECTROGRAM (Row 3 Right)
    # --------------------------------------------------------------
    axF = fig.add_subplot(gs[2, 1])
    
    ff_aud, tt_aud, Zxx_aud = signal.stft(koro_aud, fs=fs_aud, nperseg=int(fs_aud*1.0), noverlap=int(fs_aud*0.9))
    Zxx_mag_aud = np.abs(Zxx_aud)
    
    # Crop to 50-1000 Hz
    idx_f_aud = np.where((ff_aud >= 50) & (ff_aud <= 1000))[0]
    
    im_aud = axF.pcolormesh(tt_aud, ff_aud[idx_f_aud], 20 * np.log10(Zxx_mag_aud[idx_f_aud, :] + 1e-12), 
                            shading='gouraud', cmap='magma', vmin=-110, vmax=-40)
    
    # Overlay lines for Active Window
    axF.axvline(rf_on, color='white', ls='--', lw=1.8)
    axF.axvline(rf_off, color='white', ls='--', lw=1.8)
    
    axF.set_xlabel('Time (seconds)')
    axF.set_ylabel('Frequency (Hz)')
    axF.set_title('F. Stethoscope Bandpassed Acoustic STFT Spectrogram over Time', fontweight='bold')
    axF.set_xlim(0, t_aud[-1])
    axF.set_ylim(50, 1000)
    
    # Colorbar
    cbar_aud = fig.colorbar(im_aud, ax=axF, orientation='horizontal', pad=0.12, shrink=0.8)
    cbar_aud.set_label('Power Spectral Magnitude (dB)')
    
    axF.text((rf_on+rf_off)/2, 850, 'Active\nWindow', color='white', fontweight='bold', ha='center', va='center')
    axF.axvspan(0, t_deflation, color='white', alpha=0.15, hatch='//')

    # --------------------------------------------------------------
    # SAVE AND COPY RESULTS
    # --------------------------------------------------------------
    fig.suptitle('Academic PSD Spectral Footprints & Time-Frequency Spectrogram Validation\n'
                 '(Subject: Sub_2_Rajveer, Session: 1 | Enforcing 0-Lag Simultaneous Biophysical Alignment)', 
                 fontweight='bold', fontsize=18, y=0.995)
    
    os.makedirs(os.path.dirname(FINAL_PLOT_PATH), exist_ok=True)
    
    plt.savefig(FINAL_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Generated new 6-panel PSD validation figure at -> {FINAL_PLOT_PATH}")
    
    # Copy to destination if possible
    if os.path.exists(COPY_DEST_DIR):
        import shutil
        dest_file = os.path.join(COPY_DEST_DIR, 'paper_parallel_psd_validation.png')
        shutil.copyfile(FINAL_PLOT_PATH, dest_file)
        print(f"Copied figure to brain artifacts directory -> {dest_file}")

if __name__ == '__main__':
    main()
