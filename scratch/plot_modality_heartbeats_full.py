import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

# ── GLOBAL CONSTANTS ─────────────────────────────────────────────────
FS_RF     = 10_000
FC_HZ     = 0.9e9
C_LIGHT   = 299792458.0
LAMBDA_MM = (C_LIGHT / FC_HZ) * 1000      # ~333.1 mm
SCALE     = LAMBDA_MM / (4 * np.pi)        # ~26.5 mm/rad

BASE = r"D:\Bioview\My_RF_work_v1\data_new\data_latest"
SUMMARY_DIR = os.path.join(BASE, "Multi_Subject_Summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

CSV_REPORT = os.path.join(SUMMARY_DIR, 'cross_subject_report.csv')

SUBJECT_CONFIGS = [
    {
        "name": "Prof_Kan",
        "label": "Prof. Kan (Sub 1)",
        "color": "#E84393",
        "folder": os.path.join(BASE, "Sub_1_Prof_kan"),
        "best_rec": 6,
        "steth_file": "sthethoscope_rec06.wav"
    },
    {
        "name": "Rajveer",
        "label": "Rajveer (Sub 2)",
        "color": "#2196F3",
        "folder": os.path.join(BASE, "Sub_2_Rajveer"),
        "best_rec": 4,
        "steth_file": "sthethoscope_rec04.wav"
    }
]

def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k) / k, mode='same')

def b210_iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1 = np.mean(ic**2); p2 = np.mean(qc**2); p3 = np.mean(ic * qc)
    sp = np.clip(p3 / np.sqrt(p1 * p2 + 1e-20), -1, 1)
    cp = np.sqrt(max(1 - sp**2, 1e-10))
    al = np.sqrt(p2 / (p1 + 1e-20))
    i_new = ic
    q_new = (qc - ic * sp / al) / cp
    return i_new + 1j * q_new

def detect_cuff_max_pressure_point(i_raw, q_raw, fs=10000.0, onset_limit=None):
    iq = -i_raw + 1j * q_raw
    sos_hp = butter(4, 5.0, btype='highpass', fs=fs, output='sos')
    iq_hp = sosfiltfilt(sos_hp, iq)
    energy = np.abs(iq_hp)
    
    ds = int(fs / 100)
    t_ds = np.arange(len(i_raw))[::ds] / fs
    energy_ds = energy[::ds]
    
    w_size = 100
    energy_smooth = np.convolve(energy_ds, np.ones(w_size)/w_size, mode='same')
    
    max_search_sec = 25.0
    if onset_limit is not None:
        max_search_sec = min(max_search_sec, onset_limit - 1.0)
    
    search_mask = t_ds <= max_search_sec
    if not np.any(search_mask):
        return 8.0
        
    t_search = t_ds[search_mask]
    e_search = energy_smooth[search_mask]
    
    peak_idx = np.argmax(e_search)
    peak_val = e_search[peak_idx]
    
    end_val = np.mean(energy_smooth[max(0, int(max_search_sec*100)-50):int(max_search_sec*100)])
    
    if peak_val < 5.0e-3 or (peak_val / (end_val + 1e-20)) < 3.0:
        return 0.0
        
    baseline = np.median(e_search[peak_idx:])
    threshold = baseline + 0.10 * (peak_val - baseline)
    
    t_det = 8.0
    for i in range(peak_idx, len(t_search)):
        if np.all(e_search[i:i+150] < threshold):
            t_det = t_search[i]
            break
            
    return t_det

def process_modality_signals(sub, onset, offset):
    # ── 1. PROCESS RF DATA ──
    h5_path = os.path.join(sub['folder'], f"Rec_{sub['best_rec']}.h5")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
    i_raw, q_raw = data[0], data[1]
    t_rf = np.arange(len(i_raw)) / FS_RF
    
    t_start = detect_cuff_max_pressure_point(i_raw, q_raw, fs=FS_RF, onset_limit=onset)
    idx_def = int(t_start * FS_RF) if t_start > 0.5 else int(8.0 * FS_RF)
    
    iq     = b210_iq_condition(-i_raw + 1j * q_raw)
    sos_lp = butter(4, 50.0, btype='low', fs=FS_RF, output='sos')
    iq_c   = sosfiltfilt(sos_lp, iq)
    
    puw = np.unwrap(np.angle(iq_c[idx_def:]))
    dp  = np.diff(puw)
    dp -= np.median(dp)
    dp  = np.clip(dp, -0.5, 0.5)
    ph_def = np.insert(np.cumsum(dp), 0, 0.0)
    
    ph_inf  = np.angle(iq_c[:idx_def])
    w_size = min(int(FS_RF), idx_def)
    if w_size >= 10:
        ph_inf -= (pd.Series(ph_inf).rolling(w_size, center=True)
                   .mean().bfill().ffill().values)
    if len(ph_inf) > 0:
        ph_inf += ph_def[0] - ph_inf[-1]
        phase_clean = np.concatenate([ph_inf, ph_def])
    else:
        phase_clean = ph_def
        
    sos_h = butter(4, [0.4, 3.0], btype='band', fs=FS_RF, output='sos')
    dh_rf = sosfiltfilt(sos_h, phase_clean) * SCALE * 0.1
    dh_rf_env = smooth(np.abs(hilbert(dh_rf)), int(1.5 * FS_RF))
    
    # RF MAP peak
    mid_start   = onset + 0.15 * (offset - onset)
    mid_end     = offset - 0.15 * (offset - onset)
    mid_mask_rf = (t_rf >= mid_start) & (t_rf <= mid_end)
    if np.any(mid_mask_rf):
        t_map_rf = t_rf[mid_mask_rf][np.argmax(dh_rf_env[mid_mask_rf])]
    else:
        t_map_rf = t_rf[(t_rf >= onset) & (t_rf <= offset)][np.argmax(dh_rf_env[(t_rf >= onset) & (t_rf <= offset)])]
        
    # ── 2. PROCESS STETHOSCOPE DATA ──
    wav_path = os.path.join(sub['folder'], sub['steth_file'])
    fs_a, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t_a = np.arange(len(audio)) / fs_a
    
    sos_a = butter(4, [20, 150], btype='band', fs=fs_a, output='sos')
    audio_filt = sosfiltfilt(sos_a, audio)
    audio_env = np.abs(hilbert(audio_filt))
    
    sos_ah = butter(4, [0.4, 3.0], btype='band', fs=fs_a, output='sos')
    dh_acoustic = sosfiltfilt(sos_ah, audio_env)
    dh_a_env = smooth(np.abs(hilbert(dh_acoustic)), int(1.5 * fs_a))
    
    # Acoustic MAP peak
    mid_mask_a = (t_a >= mid_start) & (t_a <= mid_end)
    if np.any(mid_mask_a):
        t_map_acoustic = t_a[mid_mask_a][np.argmax(dh_a_env[mid_mask_a])]
    else:
        t_map_acoustic = t_a[(t_a >= onset) & (t_a <= offset)][np.argmax(dh_a_env[(t_a >= onset) & (t_a <= offset)])]
        
    return {
        "t_rf": t_rf,
        "dh_rf": dh_rf,
        "t_a": t_a,
        "dh_acoustic": dh_acoustic,
        "t_start": t_start,
        "t_map_rf": t_map_rf,
        "t_map_steth": t_map_acoustic,
        "fs_a": fs_a
    }
 
def main():
    print("=" * 80)
    # Load report CSV
    df = pd.read_csv(CSV_REPORT)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=False)
    plt.subplots_adjust(hspace=0.20)
    
    for i, sub in enumerate(SUBJECT_CONFIGS):
        ax = axes[i]
        match = df[(df['subject'] == sub['label']) & (df['rec'] == sub['best_rec'])]
        if match.empty:
            print(f"Skipping {sub['label']} due to missing CSV row.")
            continue
            
        onset = float(match.iloc[0]['rf_onset'])
        offset = float(match.iloc[0]['rf_offset'])
        
        # Process and extract signals
        sig = process_modality_signals(sub, onset, offset)
        
        t_rf = sig['t_rf']
        dh_rf = sig['dh_rf']
        t_a = sig['t_a']
        dh_acoustic = sig['dh_acoustic']
        t_start = sig['t_start']
        t_map_rf = sig['t_map_rf']
        t_map_steth = sig['t_map_steth']
        fs_a = sig['fs_a']
        
        # Normalize whole signals (z-score) to make them directly comparable on the same Y-axis
        idx_rf_active = (t_rf >= onset) & (t_rf <= offset)
        idx_steth_active = (t_a >= onset) & (t_a <= offset)
        
        # Use active window std and mean for normalizing to preserve relative amplitude differences
        rf_mean, rf_std = np.mean(dh_rf[idx_rf_active]), np.std(dh_rf[idx_rf_active])
        steth_mean, steth_std = np.mean(dh_acoustic[idx_steth_active]), np.std(dh_acoustic[idx_steth_active])
        
        rf_norm = (dh_rf - rf_mean) / (rf_std + 1e-20)
        steth_norm = (dh_acoustic - steth_mean) / (steth_std + 1e-20)
        
        # ── TIME AXIS LOCKING & PREPENDING (Deflation onset locked at 20.0s physical time) ──
        t_shift = 0.0
        t_start_phys = t_start
        
        target_sbp = 125.0
        target_dbp = 75.0
        csv_abp_path = os.path.join(SUMMARY_DIR, 'cross_subject_abp_report.csv')
        if os.path.exists(csv_abp_path):
            df_abp = pd.read_csv(csv_abp_path)
            match_abp = df_abp[(df_abp['subject'] == sub['label']) & (df_abp['rec'] == sub['best_rec'])]
            if not match_abp.empty:
                target_sbp = float(match_abp.iloc[0]['sbp_rf'])
                target_dbp = float(match_abp.iloc[0]['dbp_rf'])
                
        t_sbp_phys = onset
        t_dbp_phys = offset
        
        sub_xlim = [0.0, t_dbp_phys + 5.0]
        
        t_rf_phys = t_rf
        rf_norm_full = rf_norm
        
        t_a_phys = t_a
        steth_norm_full = steth_norm
        
        # Plot full waveforms (downsampled slightly for plotting performance)
        ds = 5
        ax.plot(t_rf_phys[::ds], rf_norm_full[::ds], color='#2196F3', lw=1.2, alpha=0.9, label='RF Cardiac Displacement (RMG)')
        ax.plot(t_a_phys[::ds], steth_norm_full[::ds], color='#4CAF50', lw=1.0, alpha=0.75, ls='--', label='Stethoscope Acoustic Heartbeat (PCG)')
        
        # Add 3-zone physiological shading
        ax.axvspan(t_shift,      t_sbp_phys,  color='#DFE6E9', alpha=0.30, zorder=0)
        ax.axvspan(t_sbp_phys,   t_dbp_phys,  color='#FFEAA7', alpha=0.25, zorder=0)
        ax.axvspan(t_dbp_phys,   sub_xlim[1], color='#D1F2D9', alpha=0.30, zorder=0)
        
        # Add vertical dashed lines for SBP (onset) and DBP (offset)
        ax.axvline(t_sbp_phys, color='red', ls='--', lw=1.8, zorder=5)
        ax.text(t_sbp_phys - 0.3, 3.0, f'SBP (SYS)\nt = {t_sbp_phys:.2f}s', color='red', fontsize=9, fontweight='bold', ha='right')
        
        ax.axvline(t_dbp_phys, color='blue', ls='--', lw=1.8, zorder=5)
        ax.text(t_dbp_phys + 0.3, 3.0, f'DBP (DIA)\nt = {t_dbp_phys:.2f}s', color='blue', fontsize=9, fontweight='bold', ha='left')
        
        # Add compliance MAP peaks
        t_map_rf_phys = t_map_rf + t_shift
        ax.axvline(t_map_rf_phys, color='orange', ls=':', lw=1.5, zorder=5)
        ax.plot(t_map_rf_phys, rf_norm_full[np.argmin(np.abs(t_rf_phys - t_map_rf_phys))], '^', color='orange', ms=12, mec='black', mew=1.0, zorder=6, label=f'RF Compliance Peak (MAP) at t={t_map_rf_phys:.2f}s')
        
        t_map_steth_phys = t_map_steth + t_shift
        ax.axvline(t_map_steth_phys, color='gold', ls=':', lw=1.5, zorder=5)
        ax.plot(t_map_steth_phys, steth_norm_full[np.argmin(np.abs(t_a_phys - t_map_steth_phys))], '*', color='gold', ms=14, mec='black', mew=1.0, zorder=6, label=f'Steth Compliance Peak (MAP) at t={t_map_steth_phys:.2f}s')
        
        # Phase Text Annotations
        ymin, ymax = -4.2, 4.2
        ax.set_ylim([ymin, ymax])
        ax.text((t_shift + t_sbp_phys) / 2,        3.7, 'OCCLUDED\n(Phase I)',    ha='center', va='top', fontsize=9, color='#636E72', style='italic', fontweight='bold')
        ax.text((t_sbp_phys + t_dbp_phys) / 2,     3.7, 'KOROTKOFF\n(Phase II)',  ha='center', va='top', fontsize=9, color='#B7950B', style='italic', fontweight='bold')
        ax.text((t_dbp_phys + sub_xlim[1]) / 2,    3.7, 'UNOCCLUDED\n(Phase III)',ha='center', va='top', fontsize=9, color='#1E8449', style='italic', fontweight='bold')
        
        ax.set_title(f'{sub["label"]} — Whole Recording ({sub_xlim[0]:.1f}s to {sub_xlim[1]:.1f}s physical time) Modality Waveform Correlation', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Normalized Amplitude (z-score)', fontsize=11)
        ax.set_xlim(sub_xlim)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc='lower left', ncol=2, framealpha=0.9)
        
    axes[1].set_xlabel('Physical Time (s)', fontsize=11)
    
    # Bottom legend patches for physiological phases
    ph_patches = [
        mpatches.Patch(color='#DFE6E9', alpha=0.8, label='Phase I: Brachial Artery Fully Occluded (inflation & early deflation)'),
        mpatches.Patch(color='#FFEAA7', alpha=0.8, label='Phase II: Brachial Artery Compliance Window (Korotkoff heartbeat sounds)'),
        mpatches.Patch(color='#D1F2D9', alpha=0.8, label='Phase III: Brachial Artery Fully Unoccluded (Continuous laminar flow)'),
    ]
    fig.legend(handles=ph_patches, loc='lower center', ncol=1, fontsize=9.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
               
    fig.suptitle(
        'Dual-Modality Heartbeat Waveforms Correlation Over the Entire Timeline\n'
        'Direct Visual Comparison of RF Radiomyography vs. Stethoscope Phonocardiography  |  300 DPI Publication Quality',
        fontsize=15, fontweight='bold', color='#2C3E50', y=0.96)
        
    out_img = os.path.join(SUMMARY_DIR, "test_heartbeats_alignment_full.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Full recording timeline plot successfully saved to: {out_img}")
    
    # Copy to artifacts folder
    art_img = r"C:\Users\rajve\.gemini\antigravity\brain\46b248dc-1c7d-48de-9d0e-3389ddbb40e3\test_heartbeats_alignment_full.png"
    import shutil
    shutil.copy2(out_img, art_img)
    print(f"Full modality heartbeats plot copied to artifacts: {art_img}")

if __name__ == '__main__':
    main()
