import h5py, os
import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── PUBLICATION STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.labelsize':    13,
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'legend.framealpha': 0.95,
    'lines.linewidth':   1.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,  # No grid to match user's clean image
})

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def process_subject(rf_path, k_on, k_off, ax, title):
    FS_RF = 10_000
    DEC = 10
    FS = FS_RF // DEC
    
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0, :], rf[1, :]
    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c  = i_raw - xc, q_raw - yc
    
    # Compute Magnitude
    sos_lp  = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    
    mag_ds = decimate(mag_raw, DEC, ftype='fir')
    t = np.arange(len(mag_ds)) / FS
    
    # Filter bands
    # Heartbeat: 0.4 - 3.0 Hz
    hb_band = decimate(bpf(mag_raw, 0.4, 3.0, FS_RF), DEC, ftype='fir')
    # Korotkoff snaps: 10 - 200 Hz
    koro_band = decimate(bpf(mag_raw, 10, 200, FS_RF), DEC, ftype='fir')
    
    # Normalization window (Entire active window + padding)
    zoom_start = k_on - 2.0
    zoom_end   = k_off + 1.0
    zoom_mask = (t >= zoom_start) & (t <= zoom_end)
    
    hb_norm = hb_band / np.max(np.abs(hb_band[zoom_mask]))
    # Scale korotkoff slightly smaller so it doesn't completely overwrite the heartbeat peak
    koro_norm = koro_band / (np.max(np.abs(koro_band[zoom_mask])) * 1.3)
    
    # Active Window Shading
    ax.axvspan(k_on, k_off, color='#FEF9EC', alpha=1.0, zorder=0, label='Active Window')
    
    # Horizontal line at 0
    ax.axhline(0, color='#FF8888', lw=1.0, zorder=1)
    
    # Plot signals
    ax.plot(t, hb_norm, color='black', lw=2.2, zorder=3, label='Heartbeat (0.4-3.0 Hz)')
    
    # Only plot Korotkoff snaps INSIDE the active duration
    act_mask_plot = (t >= k_on) & (t <= k_off)
    ax.plot(t[act_mask_plot], koro_norm[act_mask_plot], color='#FF5555', lw=1.2, alpha=0.9, zorder=4, label='Korotkoff Snaps (10-200 Hz)')
    
    # Formatting
    ax.set_xlim([zoom_start, zoom_end])
    ax.set_ylim([-1.15, 1.15])
    
    # Calculate Heart Rate (BPM) during the active window
    act_mask = (t >= k_on) & (t <= k_off)
    act_hb = hb_band[act_mask]
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(act_hb, distance=int(FS * 0.45)) # max ~133 bpm
    if len(peaks) > 1:
        hr = 60.0 / np.mean(np.diff(peaks) / FS)
    else:
        hr = 0.0
        
    title_hr = f'{title} | Est. HR: {hr:.0f} BPM'
    ax.set_title(title_hr)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Amplitude')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Put legend outside the plot box to keep the signal clean
    ax.legend(loc='upper right', framealpha=1.0)

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT = os.path.join(BASE, 'rf_zoomed_overlay_both_subjects.png')

fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, facecolor='#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# Subject 1
p1 = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
process_subject(p1, 27.7, 43.5, axes[0], '(A) Subject 1 (Prof. Kan) — Zoomed Overlay: Heartbeats vs. Korotkoff Snaps')

# Subject 2
p2 = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
process_subject(p2, 27.375, 42.0, axes[1], '(B) Subject 2 (Rajveer) — Zoomed Overlay: Heartbeats vs. Korotkoff Snaps')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
print("\nDONE:", OUT)
