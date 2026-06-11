"""
Raw RF Signal Basic Property Analysis (No Advanced Filtering / Demodulation)
=============================================================================
Analyzes the raw I and Q channels, raw magnitude, and raw phase for:
  (1) nobody_noultrasound (No US, No Body)
  (2) ultra_rftable1 (US ON, Table / No Body)
  (3) ultra_rfbody1 (US ON, Body)

Computes basic statistics:
  - Raw In-phase (I) and Quadrature (Q) ranges, means, and standard deviations.
  - Raw Magnitude (A = sqrt(I^2 + Q^2)) mean, variance, and kurtosis.
  - Raw Phase (theta = atan2(Q, I)) standard deviation, range, and kurtosis.
  - Full-band Power Spectral Density (PSD) of the raw complex signal (I + jQ).

Saves plots to: data_new/Ultra/ultra_detailed_analysis/raw_basic_comparison.png
"""
import h5py, os
import numpy as np
from scipy import stats
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

FS = 10000 # 10 kHz

# ── Load data ─────────────────────────────────────────────────────────────────
files = {
    'No US, No Body'   : 'nobody_noultrasound.h5',
    'US ON, Table'     : 'ultra_rftable1.h5',
    'US ON, Body'      : 'ultra_rfbody1.h5'
}

data = {}
for label, fname in files.items():
    with h5py.File(os.path.join(ULTRA, fname), 'r') as f:
        d = f['data'][:]
    I = d[0]
    Q = d[1]
    raw_iq = I + 1j * Q
    
    # Calculate basic time-domain magnitude and phase
    mag   = np.abs(raw_iq)
    phase = np.angle(raw_iq)
    
    # Save raw arrays
    data[label] = {
        'I': I, 'Q': Q, 'mag': mag, 'phase': phase, 'raw_iq': raw_iq
    }

# ── Compute Basic Statistics ──────────────────────────────────────────────────
print("\n" + "="*80)
print("RAW RF BASIC PROPERTIES COMPARISON (NO FILTERING)")
print("="*80)

for label, d in data.items():
    I, Q, mag, phase = d['I'], d['Q'], d['mag'], d['phase']
    
    print(f"\n[{label}]")
    print(f"  Samples: {len(I)}  |  Duration: {len(I)/FS:.1f} s")
    print(f"  Raw I  : Mean={np.mean(I):.4f}, Std={np.std(I):.4f}, Range=[{np.min(I):.4f}, {np.max(I):.4f}]")
    print(f"  Raw Q  : Mean={np.mean(Q):.4f}, Std={np.std(Q):.4f}, Range=[{np.min(Q):.4f}, {np.max(Q):.4f}]")
    print(f"  Raw Mag: Mean={np.mean(mag):.4f}, Std={np.std(mag):.4f}, Var={np.var(mag):.2e}, Kurtosis={stats.kurtosis(mag, fisher=False):.2f}")
    print(f"  Raw Phs: Mean={np.mean(phase):.4f}, Std={np.std(phase):.4f}, Var={np.var(phase):.2e}, Kurtosis={stats.kurtosis(phase, fisher=False):.2f}")

# ── Plotting 6-Panel Basic Signal Properties Figure (No Constellations) ───────
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,'font.weight':'bold',
    'axes.labelsize':10,'axes.labelweight':'bold',
    'axes.titlesize':10,'axes.titleweight':'bold',
    'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.5,
})

fig = plt.figure(figsize=(18, 26), dpi=300, facecolor='#FFFFFF')
gs  = gridspec.GridSpec(5, 3, hspace=0.38, wspace=0.3, left=0.06, right=0.96, top=0.94, bottom=0.04)

COLORS = {
    'No US, No Body'   : '#2C3E50',
    'US ON, Table'     : '#1A6FC4',
    'US ON, Body'      : '#C0392B'
}

col_idx = 0
for label, d in data.items():
    I, Q, mag, phase, raw_iq = d['I'], d['Q'], d['mag'], d['phase'], d['raw_iq']
    clr = COLORS[label]
    t   = np.arange(len(I)) / FS
    
    # 1. Raw Time-Domain I & Q (first 100 ms)
    ax = fig.add_subplot(gs[0, col_idx])
    t_zoom = t[t <= 0.1]
    I_zoom = I[t <= 0.1]
    Q_zoom = Q[t <= 0.1]
    ax.plot(t_zoom * 1000, I_zoom, color=clr, label='I', lw=1.2)
    ax.plot(t_zoom * 1000, Q_zoom, color=clr, alpha=0.5, label='Q', lw=1.2)
    ax.set_title(f'Raw Time-Domain (Zoom: 100 ms)\n{label}')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (a.u.)')
    ax.legend(frameon=False)
    
    # 2. Raw Magnitude (zoom: 2 seconds)
    ax = fig.add_subplot(gs[1, col_idx])
    t_zoom2 = t[(t >= 2.0) & (t <= 4.0)]
    mag_zoom = mag[(t >= 2.0) & (t <= 4.0)]
    ax.plot(t_zoom2, mag_zoom, color=clr, lw=0.8)
    ax.set_title(f'Raw Magnitude Amplitude (2s Window)\n{label}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Magnitude (a.u.)')
    
    # 3. Raw Phase (zoom: 2 seconds)
    ax = fig.add_subplot(gs[2, col_idx])
    phase_zoom = phase[(t >= 2.0) & (t <= 4.0)]
    ax.plot(t_zoom2, phase_zoom, color=clr, lw=0.8)
    ax.set_title(f'Raw Phase Angle (2s Window)\n{label}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Phase (rad)')
    
    # Compute high-resolution PSD over the entire file
    # We use a large nperseg (8192) for high frequency resolution (1.2 Hz bins)
    f_arr, pxx = welch(raw_iq, fs=FS, nperseg=8192, return_onesided=False)
    sorted_idx = np.argsort(f_arr)
    f_arr, pxx = f_arr[sorted_idx], pxx[sorted_idx]
    
    # 4. Full-band raw Power Spectral Density (PSD)
    ax = fig.add_subplot(gs[3, col_idx])
    m_full = (f_arr >= -600) & (f_arr <= 600)
    ax.semilogy(f_arr[m_full], pxx[m_full], color=clr, lw=1.0)
    ax.set_title(f'Raw Signal PSD (-600 to 600 Hz)\n{label}')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power Density (a.u./Hz)')
    
    # 5. Zoomed PSD around Carrier Peak (95 to 105 Hz)
    ax = fig.add_subplot(gs[4, col_idx])
    m_zoom = (f_arr >= 95.0) & (f_arr <= 105.0)
    ax.semilogy(f_arr[m_zoom], pxx[m_zoom], color=clr, lw=1.2)
    ax.set_title(f'Zoomed PSD at Carrier (95–105 Hz)\n{label}')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power Density (a.u./Hz)')
    
    # Add carrier annotation
    if label != 'No US, No Body':
        ax.axvline(100.714, color='#D35400', lw=1.0, ls='--', alpha=0.7)
    
    col_idx += 1

fig.suptitle(
    'Raw RF Signal Properties: 3-Condition Basic Comparison\n'
    'No Filtering, No Demodulation, Pure Raw In-phase (I) and Quadrature (Q) Signal Properties',
    fontsize=14, fontweight='bold', y=0.97
)

out_file = os.path.join(OUT, 'raw_basic_comparison.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
