"""
RF Magnitude and Phase AC Variance: 3-Case Comparison
=====================================================
Generates a 2-panel figure comparing the high-frequency AC variance (>10 Hz)
of BOTH magnitude and phase across all three files, representing all states:
  (1) No US, No Body (US OFF Baseline)
  (2) US ON, Table (US ON Mechanical Control)
  (3) US ON, Body (US ON segment, 2-8s)
  (4) US ON, Body (US OFF segment, 12-18s)

Saves plot to: data_new/Ultra/ultra_detailed_analysis/mag_phase_3way.png
"""
import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

FS = 10000

def get_ac_signals(filepath, t0, t1):
    with h5py.File(filepath, 'r') as f:
        d = f['data'][:]
    iq = d[0] + 1j * d[1]
    
    iq_seg = iq[int(t0*FS):int(t1*FS)]
    mag = np.abs(iq_seg)
    phs = np.unwrap(np.angle(iq_seg))
    
    # 10 Hz highpass filter
    sos = butter(4, 10.0 / (0.5*FS), btype='high', output='sos')
    mag_ac = sosfiltfilt(sos, mag)
    phs_ac = sosfiltfilt(sos, phs)
    return mag_ac, phs_ac

# ── Load ──
mag_no,  phs_no  = get_ac_signals(os.path.join(ULTRA, 'nobody_noultrasound.h5'), 2.0, 8.0)
mag_tbl, phs_tbl = get_ac_signals(os.path.join(ULTRA, 'ultra_rftable1.h5'), 6.0, 12.0)
mag_b_on, phs_b_on = get_ac_signals(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 2.0, 7.0)
mag_b_off, phs_b_off = get_ac_signals(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 12.0, 17.0)

var_mag = [np.var(mag_no), np.var(mag_tbl), np.var(mag_b_on), np.var(mag_b_off)]
var_phs = [np.var(phs_no), np.var(phs_tbl), np.var(phs_b_on), np.var(phs_b_off)]

labels = [
    'No US, No Body\n(US OFF)',
    'US ON, Table\n(US ON)',
    'US Body (ON Window)\n(Active US)',
    'US Body (OFF Window)\n(Shielded Quiet)'
]

# ── Plotting ──
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,'font.weight':'bold',
    'axes.labelsize':10,'axes.labelweight':'bold',
    'axes.titlesize':10,'axes.titleweight':'bold',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300, facecolor='#FFFFFF')

# Panel 1: Magnitude Variance
colors = ['#2C3E50', '#1A6FC4', '#C0392B', '#7F8C8D']
bars1 = ax1.bar(labels, var_mag, color=colors, width=0.5)
ax1.set_title('(A) RF Signal Magnitude AC Variance (>10 Hz)')
ax1.set_ylabel('Variance (a.u.)')
ax1.set_yscale('log')
ax1.grid(True, ls=':', alpha=0.6, axis='y')

# Annotate values
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f'{yval:.2e}', 
             ha='center', va='bottom', fontsize=8, color='black')

# Panel 2: Phase Variance
bars2 = ax2.bar(labels, var_phs, color=colors, width=0.5)
ax2.set_title('(B) RF Signal Phase AC Variance (>10 Hz)')
ax2.set_ylabel('Variance (rad²)')
ax2.set_yscale('log')
ax2.grid(True, ls=':', alpha=0.6, axis='y')

# Annotate values
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f'{yval:.2e}', 
             ha='center', va='bottom', fontsize=8, color='black')

# Rotate x-axis labels slightly for readability
ax1.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_xticklabels(labels, rotation=15, ha='right')

fig.suptitle(
    'RF Magnitude & Phase Variance: Complete 3-Case Comparison\n'
    'Compares open-air controls against active body sensing and self-shielded body baseline',
    fontsize=13, fontweight='bold', y=0.98
)

plt.tight_layout()
out_file = os.path.join(OUT, 'mag_phase_3way.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
