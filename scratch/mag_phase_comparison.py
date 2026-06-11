"""
Ultrasound Effect on RF Magnitude and Phase: Verification Plot
=============================================================
Generates a 2-panel figure showing the active modulation of BOTH magnitude and phase
when the ultrasound is turned ON vs OFF on the body:
  - Left Panel: AC Magnitude Variance (ON vs OFF) -> 976.3x increase.
  - Right Panel: AC Phase Variance (ON vs OFF) -> 225.3x increase.

Saves plot to: data_new/Ultra/ultra_detailed_analysis/mag_phase_comparison.png
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
mag_off, phs_off = get_ac_signals(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 12.0, 15.0)
mag_on,  phs_on  = get_ac_signals(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 2.0, 5.0)

var_mag_off, var_phs_off = np.var(mag_off), np.var(phs_off)
var_mag_on,  var_phs_on  = np.var(mag_on), np.var(phs_on)

# ── Plotting ──
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':11,'axes.titleweight':'bold',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor='#FFFFFF')

# Panel 1: Magnitude Variance
bars1 = ax1.bar(['US OFF\n(Body)', 'US ON\n(Body)'], 
                [var_mag_off, var_mag_on], 
                color=['#95A5A6', '#C0392B'], width=0.45)
ax1.set_title('(A) RF Signal Magnitude Variance\n(AC amplitude above 10 Hz)')
ax1.set_ylabel('Magnitude Variance (a.u.)')
ax1.set_yscale('log')
ax1.grid(True, ls=':', alpha=0.6, axis='y')

ratio_mag = var_mag_on / var_mag_off
ax1.text(0.5, np.sqrt(var_mag_off * var_mag_on), f'{ratio_mag:.1f}x\nINCREASE', 
         ha='center', va='center', color='black', fontweight='bold', fontsize=12)

# Panel 2: Phase Variance
bars2 = ax2.bar(['US OFF\n(Body)', 'US ON\n(Body)'], 
                [var_phs_off, var_phs_on], 
                color=['#95A5A6', '#D35400'], width=0.45)
ax2.set_title('(B) RF Signal Phase Variance\n(AC phase above 10 Hz)')
ax2.set_ylabel('Phase Variance (rad²)')
ax2.set_yscale('log')
ax2.grid(True, ls=':', alpha=0.6, axis='y')

ratio_phs = var_phs_on / var_phs_off
ax2.text(0.5, np.sqrt(var_phs_off * var_phs_on), f'{ratio_phs:.1f}x\nINCREASE', 
         ha='center', va='center', color='black', fontweight='bold', fontsize=12)

fig.suptitle(
    'Acousto-RF Dual Modulation Proof: Magnitude vs. Phase\n'
    'Ultrasound pressure waves modulate BOTH the RF magnitude (AM) and RF phase (PM)',
    fontsize=13, fontweight='bold', y=0.98
)

plt.tight_layout()
out_file = os.path.join(OUT, 'mag_phase_comparison.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
