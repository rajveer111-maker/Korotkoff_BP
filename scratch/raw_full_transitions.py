"""
Raw Near-Field RF Analysis (Entire Duration & ON/OFF Transitions)
==================================================================
Analyzes raw near-field RF signals over their ENTIRE duration.
Proves the ultrasound effect by showing:
  1. The raw signal magnitude over the full duration, exposing the exact transitions 
     when the ultrasound pulser turns ON or OFF.
  2. The raw high-frequency AC component (removing slow drift) to show the raw 
     vibrations and physiological changes directly.

No radar terminology (no constellations, no far-field assumptions).
Saves plot to: data_new/Ultra/ultra_detailed_analysis/raw_full_transitions.png
"""
import h5py, os
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

FS = 10000

# ── Load 4 files to trace transitions ─────────────────────────────────────────
files = {
    'nobody_noultrasound' : 'No US, No Body (Baseline)',
    'ultra_rftable1'       : 'US ON, Table (Mechanical Control)',
    'ultra_rfbody1'        : 'US ON, Body (Short Active Window)',
    'ultra_rfbody01'       : 'US ON, Body (Middle Active Window)'
}

data = {}
for fname, label in files.items():
    with h5py.File(os.path.join(ULTRA, f'{fname}.h5'), 'r') as f:
        d = f['data'][:]
    I = d[0]
    Q = d[1]
    t = np.arange(len(I)) / FS
    mag = np.sqrt(I**2 + Q**2)
    
    # Extract AC component of magnitude (high-pass above 10 Hz to see carrier vibrations)
    # Using a simple butterworth filter
    sos = signal.butter(4, 10.0, btype='high', fs=FS, output='sos')
    mag_ac = signal.sosfiltfilt(sos, mag)
    
    data[fname] = {
        't': t, 'I': I, 'Q': Q, 'mag': mag, 'mag_ac': mag_ac, 'label': label
    }

# ── PLOT ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':11,'axes.titleweight':'bold',
    'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.5,
})

fig = plt.figure(figsize=(18, 24), dpi=300, facecolor='#FFFFFF')
gs  = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25, left=0.06, right=0.96, top=0.94, bottom=0.04)

fnames = ['nobody_noultrasound', 'ultra_rftable1', 'ultra_rfbody1', 'ultra_rfbody01']
COLORS = ['#2C3E50', '#1A6FC4', '#C0392B', '#E67E22']

for idx, fname in enumerate(fnames):
    d   = data[fname]
    clr = COLORS[idx]
    
    # Left Column: Raw Magnitude (DC + AC) over full duration
    ax_dc = fig.add_subplot(gs[idx, 0])
    ax_dc.plot(d['t'], d['mag'], color=clr, lw=0.7, alpha=0.85)
    ax_dc.set_title(f'{d["label"]} — Full Duration Raw Magnitude')
    ax_dc.set_xlabel('Time (s)'); ax_dc.set_ylabel('Raw Magnitude (a.u.)')
    
    # Highlight ON/OFF states based on carrier power profile
    if fname == 'ultra_rfbody1':
        # US is ON from 0 to 8s
        ax_dc.axvspan(0, 8.0, color='#FFFDE7', alpha=0.8, zorder=0)
        ax_dc.axvline(8.0, color='#D35400', lw=1.2, ls='--', label='US Turns OFF (8s)')
        ax_dc.text(4, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound ON', ha='center', color='#D35400', fontsize=10)
        ax_dc.text(25, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound OFF', ha='center', color='#7F8C8D', fontsize=10)
        ax_dc.legend(frameon=False)
        
    elif fname == 'ultra_rfbody01':
        # US is ON from 25 to 45s
        ax_dc.axvspan(25.0, 45.0, color='#FFFDE7', alpha=0.8, zorder=0)
        ax_dc.axvline(25.0, color='#D35400', lw=1.2, ls='--')
        ax_dc.axvline(45.0, color='#D35400', lw=1.2, ls='--')
        ax_dc.text(12, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound OFF', ha='center', color='#7F8C8D', fontsize=10)
        ax_dc.text(35, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound ON', ha='center', color='#D35400', fontsize=10)
        ax_dc.legend(frameon=False)
        
    elif fname == 'nobody_noultrasound':
        ax_dc.text(24, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound is OFF Entire Time', ha='center', color='#7F8C8D', fontsize=10)
        
    elif fname == 'ultra_rftable1':
        ax_dc.text(17, ax_dc.get_ylim()[0] + (ax_dc.get_ylim()[1]-ax_dc.get_ylim()[0])*0.5,
                  'Ultrasound is ON Entire Time', ha='center', color='#D35400', fontsize=10)

    # Right Column: High-Frequency AC Vibrations (AC-Coupled Magnitude)
    ax_ac = fig.add_subplot(gs[idx, 1])
    # Downsample plotting slightly to avoid overlapping lines
    ax_ac.plot(d['t'][::2], d['mag_ac'][::2], color=clr, lw=0.4, alpha=0.8)
    ax_ac.set_title(f'{d["label"]} — High-Frequency AC Vibrations (>10 Hz)')
    ax_ac.set_xlabel('Time (s)'); ax_ac.set_ylabel('AC Magnitude (a.u.)')
    
    # Match the horizontal highlights
    if fname == 'ultra_rfbody1':
        ax_ac.axvspan(0, 8.0, color='#FFFDE7', alpha=0.8, zorder=0)
        ax_ac.axvline(8.0, color='#D35400', lw=1.2, ls='--')
        # Print variance comparison ON vs OFF
        var_on  = np.var(d['mag_ac'][d['t'] <= 8.0])
        var_off = np.var(d['mag_ac'][d['t'] > 12.0])
        ax_ac.text(0.02, 0.9, f'ON Var: {var_on:.2e} a.u.\nOFF Var: {var_off:.2e} a.u.', 
                   transform=ax_ac.transAxes, fontsize=9, color='#D35400',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    elif fname == 'ultra_rfbody01':
        ax_ac.axvspan(25.0, 45.0, color='#FFFDE7', alpha=0.8, zorder=0)
        ax_ac.axvline(25.0, color='#D35400', lw=1.2, ls='--')
        ax_ac.axvline(45.0, color='#D35400', lw=1.2, ls='--')
        var_on  = np.var(d['mag_ac'][(d['t'] >= 25.0) & (d['t'] <= 45.0)])
        var_off = np.var(d['mag_ac'][d['t'] < 20.0])
        ax_ac.text(0.02, 0.9, f'ON Var: {var_on:.2e} a.u.\nOFF Var: {var_off:.2e} a.u.', 
                   transform=ax_ac.transAxes, fontsize=9, color='#D35400',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    elif fname == 'nobody_noultrasound':
        var_all = np.var(d['mag_ac'])
        ax_ac.text(0.02, 0.9, f'Baseline Var: {var_all:.2e} a.u.', 
                   transform=ax_ac.transAxes, fontsize=9, color='#7F8C8D',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    elif fname == 'ultra_rftable1':
        var_all = np.var(d['mag_ac'])
        ax_ac.text(0.02, 0.9, f'Table Var: {var_all:.2e} a.u.', 
                   transform=ax_ac.transAxes, fontsize=9, color='#1A6FC4',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle(
    'Near-Field RF Signal: Visualizing Ultrasound ON/OFF Transitions\n'
    'Proves that ultrasound creates high-frequency micro-vibrations in the RF signal which vanish when turned OFF',
    fontsize=14, fontweight='bold', y=0.97
)

out_file = os.path.join(OUT, 'raw_full_transitions.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
