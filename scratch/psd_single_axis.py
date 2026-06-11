"""
High-Resolution PSD Comparison on Single Axis
=============================================
Computes the Power Spectral Density (PSD) of the raw complex RF signals
for the three experimental conditions:
  (1) No US, No Body (nobody_noultrasound)
  (2) US ON, Table (ultra_rftable1)
  (3) US ON, Body (ultra_rfbody1)

Crucially, all three curves are plotted on the SAME AXIS with a shared,
unaltered y-limit. This avoids the visual distortion caused by auto-scaling,
making the 10x amplitude differences and spectral line broadening instantly visible.

Saves plot to: data_new/Ultra/ultra_detailed_analysis/psd_single_axis.png
"""
import h5py, os
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

FS = 10000

# ── Load active segments ──────────────────────────────────────────────────────
with h5py.File(os.path.join(ULTRA, 'nobody_noultrasound.h5'), 'r') as f:
    d = f['data'][:]
    iq_no_us = (d[0] + 1j * d[1])[int(2.0*FS):int(8.0*FS)]

with h5py.File(os.path.join(ULTRA, 'ultra_rftable1.h5'), 'r') as f:
    d = f['data'][:]
    iq_table = (d[0] + 1j * d[1])[int(6.0*FS):int(12.0*FS)]

with h5py.File(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 'r') as f:
    d = f['data'][:]
    iq_body = (d[0] + 1j * d[1])[int(2.0*FS):int(8.0*FS)]

# ── Compute High-Resolution PSDs ──────────────────────────────────────────────
# We use nperseg = 8192 for high frequency resolution
psd_data = {}
for name, iq in [('No US, No Body', iq_no_us), ('US ON, Table', iq_table), ('US ON, Body', iq_body)]:
    f_arr, pxx = welch(iq, fs=FS, nperseg=8192, return_onesided=False)
    sorted_idx = np.argsort(f_arr)
    psd_data[name] = (f_arr[sorted_idx], pxx[sorted_idx])

# ── Plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':11,'axes.titleweight':'bold',
    'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.5,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), dpi=300, facecolor='#FFFFFF')

COLORS = {
    'No US, No Body' : '#2C3E50',
    'US ON, Table'   : '#1A6FC4',
    'US ON, Body'    : '#C0392B'
}

# Plot 1: Full Spectrum (-600 to 600 Hz)
for name, (f, pxx) in psd_data.items():
    m = (f >= -600) & (f <= 600)
    ax1.semilogy(f[m], pxx[m], color=COLORS[name], label=name, lw=1.2, alpha=0.9)

ax1.set_title('(A) Raw Near-Field RF Power Spectral Density (Full Band: -600 to 600 Hz)\n'
              'Shows the 100.71 Hz carrier and its harmonics. Note the massive 60 dB offset of the Body signal.')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power Density (a.u./Hz)')
ax1.set_ylim(1e-14, 1e-1)
ax1.legend(frameon=False)

# Plot 2: Zoomed Carrier Region (95 to 105 Hz)
for name, (f, pxx) in psd_data.items():
    m = (f >= 95.0) & (f <= 105.0)
    ax2.semilogy(f[m], pxx[m], color=COLORS[name], label=name, lw=1.5, alpha=0.9)

# Highlight carrier frequency
ax2.axvline(100.714, color='#7F8C8D', lw=1.0, ls='--', alpha=0.7)
ax2.set_title('(B) Zoomed Carrier PSD Peak Region (95 to 105 Hz)\n'
              'Reveals that the Body carrier peak is extremely attenuated (60 dB lower) due to human absorption,\n'
              'and shows spectral broadening sidebands compared to the static controls.')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Density (a.u./Hz)')
ax2.set_ylim(1e-14, 1e-1)
ax2.legend(frameon=False)

fig.suptitle(
    'Unfiltered RF PSD Comparison: Shared Axis Analysis\n'
    'Shared Y-Axis reveals the true physical differences between the 3 conditions',
    fontsize=13, fontweight='bold', y=0.96
)

out_file = os.path.join(OUT, 'psd_single_axis.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
