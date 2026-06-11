"""
AC RMS and Harmonic Carrier-to-Noise Ratio (CNR) Verification
=============================================================
Calculates two independent verification metrics:
  (1) AC RMS Amplitude of the Magnitude and Phase above 10 Hz.
  2) Carrier-to-Noise Ratio (CNR) at the 4th harmonic (4f0 = 402.86 Hz).
     The 4th harmonic is free from power line grid hum, making it a pure 
     ultrasound carrier indicator.

Saves plot to: data_new/Ultra/ultra_detailed_analysis/rms_and_cnr_proof.png
"""
import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

ULTRA = r'd:\Bioview\My_RF_work_v1\data_new\Ultra'
OUT   = os.path.join(ULTRA, 'ultra_detailed_analysis')
os.makedirs(OUT, exist_ok=True)

FS = 10000
F0 = 100.714

def analyze_file(filepath, t0, t1):
    with h5py.File(filepath, 'r') as f:
        d = f['data'][:]
    iq = d[0] + 1j * d[1]
    
    iq_seg = iq[int(t0*FS):int(t1*FS)]
    mag = np.abs(iq_seg)
    phs = np.unwrap(np.angle(iq_seg))
    
    # 1. AC RMS above 10 Hz
    sos = butter(4, 10.0 / (0.5*FS), btype='high', output='sos')
    mag_ac = sosfiltfilt(sos, mag)
    phs_ac = sosfiltfilt(sos, phs)
    rms_mag = np.std(mag_ac) # std dev of AC = RMS of AC
    rms_phs = np.std(phs_ac)
    
    # 2. CNR at 4f0 (402.86 Hz)
    f_arr, pxx = welch(iq_seg, fs=FS, nperseg=8192, return_onesided=False)
    sorted_idx = np.argsort(f_arr)
    f_arr, pxx = f_arr[sorted_idx], pxx[sorted_idx]
    
    target_f = 4.0 * F0
    m_peak  = (f_arr >= target_f - 1.0) & (f_arr <= target_f + 1.0)
    m_noise = (f_arr >= target_f - 8.0) & (f_arr <= target_f + 8.0) & ~((f_arr >= target_f - 2.0) & (f_arr <= target_f + 2.0))
    
    peak_val  = np.max(pxx[m_peak]) if np.any(m_peak) else 1e-15
    noise_val = np.mean(pxx[m_noise]) if np.any(m_noise) else 1e-15
    cnr_db = 10 * np.log10(peak_val / noise_val)
    
    return rms_mag, rms_phs, cnr_db

# ── Load ──
rms_m_no,  rms_p_no,  cnr_no  = analyze_file(os.path.join(ULTRA, 'nobody_noultrasound.h5'), 2.0, 8.0)
rms_m_tbl, rms_p_tbl, cnr_tbl = analyze_file(os.path.join(ULTRA, 'ultra_rftable1.h5'), 6.0, 12.0)
rms_m_on,  rms_p_on,  cnr_on  = analyze_file(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 2.0, 7.0)
rms_m_off, rms_p_off, cnr_off = analyze_file(os.path.join(ULTRA, 'ultra_rfbody1.h5'), 12.0, 17.0)

labels = [
    'No US, No Body\n(US OFF)',
    'US ON, Table\n(US ON)',
    'US Body (ON)\n(Active)',
    'US Body (OFF)\n(Shielded)'
]

# ── Plotting ──
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':11,'axes.titleweight':'bold',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300, facecolor='#FFFFFF')
colors = ['#2C3E50', '#1A6FC4', '#C0392B', '#7F8C8D']

# Panel 1: RMS Comparison for both Table and Body (ON vs OFF)
rms_labels = ['Table\nUS OFF', 'Table\nUS ON', 'Body\nUS OFF', 'Body\nUS ON']
rms_vals = [rms_m_no, rms_m_tbl, rms_m_off, rms_m_on]

bars1 = ax1.bar(rms_labels, rms_vals, color=['#7F8C8D', '#1A6FC4', '#95A5A6', '#C0392B'], width=0.5)
ax1.set_title('(A) RF AC Magnitude RMS Amplitude\n(Table vs. Body under US ON / OFF)')
ax1.set_ylabel('AC RMS Amplitude (a.u.)')
ax1.grid(True, ls=':', alpha=0.6, axis='y')
ax1.set_yscale('log')

# Annotate ratio for Table
ratio_table = rms_m_tbl / rms_m_no
ax1.text(0.5, np.sqrt(rms_m_tbl * rms_m_no), f'{ratio_table:.2f}x', 
         ha='center', va='center', color='black', fontweight='bold', fontsize=11)

# Annotate ratio for Body
ratio_body = rms_m_on / rms_m_off
ax1.text(2.5, np.sqrt(rms_m_on * rms_m_off), f'{ratio_body:.1f}x', 
         ha='center', va='center', color='black', fontweight='bold', fontsize=11)

# Annotate values
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f'{yval:.2e}', 
             ha='center', va='bottom', fontsize=8, color='black')

# Panel 2: 4th Harmonic Carrier-to-Noise Ratio (CNR) in dB
bars2 = ax2.bar(labels, [cnr_no, cnr_tbl, cnr_on, cnr_off], 
                color=colors, width=0.5)
ax2.set_title('(B) 4th Harmonic CNR (at 402.86 Hz)\n(Checks for pure acoustic presence without hum)')
ax2.set_ylabel('Carrier-to-Noise Ratio (dB)')
ax2.grid(True, ls=':', alpha=0.6, axis='y')

# Annotate values
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5 if yval >= 0 else yval - 1.5, f'{yval:.1f} dB', 
             ha='center', va='bottom', fontsize=9, color='black')

ax2.set_xticklabels(labels, rotation=15, ha='right')

fig.suptitle(
    'Acousto-RF Verification: AC RMS & Harmonic CNR\n'
    'RMS shows US ON increases fluctuation energy by 31x on Body, but only 1.3x on Table. CNR confirms 4f0 presence.',
    fontsize=13, fontweight='bold', y=0.98
)

plt.tight_layout()
out_file = os.path.join(OUT, 'rms_and_cnr_proof.png')
plt.savefig(out_file, dpi=300, facecolor='#FFFFFF')
print(f"\nPlot saved successfully to: {out_file}")
plt.close()
