import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

FILE_PATH  = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_10s_zoom_analysis.png'
FS         = 10_000
FC_HZ      = 0.9e9
C          = 299792458.0
LAMBDA_MM  = (C / FC_HZ) * 1000
SCALE      = LAMBDA_MM / (4 * np.pi)

def apply_iq(i, q):
    return -i + 1j * q

def iq_condition(iq, keep_dc=True):
    i_mean, q_mean = iq.real.mean(), iq.imag.mean()
    ic, qc = iq.real - i_mean, iq.imag - q_mean
    p1, p2, p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    sp = p3 / np.sqrt(p1*p2+1e-20)
    cp = np.sqrt(max(1-sp**2, 1e-10))
    al = np.sqrt(p2/(p1+1e-20))
    if abs(np.degrees(np.arcsin(np.clip(sp,-1,1)))) < 90:
        qc_corr = (qc - sp*ic) / (al*cp + 1e-15)
    else:
        qc_corr = qc
    if keep_dc:
        return (ic + i_mean) + 1j * (qc_corr + q_mean)
    else:
        return ic + 1j * qc_corr

def robust_phase_unwrap(iq):
    phase_unwrap = np.unwrap(np.angle(iq))
    dphi = np.diff(phase_unwrap)
    carrier_offset = np.median(dphi)
    dphi_clean = dphi - carrier_offset
    dphi_clean = np.clip(dphi_clean, -0.5, 0.5)
    phase_clean = np.insert(np.cumsum(dphi_clean), 0, 0.0)
    return signal.detrend(phase_clean)

print("Loading RF...")
with h5py.File(FILE_PATH, 'r') as f:
    data = f['data'][:]

i_raw, q_raw = data[0,:], data[1,:]
time_full = np.arange(len(i_raw)) / FS

# 1. Condition full signal with Keep-DC
iq_full = iq_condition(apply_iq(i_raw, q_raw), keep_dc=True)

# 2. Zoom exactly on the 10-second active Korotkoff window (t = [24.0, 34.0] seconds)
t_start, t_end = 24.0, 34.0
idx_start, idx_end = int(t_start * FS), int(t_end * FS)

time_koro = time_full[idx_start:idx_end]
iq_koro = iq_full[idx_start:idx_end]

# 3. Calculate Magnitude, Phase (Displacement), and Velocity
magnitude_mm = np.abs(iq_koro) * SCALE
phase_rad = robust_phase_unwrap(iq_koro)
displacement_mm = phase_rad * SCALE

# Velocity calculation
velocity_mms = np.diff(displacement_mm) * FS
velocity_mms = np.append(velocity_mms, velocity_mms[-1])

# Highpass filtered Korotkoff snaps (10-50 Hz)
sos_koro = butter(4, [10, 50], btype='bandpass', fs=FS, output='sos')
koro_snaps = sosfiltfilt(sos_koro, velocity_mms)

print("Plotting Zoomed 10-Second Active Region (Combined Panels)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
plt.subplots_adjust(hspace=0.25)

# === Panel 1: Magnitude (Left Y) vs. Velocity (Right Y) ===
color1 = 'teal'
ax1.plot(time_koro, magnitude_mm - np.mean(magnitude_mm), color=color1, lw=1.5, label='Arterial Magnitude (AC Coupled)')
ax1.set_ylabel('Magnitude Deviation (mm)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)
ax1.set_title('Panel 1: Zoomed Magnitude (Drifting) vs. Physical Velocity (Drift-Free)', fontsize=14, fontweight='bold')

# Create a twin y-axis for velocity on the same panel
ax1_twin = ax1.twinx()
color2 = 'purple'
ax1_twin.plot(time_koro, koro_snaps, color=color2, lw=1.0, alpha=0.8, label='Velocity: Korotkoff Snaps (10-50 Hz)')
ax1_twin.set_ylabel('Velocity (mm/s)', color=color2, fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor=color2)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)


# === Panel 2: Phase Displacement (Left Y) vs. Velocity (Right Y) ===
color3 = 'firebrick'
ax2.plot(time_koro, displacement_mm, color=color3, lw=1.5, label='Arterial Displacement (Phase-based)')
ax2.set_ylabel('Displacement (mm)', color=color3, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.grid(True, alpha=0.3)
ax2.set_title('Panel 2: Zoomed Displacement (Drifting Phase) vs. Physical Velocity (Drift-Free)', fontsize=14, fontweight='bold')

# Create a twin y-axis for velocity on the same panel
ax2_twin = ax2.twinx()
ax2_twin.plot(time_koro, koro_snaps, color=color2, lw=1.0, alpha=0.8, label='Velocity: Korotkoff Snaps (10-50 Hz)')
ax2_twin.set_ylabel('Velocity (mm/s)', color=color2, fontsize=12)
ax2_twin.tick_params(axis='y', labelcolor=color2)
ax2.set_xlabel('Time (s)', fontsize=12)

# Add legends
lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=10)


plt.suptitle('Direct Visual Comparison: Drift Suppression via Phase Differentiation (Velocity)\nDataset: rec_koro_may15.h5 (Active Zoom: 24s to 34s)', fontsize=18, fontweight='bold', y=0.96)
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"Zoomed combined analysis plot saved successfully to: {OUTPUT_IMG}")
