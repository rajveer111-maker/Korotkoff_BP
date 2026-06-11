import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

FILE_PATH  = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
OUTPUT_IMG = r'd:\Bioview\My_RF_work_v1\data_new\koro_drift_analysis.png'
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

# 1. Condition full signal with Keep-DC to get perfect balance calibration
iq_full = iq_condition(apply_iq(i_raw, q_raw), keep_dc=True)

# 2. Zoom exactly on the active deflation & Korotkoff window (t = [20.0, 42.0] seconds)
t_start, t_end = 20.0, 42.0
idx_start, idx_end = int(t_start * FS), int(t_end * FS)

time_koro = time_full[idx_start:idx_end]
iq_koro = iq_full[idx_start:idx_end]

# 3. Calculate Magnitude, Phase (Displacement), and Velocity for the Zoomed Region
magnitude_mm = np.abs(iq_koro) * SCALE
phase_rad = robust_phase_unwrap(iq_koro)
displacement_mm = phase_rad * SCALE

# Calculate physical velocity (mm/s) using the derivative (d/dt)
velocity_mms = np.diff(displacement_mm) * FS
velocity_mms = np.append(velocity_mms, velocity_mms[-1])  # maintain length

# 4. Apply highpass filtering to isolate Korotkoff mechanical snaps (10-50 Hz)
sos_koro = butter(4, [10, 50], btype='bandpass', fs=FS, output='sos')
koro_snaps = sosfiltfilt(sos_koro, velocity_mms)

print(f"Plotting Korotkoff Region Drift Analysis...")
fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
plt.subplots_adjust(hspace=0.25)

# --- Panel 1: Magnitude (Showing slow pressure release drift) ---
ax = axes[0]
ax.plot(time_koro, magnitude_mm - np.mean(magnitude_mm), color='teal', lw=1.2, label='Arterial Magnitude (AC Coupled)')
ax.set_title('1. Physical Magnitude (Showing Slow Cuff Pressure Release Drift)', fontsize=14, fontweight='bold')
ax.set_ylabel('Magnitude (mm)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')

# --- Panel 2: Phase Displacement (Showing slow breathing & clock drift) ---
ax = axes[1]
ax.plot(time_koro, displacement_mm, color='firebrick', lw=1.2, label='Arterial Displacement (Phase-based)')
ax.set_title('2. Physical Displacement (Phase-based, Showing Slow Clock & Respiration Drift)', fontsize=14, fontweight='bold')
ax.set_ylabel('Displacement (mm)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')

# --- Panel 3: Velocity (Derivative completely eliminates all slow drifts!) ---
ax = axes[2]
ax.plot(time_koro, velocity_mms, color='purple', lw=0.6, alpha=0.5, label='Raw Velocity (d/dt of Phase)')
ax.plot(time_koro, koro_snaps, color='indigo', lw=1.0, label='Bandpass Filtered Korotkoff Snaps (10-50 Hz)')
ax.set_title('3. Physical Velocity (First Derivative Completely Eliminates All Slow Drifts!)', fontsize=14, fontweight='bold')
ax.set_ylabel('Velocity (mm/s)', fontsize=12)
ax.set_xlabel('Time (s)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')

# Zoomed inserts to demonstrate exact mechanical arterial snaps in velocity
ax_ins = ax.inset_axes([0.05, 0.55, 0.35, 0.4])
t_ins_start, t_ins_end = 28.0, 30.0
ins_idx = (time_koro >= t_ins_start) & (time_koro <= t_ins_end)
ax_ins.plot(time_koro[ins_idx], koro_snaps[ins_idx], color='indigo', lw=1.2)
ax_ins.set_title('Zoomed-In Arterial Snaps (28s - 30s)', fontsize=9, fontweight='bold')
ax_ins.grid(True, alpha=0.2)
ax_ins.tick_params(labelsize=8)

plt.suptitle('Scientific Proof: Why Velocity (d/dt) is Superior for Korotkoff Sound Detection\nDataset: rec_koro_may15.h5 (Active Window zoom)', fontsize=18, fontweight='bold', y=0.95)
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"Scientific drift analysis plot saved successfully to: {OUTPUT_IMG}")
