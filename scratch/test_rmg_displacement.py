import h5py, numpy as np, os
from scipy import signal

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
FS_RF   = 10_000
FC_HZ   = 0.9e9
C       = 299792458.0
LAMBDA_MM = (C / FC_HZ) * 1000
SCALE   = LAMBDA_MM / (4 * np.pi)

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

# Trim like rmg_ultimate_confirmation.py
trim = int(5 * FS_RF)
if data.shape[1] > 2 * trim: data = data[:, trim:-trim]

# Detrend I and Q individually
iq = signal.detrend(data[0, :]) + 1j * signal.detrend(data[1, :])

# Extract phase
phase = np.unwrap(np.angle(iq))

# Detrend phase
phase_clean = signal.detrend(phase)

# Physical conversion
disp_mm = phase_clean * SCALE

print(f"RMG Phase Range: [{np.min(phase_clean):.4f}, {np.max(phase_clean):.4f}] rad")
print(f"RMG Physical Displacement: [{np.min(disp_mm):.4f}, {np.max(disp_mm):.4f}] mm")
