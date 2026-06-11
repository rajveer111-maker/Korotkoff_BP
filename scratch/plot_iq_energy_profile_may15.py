import h5py, numpy as np, os
from scipy import signal
from scipy.signal import butter, sosfiltfilt

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_may15.h5'
FS_RF   = 10_000

print("Loading RF...")
with h5py.File(RF_PATH, 'r') as f:
    data = f['data'][:]

# Raw I and Q
I = data[0, :]
Q = data[1, :]
time = np.arange(len(I)) / FS_RF

# Filter 10-50 Hz (Korotkoff band)
sos_koro = butter(4, [10, 49], btype='band', fs=FS_RF, output='sos')
I_hf = sosfiltfilt(sos_koro, I)
Q_hf = sosfiltfilt(sos_koro, Q)

energy = I_hf**2 + Q_hf**2

# Smooth with a 1-second moving average (10,000 samples)
win = 10_000
smooth_energy = np.convolve(energy, np.ones(win)/win, mode='same')

# Print the smooth energy every 2 seconds
print("\nSmooth Energy Profile (10-49 Hz) for rec_koro_may15.h5 at 2-second intervals:")
for t in range(0, int(time[-1]), 2):
    idx = int(t * FS_RF)
    if idx < len(smooth_energy):
        print(f"  Time: {t:2d}s | Energy: {smooth_energy[idx]:.10f}")

# Find peak energy region
peak_idx = np.argmax(smooth_energy)
peak_time = time[peak_idx]
print(f"\nPeak Energy occurs at: {peak_time:.2f}s with energy {smooth_energy[peak_idx]:.10f}")
