import numpy as np
import scipy.signal as signal
from scipy.signal import welch
import h5py

SUB1_RF = r"d:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_6.h5"

with h5py.File(SUB1_RF, 'r') as f:
    data = f['data'][:]
i_raw, q_raw = -data[0, :], data[1, :]

# Demodulate
A = np.column_stack([i_raw, q_raw, np.ones_like(i_raw)])
B = -(i_raw**2 + q_raw**2)
res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
xc, yc = -res[0]/2, -res[1]/2
i_c = i_raw - xc
q_c = q_raw - yc

iq = i_c + 1j*q_c
phi = np.unwrap(np.angle(iq))
mag = np.abs(iq)

fs = 10000
f_phi, p_phi = welch(phi, fs=fs, nperseg=fs*2)
f_mag, p_mag = welch(mag, fs=fs, nperseg=fs*2)

# Find top peaks in [30, 200] Hz range
mask = (f_phi >= 30) & (f_phi <= 200)
peaks_phi, _ = signal.find_peaks(10*np.log10(p_phi[mask]), prominence=5)
peaks_mag, _ = signal.find_peaks(10*np.log10(p_mag[mask]), prominence=5)

print("Subject 1 Phase noise peaks in [30, 200] Hz:")
for p in peaks_phi:
    print(f"  {f_phi[mask][p]:.2f} Hz -> {10*np.log10(p_phi[mask][p]):.1f} dB")

print("Subject 1 Magnitude noise peaks in [30, 200] Hz:")
for p in peaks_mag:
    print(f"  {f_mag[mask][p]:.2f} Hz -> {10*np.log10(p_mag[mask][p]):.1f} dB")
