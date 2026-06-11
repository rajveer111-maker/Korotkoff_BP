import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')

ultra_dir = r"d:\Bioview\My_RF_work_v1\data_new\Ultra"
out_dir = r"d:\Bioview\My_RF_work_v1\scratch"

FS = 10000
FC = -100.714  # Exact carrier frequency (Hz)

def lowpass_filter_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def bandpass_filter_sos(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def extract_demodulated_phase(filepath, t_start, t_end):
    with h5py.File(filepath, 'r') as f:
        # Load the segment
        start_idx = int(t_start * FS)
        end_idx = int(t_end * FS)
        data = f['data'][:, start_idx:end_idx]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS + t_start
    
    # 1. Frequency translation (Digital Downconversion)
    # Multiply by exp(-1j * 2 * pi * FC * t) to shift FC to 0 Hz
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t)
    
    # 2. Lowpass filter to isolate the carrier and its sidebands (e.g. ±10 Hz)
    iq_baseband = lowpass_filter_sos(iq_shifted, 10.0, FS)
    
    # 3. Extract phase
    dp = np.angle(iq_baseband[1:] * np.conj(iq_baseband[:-1]))
    phase = np.insert(np.cumsum(dp), 0, 0.0)
    
    # Convert phase to displacement in micrometers
    # RF wavelength lambda = c / f. Let's assume f = 900 MHz (lambda = 0.33 m)
    # For a reflection, phase sensitivity is 4 * pi * d / lambda.
    # So d = phase * lambda / (4 * pi) = phase * 26510 um.
    disp_um = phase * 26510
    
    # 4. Bandpass filter the phase to isolate physiological band (0.5 to 5 Hz)
    disp_filt = bandpass_filter_sos(disp_um, 0.5, 5.0, FS)
    
    return t, disp_filt, np.abs(iq_baseband)

# Compare Body 2 vs Table 2 during the active phase (2s to 10s)
t_b, disp_b, mag_b = extract_demodulated_phase(os.path.join(ultra_dir, 'ultra_rfbody1.h5'), 2.0, 10.0)
t_t, disp_t, mag_t = extract_demodulated_phase(os.path.join(ultra_dir, 'ultra_rftable2.h5'), 2.0, 10.0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')

# Body 2 Displacement
axes[0, 0].plot(t_b, disp_b, color='teal', linewidth=1.5)
axes[0, 0].set_title("Body 2 - Demodulated Phase Displacement (0.5 - 5 Hz)", fontsize=12, weight='bold')
axes[0, 0].set_ylabel("Displacement (µm)")
axes[0, 0].grid(True, alpha=0.3)

# Table 2 Displacement
axes[0, 1].plot(t_t, disp_t, color='crimson', linewidth=1.5)
axes[0, 1].set_title("Table 2 - Demodulated Phase Displacement (0.5 - 5 Hz)", fontsize=12, weight='bold')
axes[0, 1].set_ylabel("Displacement (µm)")
axes[0, 1].grid(True, alpha=0.3)

# Body 2 Carrier Amplitude
axes[1, 0].plot(t_b, mag_b, color='purple', linewidth=1.5)
axes[1, 0].set_title("Body 2 - Carrier Amplitude (Baseband Magnitude)", fontsize=12, weight='bold')
axes[1, 0].set_ylabel("Amplitude (V)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].grid(True, alpha=0.3)

# Table 2 Carrier Amplitude
axes[1, 1].plot(t_t, mag_t, color='navy', linewidth=1.5)
axes[1, 1].set_title("Table 2 - Carrier Amplitude (Baseband Magnitude)", fontsize=12, weight='bold')
axes[1, 1].set_ylabel("Amplitude (V)")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Demodulating the -100.71 Hz Ultrasound-Modulated RF Carrier", fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_demodulated_100hz.png"), dpi=200)
plt.close()
print("Saved ultra_demodulated_100hz.png")
