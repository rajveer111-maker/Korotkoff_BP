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
FC = -100.714  # Carrier frequency

def lowpass_filter_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def get_carrier_envelope(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
    
    iq_raw = data[0, :] + 1j * data[1, :]
    t = np.arange(len(iq_raw)) / FS
    
    # 1. Digital Downconversion
    iq_shifted = iq_raw * np.exp(-1j * 2 * np.pi * FC * t)
    
    # 2. Lowpass filter (cutoff 5 Hz) to isolate the carrier amplitude
    iq_baseband = lowpass_filter_sos(iq_shifted, 5.0, FS)
    
    # 3. Carrier amplitude in mV
    carrier_amp_mv = np.abs(iq_baseband) * 1000  # Convert to mV
    
    return t, carrier_amp_mv

# Get envelope for Table 2 and Body 2
t_t2, amp_t2 = get_carrier_envelope(os.path.join(ultra_dir, 'ultra_rftable2.h5'))
t_b2, amp_b2 = get_carrier_envelope(os.path.join(ultra_dir, 'ultra_rfbody1.h5'))

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')

# Table 2
axes[0].plot(t_t2, amp_t2, color='crimson', linewidth=1.5, label='Carrier Amplitude')
axes[0].axvspan(0, 10, color='gray', alpha=0.15, label='Active Window (0-10s)')
axes[0].set_title("Table 2 - Carrier Amplitude Temporal Profile", fontsize=12, weight='bold')
axes[0].set_ylabel("Amplitude (mV)")
axes[0].set_xlim([0, 20])
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Body 2
axes[1].plot(t_b2, amp_b2, color='teal', linewidth=1.5, label='Carrier Amplitude')
axes[1].axvspan(0, 10, color='gray', alpha=0.15, label='Active Window (0-10s)')
axes[1].set_title("Body 2 - Carrier Amplitude Temporal Profile", fontsize=12, weight='bold')
axes[1].set_ylabel("Amplitude (mV)")
axes[1].set_xlabel("Time (s)")
axes[1].set_xlim([0, 20])
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle("Ultrasound-RF Coupling: Temporal Envelope of the 100.71 Hz Carrier", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ultra_carrier_envelope.png"), dpi=200)
plt.close()
print("Saved ultra_carrier_envelope.png")
