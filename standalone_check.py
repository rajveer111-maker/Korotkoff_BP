import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

WAV_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\new_recoding.wav'
OUTPUT_PLOT = r'C:\Users\rajve\.gemini\antigravity\brain\b11c4ec4-c7a3-4eaf-86b7-1efc0188caab\new_recoding_check.png'

print("Loading...", flush=True)
fs, data = wavfile.read(WAV_PATH)
if data.ndim > 1:
    data = data.mean(axis=1)

data = data.astype(np.float64) / np.max(np.abs(data))
t = np.arange(len(data)) / fs

# Filter 50-1000Hz
sos = signal.butter(4, [50, 1000], btype='band', fs=fs, output='sos')
data_filt = signal.sosfiltfilt(sos, data)

# Envelope (Energy)
win = int(fs * 0.1)
env = np.convolve(data_filt**2, np.ones(win)/win, mode='same')

# Spectrogram
f, t_spec, Sxx = signal.spectrogram(data, fs=fs, nperseg=int(fs*0.05), noverlap=int(fs*0.04))

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

ax1.plot(t, data, alpha=0.5, color='gray', label='Raw Audio (Far from cuff)')
ax1.plot(t, data_filt, alpha=0.8, color='red', label='Bandpassed (50-1000 Hz)')
ax1.set_title('A. Stethoscope Audio Waveform')
ax1.legend()
ax1.grid(True)

ax2.plot(t, env, color='black')
ax2.set_title('B. Korotkoff Energy Envelope')
ax2.grid(True)

im = ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma', vmin=-100, vmax=-20)
ax3.set_ylim(0, 1000)
ax3.set_title('C. Spectrogram (0-1000 Hz)')
ax3.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=200)
print("Done!", flush=True)
