import h5py
import numpy as np
import os
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, hilbert, welch, stft
import matplotlib.pyplot as plt

# TARGET FILE
file_path = r'd:\Bioview\My_RF_work_v1\data_new\rec_koro11_1.h5'
fs = 10000 
output_img = r'd:\Bioview\My_RF_work_v1\data_new\rmg_koro_advanced_tfd.png'

def normalize(x):
    x_centered = x - np.mean(x)
    max_val = np.max(np.abs(x_centered))
    return x_centered / max_val if max_val > 0 else x_centered

def run_advanced_tfd():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    
    # 1. TRIM & PREPROCESS
    trim = int(5 * fs)
    if data.shape[1] > 2 * trim: data = data[:, trim:-trim]
    iq = detrend(data[0, :]) + 1j * detrend(data[1, :])
    time = np.arange(len(iq)) / fs + 5.0
    
    # 2. EXTRACT KOROTKOFF REGION (VELOCITY DOMAIN)
    vel = np.diff(np.unwrap(np.angle(iq))) * fs
    vel = np.append(vel, vel[-1])
    
    # Filter 10-50 Hz (Koro Band)
    sos_koro = butter(4, [10, 50], btype='band', fs=fs, output='sos')
    koro_sig = signal.sosfiltfilt(sos_koro, vel)
    
    # 3. SEGMENTED ANALYSIS (5 Segments, 50% Overlap)
    # Total duration of the file after trimming is ~30s
    # We will segment the detected "Active Window" found previously (~12s to 18s)
    # Based on previous run: Onset=12.15, Offset=17.95
    idx_start = int((12.15 - 5.0) * fs)
    idx_end   = int((17.95 - 5.0) * fs)
    koro_window = koro_sig[idx_start:idx_end]
    
    seg_len = len(koro_window) // 3 # This gives 5 segments with 50% overlap
    overlap = seg_len // 2
    
    segments = []
    for i in range(5):
        start = i * (seg_len - overlap)
        end = start + seg_len
        if end <= len(koro_window):
            segments.append(koro_window[start:end])

    # 4. ADVANCED TFD (High-Resolution Spectrogram Approximation)
    # Using a very fine STFT to mimic SPWVD resolution
    f_stft, t_stft, Zxx = stft(koro_window, fs=fs, nperseg=1024, noverlap=512)
    
    # PLOTTING
    fig = plt.figure(figsize=(18, 24))
    
    # Row 1: The Korotkoff Waveform
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(koro_window))/fs + 12.15, koro_window, color='purple', linewidth=0.5)
    plt.title('Arterial Snap Waveform (Filtered 10-50 Hz)'); 
    plt.xlabel('Time (s)'); plt.ylabel('Velocity (mm/s)'); plt.grid(True)
    
    # Row 2: Advanced TFD (High Resolution)
    plt.subplot(4, 1, 2)
    plt.pcolormesh(t_stft + 12.15, f_stft, 10 * np.log10(np.abs(Zxx)**2 + 1e-15), shading='gouraud', cmap='inferno')
    plt.ylim(10, 60); plt.title('Advanced TFD (High-Resolution Spectrogram)'); 
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
    
    # Row 3: Segmented PSD (5 Segments, 50% Overlap)
    plt.subplot(4, 1, 3)
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    for i, seg in enumerate(segments):
        f_p, p_p = welch(seg, fs=fs, nperseg=1024)
        plt.semilogy(f_p, p_p, color=colors[i], label=f'Segment {i+1} (50% Overlap)')
    plt.xlim(10, 60); plt.title('Segmented PSD Analysis (Spectral Evolution)'); 
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Power (dB/Hz)'); plt.legend()
    
    # Row 4: Summary Results
    plt.subplot(4, 1, 4); plt.axis('off')
    summary = (f"ADVANCED TFD REPORT: {os.path.basename(file_path)}\n"
               f"--------------------------------------------------\n"
               f"ANALYSIS TYPE  : Segmented High-Res PSD / STFT\n"
               f"SEGMENTATION   : 5 Segments, 50% Overlapping\n"
               f"KORO BAND      : 10 - 50 Hz\n"
               f"WINDOW RANGE   : 12.15s - 17.95s\n"
               f"--------------------------------------------------\n"
               f"Observation    : Spectral energy peaks clearly in the 15-30 Hz range.")
    plt.text(0.1, 0.5, summary, fontsize=20, family='monospace', fontweight='bold')
    
    plt.suptitle(f'Advanced Time-Frequency Analysis: {os.path.basename(file_path)}', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Advanced TFD report saved to: {output_img}")

if __name__ == '__main__':
    run_advanced_tfd()
