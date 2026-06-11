import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import traceback

print("STARTING TEST KORO", flush=True)

try:
    wav_path = r'C:\Users\rajve\.gemini\antigravity\scratch\temp_audio.wav'
    print(f"Loading {wav_path}", flush=True)
    
    fs_audio, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    audio = data.astype(np.float64) / 32768.0
        
    t = np.arange(len(audio)) / fs_audio
    
    # 10Hz to 200Hz as requested by user!
    sos_k = signal.butter(4, [10, 200], btype='band', fs=fs_audio, output='sos')
    koro_audio = signal.sosfiltfilt(sos_k, audio)
    aud_env = np.abs(signal.hilbert(koro_audio))
    
    # Smooth over 0.25 seconds to capture the 10-200Hz energy better
    win = int(fs_audio * 0.25)
    env_smooth = np.convolve(aud_env, np.ones(win)/win, mode='same')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(t, audio, alpha=0.5, color='gray', label='Raw Audio')
    ax1.plot(t, koro_audio, alpha=0.8, color='red', label='Bandpassed (10-200 Hz)')
    ax1.set_title('New Recording: Stethoscope Audio (10-200 Hz)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, env_smooth, color='black', label='Smoothed Energy')
    
    from scipy.signal import find_peaks
    # Lowered prominence to detect weaker signals
    peaks, _ = find_peaks(env_smooth, prominence=0.005, distance=int(fs_audio*0.5))
    if len(peaks) > 0:
        ax2.plot(t[peaks], env_smooth[peaks], 'rx', markersize=10, label='Detected Peaks')
        ax2.axvspan(t[peaks[0]], t[peaks[-1]], color='yellow', alpha=0.3, label=f'Predicted Koro Region ({t[peaks[-1]]-t[peaks[0]]:.1f}s duration)')
    else:
        ax2.text(0.5, 0.5, 'No Peaks Detected (10-200Hz)', transform=ax2.transAxes, ha='center', va='center', fontsize=12, color='red')
        
    ax2.set_title('Korotkoff Energy Envelope (10-200 Hz)')
    ax2.legend()
    ax2.grid(True)
    
    # Highlight 5-15 sec region as requested
    ax2.axvspan(5, 15, color='blue', alpha=0.1, label='User Expected Region (5-15s)')
    ax2.legend()
    
    plt.tight_layout()
    output_path = r'd:\Bioview\My_RF_work_v1\paper_results\new_rec_actual_results.png'
    plt.savefig(output_path, dpi=300)
    
    output_path2 = r'C:\Users\rajve\.gemini\antigravity\brain\b11c4ec4-c7a3-4eaf-86b7-1efc0188caab\new_rec_actual_results.png'
    plt.savefig(output_path2, dpi=300)
    print(f"Saved plot", flush=True)

except Exception as e:
    print("EXCEPTION:", flush=True)
    traceback.print_exc()
