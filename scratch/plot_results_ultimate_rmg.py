import h5py
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt

# TARGET FILE
file_name = 'rec_koro11_1.h5'
data_dir = r'd:\Bioview\My_RF_work_v1\data_new'
file_path = os.path.join(data_dir, file_name)
fs = 10000 
output_img = os.path.join(data_dir, 'rmg_standard_analysis.png')

# Physical Constants (Verified Units)
carrier_freq = 0.9e9 
c = 299792458
wavelength_mm = (c / carrier_freq) * 1000 # ~333.10 mm

def run_rmg_analysis():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        
    i_raw = data[0,:]
    q_raw = data[1,:]
    time = np.arange(len(i_raw)) / fs
    
    # 1. IQ Demodulation & Centering
    i_c = i_raw - np.mean(i_raw)
    q_c = q_raw - np.mean(q_raw)
    complex_sig = i_c + 1j * q_c
    
    # 2. RMG Signal Extraction (Standard Labels)
    # NCS_am (Amplitude Modulation)
    ncs_am = np.abs(complex_sig)
    ncs_am_detrend = signal.detrend(ncs_am)
    
    # NCS_ph (Phase Modulation -> Displacement)
    ncs_ph_rad = np.unwrap(np.angle(complex_sig))
    ncs_ph_mm = (ncs_ph_rad * wavelength_mm) / (4 * np.pi)
    ncs_ph_detrend = signal.detrend(ncs_ph_mm)
    
    # RMG Velocity (d/dt of Phase)
    rmg_vel = np.diff(ncs_ph_detrend) * fs
    rmg_vel = np.append(rmg_vel, rmg_vel[-1])
    
    # 3. Physiological Filtering
    # HR Pulse (0.7 - 2.5 Hz) from Phase
    sos_hr = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
    hr_rmg = signal.sosfiltfilt(sos_hr, ncs_ph_detrend)
    
    # Korotkoff Snaps (10 - 50 Hz) from Velocity
    sos_koro = signal.butter(4, [10, 50], btype='bandpass', fs=fs, output='sos')
    koro_rmg = signal.sosfiltfilt(sos_koro, rmg_vel)

    # PLOTTING (RMG STYLE)
    fig = plt.figure(figsize=(18, 30))
    
    # Row 1: Raw Quadrature Inputs
    plt.subplot(7, 2, 1); plt.plot(time, i_raw, color='blue', alpha=0.7); plt.title('Raw I Signal'); plt.ylabel('Amp (a.u.)')
    plt.subplot(7, 2, 2); plt.plot(time, q_raw, color='orange', alpha=0.7); plt.title('Raw Q Signal'); plt.ylabel('Amp (a.u.)')
    
    # Row 2: NCS_am and NCS_ph (RMG Standard)
    plt.subplot(7, 2, 3)
    plt.plot(time, ncs_am_detrend, color='green')
    plt.title('NCS_am (Amplitude Modulation - RMG)'); plt.ylabel('ΔAmp (a.u.)')
    
    plt.subplot(7, 2, 4)
    plt.plot(time, ncs_ph_detrend, color='red')
    plt.title('NCS_ph (Phase Displacement - RMG)'); plt.ylabel('Disp (mm)')
    plt.ylim(np.percentile(ncs_ph_detrend, 1), np.percentile(ncs_ph_detrend, 99))
    
    # Row 3: RMG Velocity (Primary Korotkoff Source)
    plt.subplot(7, 1, 3)
    plt.plot(time, rmg_vel, color='purple', linewidth=0.7)
    plt.title('RMG Velocity (d/dt of NCS_ph)'); plt.ylabel('Vel (mm/s)')
    plt.ylim(np.percentile(rmg_vel, 0.5), np.percentile(rmg_vel, 99.5))
    
    # Row 4: Extracted Vitals
    plt.subplot(7, 1, 4)
    plt.plot(time, hr_rmg, color='brown', linewidth=1.5)
    plt.title('Extracted Cardiac Pulse (0.7-2.5 Hz)'); plt.ylabel('Disp (mm)')
    
    # Row 5: Frequency Confirmation
    ax_fft = plt.subplot(7, 1, 5)
    f_psd, p_psd = signal.welch(hr_rmg, fs, nperseg=int(fs*10))
    ax_fft.semilogy(f_psd, p_psd, color='brown')
    ax_fft.set_xlim(0, 5); ax_fft.set_title('Cardiac Power Spectrum (RMG FFT Proof)'); ax_fft.set_xlabel('Freq (Hz)')
    
    # Row 6: TFD Analysis (Korotkoff Bursts)
    plt.subplot(7, 1, 6)
    f_s, t_s, Sxx = signal.spectrogram(koro_rmg, fs, nperseg=int(fs/4))
    plt.pcolormesh(t_s, f_s, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='magma')
    plt.ylim(0, 60); plt.title('Time-Frequency Analysis (Korotkoff TFD)'); plt.ylabel('Freq (Hz)')
    
    # Summary
    plt.subplot(7, 1, 7)
    plt.axis('off')
    hr_bpm = f_psd[np.argmax(p_psd)] * 60
    summary = (f"RADIOMYOGRAPHY (RMG) STANDARD REPORT\n"
               f"--------------------------------------------------\n"
               f"Primary Metric: NCS_ph (Phase Displacement)\n"
               f"Secondary Metric: NCS_am (Amplitude Modulation)\n"
               f"Confirmed Heart Rate: {hr_bpm:.2f} BPM\n"
               f"Detection Method: Phase-Derivative (Velocity) for Korotkoff")
    plt.text(0.05, 0.5, summary, fontsize=15, family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"RMG Standard analysis saved to: {output_img}")

if __name__ == '__main__':
    run_rmg_analysis()
