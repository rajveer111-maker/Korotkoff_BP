import h5py, os, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, iirnotch, filtfilt
from scipy.io import wavfile

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
sub_dir = 'Sub_1_Prof_kan'
rec_idx = 1
wav_path = os.path.join(BASE, sub_dir, f'sthethoscope_rec{rec_idx:02d}.wav')

fs_a, audio = wavfile.read(wav_path)
print("Audio shape:", audio.shape, "dtype:", audio.dtype, "fs:", fs_a)
print("Audio max/min:", np.max(audio), np.min(audio))

# Let's check if the audio is mono or stereo
if audio.ndim > 1:
    audio = audio.mean(axis=1)

def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def tkeo(x):
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(e, 0)

def smooth(x, w, fs):
    k = max(1, int(w * fs))
    # We can use a simpler moving average
    return np.convolve(np.maximum(x, 0), np.ones(k)/k, mode='same')

audio_float = audio.astype(np.float64) / 32768.0
st_bp = bpf(audio_float, 30, 1000, fs_a)
st_hilb = np.abs(signal.hilbert(st_bp))
st_koro = bpf(st_hilb, 20, min(200, fs_a/2 - 1), fs_a)

print("st_koro max/min/mean:", np.max(st_koro), np.min(st_koro), np.mean(st_koro))
tk = tkeo(st_koro)
print("tk max/min/mean:", np.max(tk), np.min(tk), np.mean(tk))

# Let's downsample tk to 1000 Hz before smoothing to avoid huge convolution
tk_1k = decimate(tk, int(fs_a / 1000), ftype='fir')
print("tk_1k shape:", tk_1k.shape, "max/min:", np.max(tk_1k), np.min(tk_1k))

st_wide_1k = smooth(tk_1k, 1.5, 1000)
print("st_wide_1k max/min:", np.max(st_wide_1k), np.min(st_wide_1k))
print("First 20 values of st_wide_1k:", st_wide_1k[30000:30020])
