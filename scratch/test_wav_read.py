import scipy.io.wavfile as wav
import os

WAV_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\sthethoscope_rec02.wav'
fs, audio = wav.read(WAV_PATH)
print(f"File: {os.path.basename(WAV_PATH)}")
print(f"Sampling rate: {fs} Hz")
print(f"Data shape: {audio.shape}")
print(f"Duration: {len(audio)/fs:.2f} s")
