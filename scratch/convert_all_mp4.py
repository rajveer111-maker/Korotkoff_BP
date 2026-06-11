import os
from moviepy import AudioFileClip

DATA_DIR = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
SUBJECTS = ['Sub_1_Prof_kan', 'Sub_2_Rajveer']

def convert_all():
    print("Starting MP4 to WAV conversion...")
    for sub in SUBJECTS:
        sub_dir = os.path.join(DATA_DIR, sub)
        if not os.path.exists(sub_dir):
            print(f"Directory {sub_dir} does not exist. Skipping.")
            continue
            
        print(f"\nProcessing directory: {sub}")
        for file in os.listdir(sub_dir):
            if file.endswith('.mp4'):
                mp4_path = os.path.join(sub_dir, file)
                wav_name = file.replace('.mp4', '.wav')
                wav_path = os.path.join(sub_dir, wav_name)
                
                if os.path.exists(wav_path):
                    print(f"  Already exists: {wav_name}")
                else:
                    print(f"  Converting {file} -> {wav_name} ...")
                    try:
                        clip = AudioFileClip(mp4_path)
                        clip.write_audiofile(wav_path)
                        clip.close()
                        print(f"  Successfully saved: {wav_name}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to convert {file}: {e}")

if __name__ == '__main__':
    convert_all()
