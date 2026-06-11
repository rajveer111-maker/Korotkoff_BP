import os
print("Importing moviepy...")
try:
    from moviepy import AudioFileClip
    print("Success importing moviepy!")
    
    path = r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope2.mp4'
    print(f"Loading {path}...")
    clip = AudioFileClip(path)
    print("Clip loaded!")
    audio = clip.to_soundarray()
    print(f"Success! Shape: {audio.shape}, FPS: {clip.fps}")
    clip.close()
except Exception as e:
    print(f"Failed with error: {e}")
