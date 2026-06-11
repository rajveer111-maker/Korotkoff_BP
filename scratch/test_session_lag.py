import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from koro_parallel_features import extract_session_features

RF_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\Rec_1.h5'
AUDIO_PATH = r'd:\Bioview\My_RF_work_v1\data_new\data_latest\Sub_1_Prof_kan\sthethoscope_rec01.mp4'

def main():
    if not os.path.exists(RF_PATH):
        print(f"File not found: {RF_PATH}")
        return
    df = extract_session_features(RF_PATH, AUDIO_PATH, 'Sub_1_Prof_kan_Session_1', 'Sub_1_Prof_kan')
    print("Done! Feature dataframe shape:", df.shape if df is not None else "None")

if __name__ == '__main__':
    main()
