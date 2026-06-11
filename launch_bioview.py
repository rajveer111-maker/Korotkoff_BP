import os 
import time
from bioview import Viewer
from bioview.types import UsrpConfiguration, ExperimentConfiguration
from PyQt6.QtWidgets import QApplication

# Directory setup 
curr_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(curr_dir, 'data_new')
if not os.path.exists(save_dir): os.makedirs(save_dir)

# Experiment variables - Korotkoff Recording
exp_config = ExperimentConfiguration(
    save_dir = save_dir, 
    file_name = f'rec_koro_{time.strftime("%m%d_%H%M%S")}', 

    save_ds = 50, 
    disp_ds = 20, 
    disp_filter_spec = {'bounds': [0.2, 50], 'btype': 'band', 'ftype': 'butter'}, 
    disp_channels = ['Tx1Rx1_I', 'Tx1Rx1_Q'], 
    show_phase = False, 
)

# USRP variables - OPTIMIZED for Korotkoff
usrp = UsrpConfiguration(
    device_name = 'Koro Radar', 
    device_addr = 'serial=356E5B9', 
    if_freq = [100],             # Low IF: minimizes carrier offset (0 crashes TX)
    if_bandwidth = 1e6, 
    rx_gain = [45],             # RX: 45 dB (good sensitivity without saturation)
    tx_gain = [35],             # TX: 35 dB (reduced to avoid saturation)
    samp_rate = 500000,
    carrier_freq = 0.9e9,
    tx_channels = [0],
    rx_channels = [0],
    tx_subdev = 'A:A',
    rx_subdev = 'A:A',
)

bio = None 

if __name__ == '__main__':
    app = QApplication([])
    window = Viewer(exp_config=exp_config,
                    usrp_config=[usrp], 
                    bio_config=bio)
    window.show()
    app.exec()