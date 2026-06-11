import time
import os
import h5py
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import sys

from bioview.app import Viewer
from bioview.types import UsrpConfiguration, ExperimentConfiguration

def test_recording():
    app = QApplication(sys.argv)
    
    # Configuration from launch_bioview.py
    exp_config = ExperimentConfiguration(
        save_dir = r'D:\Bioview\data',
        file_name = 'test_chest_1x1', 
        save_ds = 100,    
        disp_ds = 10, 
        disp_filter_spec = {
            'bounds': 10,
            'btype': 'low',
            'ftype': 'butter' 
        },
        disp_channels = ['Tx1Rx1'],
    )
    
    usrp = UsrpConfiguration(
        device_name = '', 
        if_freq = [100000],
        if_bandwidth = 1e6, 
        rx_gain = [30],
        tx_gain = [30],
        samp_rate = 1000000,
        carrier_freq = 0.9e9,
        tx_channels = [0],
        rx_channels = [1]
    )

    print("Initializing Viewer...")
    window = Viewer(exp_config=exp_config, usrp_config=[usrp], bio_config=None)
    
    print("Starting Initialization...")
    
    # Capture all logs coming through the panel
    original_log = window.log_display_panel.log_message
    def intercept_log(level, msg):
        print(f"[LOG: {level}] {msg}")
        original_log(level, msg)
    window.log_display_panel.log_message = intercept_log
    
    window.start_initialization()
    
    # We need to wait for init to finish properly and events to be processed
    def on_init_check():
        if window.connection_status.name == 'CONNECTED':
            print("Initialization Successful! Starting Recording...")
            window.update_save_state(True)
            window.start_recording()
            
            # Stop after 5 seconds
            QTimer.singleShot(5000, stop_and_verify)
        else:
            print(f"Initialization Failed. Status: {window.connection_status.name}")
            app.quit()
            
    def stop_and_verify():
        print("Stopping Recording...")
        window.stop_recording()
        
        # Give threads time to close and save to flush
        QTimer.singleShot(1000, verify_file)
        
    def verify_file():
        file_path = exp_config.get_save_path()
        print(f"Checking file: {file_path}")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File exists! Size: {file_size} bytes")
            
            # Open HDF5 and read shape
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' in f:
                        ds = f['data']
                        print(f"Dataset 'data' shape: {ds.shape}")
                        if ds.shape[1] > 0:
                            print("RECORDING TEST PASSED: Data was successfully written.")
                        else:
                            print("RECORDING TEST FAILED: Dataset is empty.")
                    else:
                        print("RECORDING TEST FAILED: 'data' dataset not found in H5.")
            except Exception as e:
                print(f"Failed to read H5 file: {e}")
        else:
            print("RECORDING TEST FAILED: File was not created.")
            
        app.quit()
        
    QTimer.singleShot(1000, on_init_check)
    app.exec()

if __name__ == '__main__':
    test_recording()
