import uhd
import numpy as np
import time

def test_hardware():
    print("Finding USRP devices...")
    devs = uhd.find('')
    if not devs:
        print("No USRP found!")
        return
    print(f"Found USRP: {devs[0].to_dict()}")

    print("Initializing MultiUSRP...")
    usrp = uhd.usrp.MultiUSRP("type=b200")
    
    # Configure clock and PPS
    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")
    
    # Configure S21 setup: TX on channel 0, RX on channel 1
    print("Configuring for S21: TX channel 0, RX channel 1...")
    samp_rate = 1e6
    carrier_freq = 0.9e9
    
    # TX Setup
    usrp.set_tx_rate(samp_rate, 0)
    usrp.set_tx_freq(carrier_freq, 0)
    usrp.set_tx_gain(30, 0)
    usrp.set_tx_antenna("TX/RX", 0)
    
    # RX Setup
    usrp.set_rx_rate(samp_rate, 1)
    usrp.set_rx_freq(carrier_freq, 1)
    usrp.set_rx_gain(30, 1)
    usrp.set_rx_antenna("RX2", 1)
    
    print("USRP Configuration complete. Setup is valid!")
    
    # Check if streams can be created
    st_args_tx = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args_tx.channels = [0]
    tx_stream = usrp.get_tx_stream(st_args_tx)
    
    st_args_rx = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args_rx.channels = [1]
    rx_stream = usrp.get_rx_stream(st_args_rx)
    
    print("Hardware streams created successfully!")

if __name__ == "__main__":
    test_hardware()
