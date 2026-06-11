import numpy as np
import time
from scipy.signal import hilbert
from scipy.fft import next_fast_len

x = np.random.randn(2243367)

print("Starting standard hilbert...")
t0 = time.time()
try:
    # Standard hilbert on non-power-of-2
    # We will limit wait to 3 seconds
    import signal
    res_std = hilbert(x)
    print(f"Standard hilbert completed in {time.time() - t0:.4f}s")
except Exception as e:
    print(f"Standard failed: {e}")

print("Starting optimized hilbert...")
t0 = time.time()
n_fast = next_fast_len(len(x))
res_opt = hilbert(x, N=n_fast)[:len(x)]
print(f"Optimized hilbert (N={n_fast}) completed in {time.time() - t0:.4f}s")
