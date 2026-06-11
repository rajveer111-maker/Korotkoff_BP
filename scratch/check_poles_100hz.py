import numpy as np
from scipy.signal import butter, sos2zpk

fs = 100
sos = butter(4, [0.4, 3.0], btype='band', fs=fs, output='sos')
print("SOS coefficients for [0.4, 3.0] Hz at 100 Hz:")
print(sos)

z, p, k = sos2zpk(sos)
print("\nPoles:")
for pi in p:
    print(f"  {pi} | Magnitude: {np.abs(pi):.7f} | Distance from unit circle: {1.0 - np.abs(pi):.7f}")
