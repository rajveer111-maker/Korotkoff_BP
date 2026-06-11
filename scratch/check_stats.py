import numpy as np
phi_hr = np.load('phi_hr.npy') if False else None
# Let's read the statistics directly in python:
import h5py, os
from scipy.signal import butter, sosfiltfilt, decimate, detrend

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
s = dict(sub_dir='Sub_1_Prof_kan', rec=6)
rp = os.path.join(BASE, s['sub_dir'], f"Rec_{s['rec']}.h5")
with h5py.File(rp, 'r') as f: raw = f['data'][:]
ic, qc = -raw[0,:], raw[1,:]
# Let's print raw statistics
print("Raw ic mean:", np.mean(ic), "std:", np.std(ic))
print("Raw qc mean:", np.mean(qc), "std:", np.std(qc))
# Let's see if circle fit center is correct
def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    r, *_ = np.linalg.lstsq(A, B, rcond=None)
    return -r[0]/2, -r[1]/2

xc, yc = fit_circle(ic, qc)
print("fit_circle xc, yc:", xc, yc)
ic -= xc; qc -= yc
phi = np.angle(ic + 1j*qc)
print("phi mean:", np.mean(phi), "std:", np.std(phi), "range:", np.min(phi), "to", np.max(phi))
# let's see why robust_phase_clipping has 0.0002 clipping
dphi = np.angle((ic[1:] + 1j*qc[1:]) * np.conj(ic[:-1] + 1j*qc[:-1]))
print("dphi range:", np.min(dphi), "to", np.max(dphi), "std:", np.std(dphi))
print("percentiles of dphi:", np.percentile(dphi, [1, 5, 25, 50, 75, 95, 99]))
