import h5py, os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = os.path.join(BASE, 'ml_feature_importance_validation.png')

FS_RF = 10_000
DEC   = 10
FS    = FS_RF // DEC
FC    = 0.9e9
SCALE = ((299_792_458.0 / FC) * 1000) / (4.0 * np.pi)

# ── HELPERS ───────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = res
    xc, yc = -a/2, -b/2
    return xc, yc, np.sqrt(xc**2 + yc**2 - c)

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)] + (bins[1]-bins[0])/2
    dphi = dphi - co
    iqr  = np.percentile(dphi, 75) - np.percentile(dphi, 25)
    dphi = np.clip(dphi, -max(3*iqr, 0.01), max(3*iqr, 0.01))
    return signal.detrend(np.insert(np.cumsum(dphi), 0, 0.0), type='linear')

def tkeo(x):
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return y

def extract_features(rf_path, k_on, k_off, sub_name):
    print(f"Extracting features for {sub_name}...")
    with h5py.File(rf_path, 'r') as f:
        rf = f['data'][:]
    i_raw, q_raw = -rf[0, :], rf[1, :]
    
    xc, yc, R = fit_circle(i_raw, q_raw)
    i_c, q_c  = i_raw - xc, q_raw - yc
    phi_raw = robust_phase(i_c, q_c)
    
    sos_lp  = butter(4, 300.0, btype='low', fs=FS_RF, output='sos')
    mag_raw = np.abs(sosfiltfilt(sos_lp, i_c + 1j*q_c))
    
    # 10-200 Hz Korotkoff band
    mag_koro = decimate(bpf(mag_raw, 10, 200, FS_RF), DEC, ftype='fir')
    phi_vel  = decimate(np.append(np.diff(bpf(phi_raw, 10, 200, FS_RF))*FS_RF, 0.0)*SCALE, DEC, ftype='fir')
    
    t = np.arange(len(mag_koro)) / FS
    
    # Epoching parameters (0.5 second windows)
    epoch_len = int(0.5 * FS)
    n_epochs = len(mag_koro) // epoch_len
    
    data = []
    for i in range(n_epochs):
        start = i * epoch_len
        end = start + epoch_len
        t_center = t[start + epoch_len//2]
        
        # Skip cuff inflation transient (ignore first 20 seconds)
        if t_center < 20.0: continue
        
        mk = mag_koro[start:end]
        pv = phi_vel[start:end]
        
        is_koro = 1 if (t_center >= k_on and t_center <= k_off) else 0
        
        # Extract 6 Features per domain (12 total)
        feat = {
            'Subject': sub_name,
            'Time': t_center,
            'Mag_RMS': np.sqrt(np.mean(mk**2)),
            'Ph_RMS': np.sqrt(np.mean(pv**2)),
            'Mag_MAV': np.mean(np.abs(mk)),
            'Ph_MAV': np.mean(np.abs(pv)),
            'Mag_Var': np.var(mk),
            'Ph_Var': np.var(pv),
            'Mag_TKEO': np.mean(tkeo(mk)),
            'Ph_TKEO': np.mean(tkeo(pv)),
            'Mag_Max': np.max(np.abs(mk)),
            'Ph_Max': np.max(np.abs(pv)),
            'Label': is_koro
        }
        data.append(feat)
        
    return pd.DataFrame(data)

# ── BUILD DATASET ─────────────────────────────────────────────────────────
p1 = os.path.join(BASE, 'Sub_1_Prof_kan', 'Rec_6.h5')
df1 = extract_features(p1, 27.7, 43.5, 'Sub_1')

p2 = os.path.join(BASE, 'Sub_2_Rajveer', 'Rec_4.h5')
df2 = extract_features(p2, 27.375, 42.0, 'Sub_2')

df = pd.concat([df1, df2], ignore_index=True)
features = ['Mag_RMS', 'Ph_RMS', 'Mag_MAV', 'Ph_MAV', 'Mag_Var', 'Ph_Var', 'Mag_TKEO', 'Ph_TKEO', 'Mag_Max', 'Ph_Max']

# ── ML MODELING ───────────────────────────────────────────────────────────
print("Training ML Models...")
scaler = StandardScaler()

def train_and_get_importance(data):
    X = data[features]
    y = data['Label']
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    acc_rf = accuracy_score(y, rf.predict(X))
    
    # Gradient Boosting
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X, y)
    acc_gbc = accuracy_score(y, gbc.predict(X))
    
    # Logistic Regression (L1) for sparse feature selection
    lr = LogisticRegression(penalty='elasticnet', l1_ratio=1.0, solver='saga', random_state=42, max_iter=10000)
    lr.fit(X_scaled, y)
    acc_lr = accuracy_score(y, lr.predict(X_scaled))
    
    imp = pd.DataFrame({
        'Feature': features,
        'Random_Forest': rf.feature_importances_,
        'Gradient_Boosting': gbc.feature_importances_,
        'LogReg_Coeffs_Abs': np.abs(lr.coef_[0]) / np.max(np.abs(lr.coef_[0])) # Normalized
    })
    
    imp['Mean_Importance'] = imp[['Random_Forest', 'Gradient_Boosting', 'LogReg_Coeffs_Abs']].mean(axis=1)
    imp = imp.sort_values(by='Mean_Importance', ascending=True)
    return imp, acc_rf, acc_gbc, acc_lr

imp_sub1, r1, g1, l1 = train_and_get_importance(df[df['Subject']=='Sub_1'])
imp_sub2, r2, g2, l2 = train_and_get_importance(df[df['Subject']=='Sub_2'])
imp_all, ra, ga, la = train_and_get_importance(df)

# ── PLOTTING ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, facecolor='#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.labelsize': 11})

def plot_imp(ax, imp_df, title, accs):
    # We will plot the Mean Importance
    y_pos = np.arange(len(imp_df))
    
    # Distinguish Magnitude (blue) vs Phase (red) features
    colors = ['#C0392B' if 'Ph_' in f else '#1A6FC4' for f in imp_df['Feature']]
    
    ax.barh(y_pos, imp_df['Mean_Importance'], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df['Feature'])
    ax.set_xlabel('Mean Normalized Importance (RF, GBC, LR)')
    ax.set_title(title + f'\n(Acc: RF {accs[0]:.2f} | GBC {accs[1]:.2f} | LR {accs[2]:.2f})', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#E8E8E8', linestyle='--')

plot_imp(axes[0], imp_sub1, '(A) Subject 1 Feature Importance', (r1, g1, l1))
plot_imp(axes[1], imp_sub2, '(B) Subject 2 Feature Importance', (r2, g2, l2))
plot_imp(axes[2], imp_all,  '(C) Combined Cohort Importance', (ra, ga, la))

# Custom Legend
blue_patch = mpatches.Patch(color='#1A6FC4', label='Magnitude Features')
red_patch = mpatches.Patch(color='#C0392B', label='Phase Features')
fig.legend(handles=[blue_patch, red_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=12)

fig.suptitle('Machine Learning Feature Importance for Korotkoff Detection (10-200 Hz)', fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print("\nDONE:", OUT)
