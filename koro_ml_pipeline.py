"""
Korotkoff ML Training, Evaluation, and Visualization Pipeline v1.0
=====================================================================
Trains and evaluates two models using Leave-One-Subject-Out (LOSO) CV:
  - Model 1: Random Forest Classifier
  - Model 2: PyTorch 1D CNN-BiLSTM Sequence Model (optimized for CPU)

Post-processes predictions using CUSUM change-point fusion, compares them
against the v3.0 heuristic baseline, and generates premium 300 DPI figures
for publication.

Usage:
  python koro_ml_pipeline.py
"""
import os, sys, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

warnings.filterwarnings('ignore')

# Config
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
DATASET_CSV = os.path.join(OUTPUT_DIR, 'koro_ml_dataset.csv')
REPORT_FILE = os.path.join(OUTPUT_DIR, 'koro_ml_batch_report.txt')
HEURISTIC_REPORT = os.path.join(OUTPUT_DIR, 'koro_batch_report_v3.txt')

# Set plotting defaults for academic papers
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

FEATURES = [
    'feat_vel_rms', 'feat_vel_tkeo', 'feat_vel_kurtosis', 'feat_vel_hilbert',
    'feat_vel_bandpower', 'feat_vel_stft', 'feat_hjorth_activity',
    'feat_hjorth_mobility', 'feat_hjorth_complexity', 'feat_phase_rms',
    'feat_disp_rms', 'feat_spec_entropy', 'feat_spec_centroid', 'feat_zcr'
]

FEATURE_LABELS = [
    'Velocity RMS', 'Velocity TKEO', 'Velocity Kurtosis', 'Velocity Hilbert',
    'Band Power Ratio', 'STFT Sub-band', 'Hjorth Activity',
    'Hjorth Mobility', 'Hjorth Complexity', 'Phase Fluctuation RMS',
    'HR Displacement RMS', 'Spectral Entropy', 'Spectral Centroid', 'Zero Crossing Rate'
]

# ------------------------------------------------------------------
# PARSE HEURISTIC BASELINE RESULTS
# ------------------------------------------------------------------
def parse_heuristic_baseline():
    """Parses koro_batch_report_v3.txt to get heuristic onset/offset and IoUs."""
    baseline = {}
    if not os.path.exists(HEURISTIC_REPORT):
        print(f"[WARN] Heuristic report not found at {HEURISTIC_REPORT}. Using defaults.")
        return baseline
        
    with open(HEURISTIC_REPORT, 'r') as f:
        content = f.read()
        
    # Split by session
    sessions_raw = content.split('-- ')
    for s_chunk in sessions_raw[1:]:
        lines = s_chunk.strip().split('\n')
        name = lines[0].split(' --')[0].strip()
        
        rf_on, rf_off = 15.0, 25.0
        st_on, st_off = 10.0, 20.0
        raw_iou, corr_iou = 0.0, 0.4
        
        for line in lines[1:]:
            if 'RF Window' in line:
                m = re.search(r'RF Window\s*:\s*([\d\.]+)s\s+([\d\.]+)s', line)
                if m:
                    rf_on, rf_off = float(m.group(1)), float(m.group(2))
            elif 'Steth Window' in line:
                m = re.search(r'Steth Window\s*:\s*([\d\.]+)s\s+([\d\.]+)s', line)
                if m:
                    st_on, st_off = float(m.group(1)), float(m.group(2))
            elif 'IoU (raw/corr)' in line:
                m = re.search(r'IoU \(raw/corr\):\s*([\d\.]+)\s*/\s*([\d\.]+)', line)
                if m:
                    raw_iou, corr_iou = float(m.group(1)), float(m.group(2))
                    
        baseline[name] = {
            'rf_onset': rf_on,
            'rf_offset': rf_off,
            'steth_onset': st_on,
            'steth_offset': st_off,
            'raw_iou': raw_iou,
            'corr_iou': corr_iou
        }
    print(f"Parsed {len(baseline)} sessions from heuristic baseline report.")
    return baseline

# ------------------------------------------------------------------
# PYTORCH DATASET AND DEEP LEARNING MODEL DEFINITIONS
# ------------------------------------------------------------------
class KoroSeqDataset(Dataset):
    """Dataset class that yields full chronological sequences for sequence modeling."""
    def __init__(self, sessions_data):
        self.sequences = []
        for name, group in sessions_data.groupby('session_name'):
            x = group[FEATURES].values
            y = group['target'].values.reshape(-1, 1)
            time = group['time'].values
            self.sequences.append({
                'name': name,
                'x': torch.FloatTensor(x),
                'y': torch.FloatTensor(y),
                'time': time
            })
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

class KoroCNNBiLSTM(nn.Module):
    """PyTorch CNN-BiLSTM Temporal Segmentation Network."""
    def __init__(self, input_dim=14):
        super().__init__()
        # Conv1D layers to capture local Morphological details
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Bidirectional LSTM to capture the global temporal cuff-deflation profile
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape: (seq_len, 14) -> Reshape to (1, 14, seq_len)
        x = x.unsqueeze(0).transpose(1, 2)
        x = self.conv(x)
        # Reshape to (1, seq_len, 32)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Final fully connected layers -> Output shape: (seq_len, 1)
        prob = self.fc(out)
        return prob.squeeze(0)

# ------------------------------------------------------------------
# POST-PROCESSING: CUSUM CHANGE-POINT DETECTION
# ------------------------------------------------------------------
def cusum_window_fusion(probs, times, min_dur=3.0, max_dur=25.0):
    """
    Fuses predictions into a continuous window using Page's CUSUM on probabilities.
    Identifies rising change-points (onset) and falling change-points (offset).
    """
    # Smooth probabilities to reduce local jitter
    probs_smoothed = np.convolve(probs, np.ones(5)/5, mode='same')
    
    mu = np.mean(probs_smoothed)
    drift = 0.25 * max(mu, 0.1)
    threshold = 2.0 * drift
    
    N = len(probs_smoothed)
    s_pos = np.zeros(N)
    for i in range(1, N):
        s_pos[i] = max(0, s_pos[i-1] + (probs_smoothed[i] - mu) - drift)
        
    # Onset detection
    onset_idx = None
    onset_candidates = np.where(s_pos > threshold)[0]
    if len(onset_candidates) > 0:
        first_alarm = onset_candidates[0]
        pre_alarm = s_pos[:first_alarm]
        near_zero = np.where(pre_alarm < threshold * 0.1)[0]
        onset_idx = near_zero[-1] if len(near_zero) > 0 else first_alarm
        
    # Offset detection: look after onset for where CUSUM drops
    offset_idx = None
    if onset_idx is not None:
        post_onset = s_pos[onset_idx:]
        peak_idx = np.argmax(post_onset) + onset_idx
        if peak_idx < N - 1:
            post_peak = s_pos[peak_idx:]
            drop_thresh = s_pos[peak_idx] * 0.35
            drop_candidates = np.where(post_peak < drop_thresh)[0]
            if len(drop_candidates) > 0:
                offset_idx = min(drop_candidates[0] + peak_idx, N - 1)
            else:
                cusum_deriv = np.diff(s_pos[peak_idx:])
                offset_idx = min(np.argmin(cusum_deriv) + peak_idx, N - 1) if len(cusum_deriv) > 0 else N - 1
        else:
            offset_idx = N - 1
            
    if onset_idx is not None and offset_idx is not None:
        on_time = times[min(onset_idx, N - 1)]
        off_time = times[min(offset_idx, N - 1)]
        
        # Verify limits
        if off_time - on_time >= min_dur and off_time - on_time <= max_dur:
            return on_time, off_time
            
    # Fallback to thresholding if CUSUM fails or yields non-physiological window
    active = probs_smoothed > 0.4
    if np.any(active):
        diff = np.diff(np.concatenate([[0], active.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        run_lengths = ends - starts
        best_idx = np.argmax(run_lengths)
        return times[starts[best_idx]], times[min(ends[best_idx], N-1)]
        
    return 10.0, 20.0  # Safe default

def calculate_iou(on1, off1, on2, off2):
    overlap_s = max(on1, on2)
    overlap_e = min(off1, off2)
    overlap = max(0, overlap_e - overlap_s)
    union = max(off1, off2) - min(on1, on2)
    return overlap / union if union > 0 else 0.0

# ------------------------------------------------------------------
# CORE LOSO VALIDATION ENGINE
# ------------------------------------------------------------------
def run_loso_pipeline(df, baseline_heuristic):
    subjects = ['Sub_1_Prof_kan', 'Sub_2_Rajveer']
    session_results = []
    rf_feature_importances = []
    
    for fold, test_sub in enumerate(subjects):
        train_sub = subjects[1 - fold]
        print(f"\n{'=' * 75}")
        print(f"  FOLD {fold+1}: Train on {train_sub} | Test on {test_sub}")
        print(f"{'=' * 75}")
        
        # Split data by subject
        train_df = df[df['subject'] == train_sub].copy()
        test_df = df[df['subject'] == test_sub].copy()
        
        # Fit scaler on training set
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_df[FEATURES])
        y_train = train_df['target'].values
        
        X_test_s = scaler.transform(test_df[FEATURES])
        y_test = test_df['target'].values
        
        # Re-insert scaled features into DataFrames for sequence loading
        train_df_s = train_df.copy()
        train_df_s[FEATURES] = X_train_s
        test_df_s = test_df.copy()
        test_df_s[FEATURES] = X_test_s
        
        # ----------------------------------------------------------
        # 1. TRAIN RANDOM FOREST
        # ----------------------------------------------------------
        print("  [1/4] Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
        rf.fit(X_train_s, y_train)
        rf_feature_importances.append(rf.feature_importances_)
        
        # Predict on Test set
        test_df['pred_rf_prob'] = rf.predict_proba(X_test_s)[:, 1]
        
        # ----------------------------------------------------------
        # 2. TRAIN PYTORCH CNN-BiLSTM SEQUENCE MODEL
        # ----------------------------------------------------------
        print("  [2/4] Training PyTorch CNN-BiLSTM Model on CPU...")
        train_seq = KoroSeqDataset(train_df_s)
        test_seq = KoroSeqDataset(test_df_s)
        
        model = KoroCNNBiLSTM(input_dim=14)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 60 epochs (using batch size = 1 since dataset is small & sequence length varies)
        model.train()
        for epoch in range(60):
            epoch_loss = 0
            for seq in train_seq:
                x_tensor = seq['x']
                y_tensor = seq['y']
                
                optimizer.zero_grad()
                pred = model(x_tensor)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        print(f"    PyTorch training complete (Final Epoch Loss: {epoch_loss/len(train_seq):.4f})")
        
        # Run inference on Test set
        model.eval()
        dl_preds = {}
        with torch.no_grad():
            for seq in test_seq:
                x_tensor = seq['x']
                pred = model(x_tensor)
                dl_preds[seq['name']] = pred.numpy().flatten()
                
        # ----------------------------------------------------------
        # 3. POST-PROCESS AND FUSE WINDOWS
        # ----------------------------------------------------------
        print("  [3/4] Fusing and evaluating prediction windows...")
        for name, group in test_df.groupby('session_name'):
            times = group['time'].values
            targets = group['target'].values
            
            rf_probs = group['pred_rf_prob'].values
            dl_probs = dl_preds[name]
            
            # Map test probability back into DataFrame
            test_df.loc[group.index, 'pred_dl_prob'] = dl_probs
            
            # Run CUSUM fusion
            rf_on, rf_off = cusum_window_fusion(rf_probs, times)
            dl_on, dl_off = cusum_window_fusion(dl_probs, times)
            
            # Retrieve baseline heuristic values and stethoscope ground truth
            # Heuristic baseline is pre-lag-corrected in the report summary, so we retrieve its onset/offset
            heur_on, heur_off = 15.0, 25.0
            st_on, st_off = 10.0, 20.0
            h_raw_iou, h_corr_iou = 0.0, 0.4
            
            if name in baseline_heuristic:
                bh = baseline_heuristic[name]
                heur_on, heur_off = bh['rf_onset'], bh['rf_offset']
                st_on, st_off = bh['steth_onset'], bh['steth_offset']
                h_raw_iou, h_corr_iou = bh['raw_iou'], bh['corr_iou']
                
            # Stethoscope ground truth on RF timeline (which we used to define targets = 1)
            # Find the onset/offset of the target = 1 stream
            target_indices = np.where(targets == 1.0)[0]
            if len(target_indices) > 0:
                true_on = times[target_indices[0]]
                true_off = times[target_indices[-1]]
            else:
                true_on, true_off = st_on, st_off  # Fallback
                
            # Calculate lag (which is physical shift between RF and steth).
            # The target timeline is already lag-corrected, so comparing directly to true_on/off gives Lag-Corrected IoU!
            rf_iou_corr = calculate_iou(rf_on, rf_off, true_on, true_off)
            dl_iou_corr = calculate_iou(dl_on, dl_off, true_on, true_off)
            
            # Raw IoU (comparing directly to acoustic stethoscope window without lag correction)
            rf_iou_raw = calculate_iou(rf_on, rf_off, st_on, st_off)
            dl_iou_raw = calculate_iou(dl_on, dl_off, st_on, st_off)
            
            # F1-Scores on point-by-point binary prediction
            rf_f1 = f1_score(targets, (rf_probs > 0.5).astype(int))
            dl_f1 = f1_score(targets, (dl_probs > 0.5).astype(int))
            
            # AUROC
            try:
                rf_auc = roc_auc_score(targets, rf_probs)
                dl_auc = roc_auc_score(targets, dl_probs)
            except Exception:
                rf_auc = dl_auc = 0.5
                
            session_results.append({
                'session_name': name,
                'subject': test_sub,
                'true_onset': true_on,
                'true_offset': true_off,
                'steth_onset': st_on,
                'steth_offset': st_off,
                # Heuristic Baseline
                'heur_onset': heur_on,
                'heur_offset': heur_off,
                'heur_raw_iou': h_raw_iou,
                'heur_corr_iou': h_corr_iou,
                # Random Forest ML
                'rf_onset': rf_on,
                'rf_offset': rf_off,
                'rf_raw_iou': rf_iou_raw,
                'rf_corr_iou': rf_iou_corr,
                'rf_f1': rf_f1,
                'rf_auc': rf_auc,
                # CNN-BiLSTM DL
                'dl_onset': dl_on,
                'dl_offset': dl_off,
                'dl_raw_iou': dl_iou_raw,
                'dl_corr_iou': dl_iou_corr,
                'dl_f1': dl_f1,
                'dl_auc': dl_auc,
            })
            
            print(f"    {name:30s} | Target: {true_on:5.1f}-{true_off:5.1f}s | Heur IoU: {h_corr_iou:.3f} | RF IoU: {rf_iou_corr:.3f} | DL IoU: {dl_iou_corr:.3f}")
            
        # Write predictions back to the global DataFrame to prevent KeyError during plotting
        df.loc[test_df.index, 'pred_rf_prob'] = test_df['pred_rf_prob']
        df.loc[test_df.index, 'pred_dl_prob'] = test_df['pred_dl_prob']
        
        # ----------------------------------------------------------
        # 4. GENERATE 16-PANEL COMPARATIVE DASHBOARDS (SELECTED SESSIONS)
        # ----------------------------------------------------------
        print("  [4/4] Generating detailed visual dashboards...")
        
    return session_results, np.mean(rf_feature_importances, axis=0), rf_feature_importances[1], rf_feature_importances[0], test_df

# ------------------------------------------------------------------
# GENERATE DETAILED SESSION DASHBOARDS
# ------------------------------------------------------------------
def generate_session_dashboard(res, df, feat_names):
    """Generates a high-quality comparative dashboard for a representative session at 300 DPI."""
    session_name = res['session_name']
    session_df = df[df['session_name'] == session_name].copy()
    
    times = session_df['time'].values
    targets = session_df['target'].values
    rf_probs = session_df['pred_rf_prob'].values
    dl_probs = session_df['pred_dl_prob'].values
    
    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25)
    
    # Stylized spans
    span_gt = dict(color='limegreen', alpha=0.15, label=f"True Ground Truth ({res['true_onset']:.1f}-{res['true_offset']:.1f}s)")
    span_heur = dict(color='gold', alpha=0.10, label=f"Heuristic ({res['heur_onset']:.1f}-{res['heur_offset']:.1f}s) IoU={res['heur_corr_iou']:.3f}")
    span_rf = dict(color='blue', alpha=0.08, label=f"RF ML ({res['rf_onset']:.1f}-{res['rf_offset']:.1f}s) IoU={res['rf_corr_iou']:.3f}")
    span_dl = dict(color='firebrick', alpha=0.08, label=f"CNN-BiLSTM ({res['dl_onset']:.1f}-{res['dl_offset']:.1f}s) IoU={res['dl_corr_iou']:.3f}")

    # Panel 1: RMS Velocity & Phase Fluctuations with Detections
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, session_df['feat_vel_rms'], color='darkgrey', lw=1.5, label='RF Velocity RMS')
    ax1.axvspan(res['true_onset'], res['true_offset'], **span_gt)
    ax1.axvspan(res['heur_onset'], res['heur_offset'], **span_heur)
    ax1.axvspan(res['dl_onset'], res['dl_offset'], **span_dl)
    ax1.set_ylabel('Velocity RMS (mm/s)', fontweight='bold')
    ax1.set_title(f"A. RF Signal Dynamics & Detection Boundary Comparison: {session_name}", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: Sliding Feature Matrix Heatmap (Normalized for visualization)
    ax2 = fig.add_subplot(gs[1, :])
    scaled_feats = StandardScaler().fit_transform(session_df[FEATURES]).T
    im = ax2.imshow(scaled_feats, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2,
                    extent=[times[0], times[-1], -0.5, len(FEATURES)-0.5])
    ax2.set_yticks(np.arange(len(FEATURES)))
    ax2.set_yticklabels(FEATURE_LABELS, fontsize=8)
    ax2.axvline(res['true_onset'], color='limegreen', ls='--', lw=2.0)
    ax2.axvline(res['true_offset'], color='limegreen', ls='--', lw=2.0)
    ax2.set_title("B. Sliding Multi-domain Feature Heatmap (Vertical bars indicate True Window Boundaries)", fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.7, label='Z-score')
    
    # Panel 3: Probability Curves (RF vs DL)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(times, targets, color='green', lw=2.5, label='Ground Truth Target (Aligned)')
    ax3.plot(times, rf_probs, color='blue', lw=1.8, alpha=0.8, label='Random Forest Prob $P_{RF}(t)$')
    ax3.plot(times, dl_probs, color='firebrick', lw=2.2, alpha=0.9, label='CNN-BiLSTM Prob $P_{DL}(t)$')
    ax3.axvspan(res['true_onset'], res['true_offset'], color='limegreen', alpha=0.08)
    ax3.set_ylabel('Probability', fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("C. ML Predicted Probabilities vs. Aligned Acoustic Targets", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2)
    
    # Panel 4: ROC Curves
    ax4 = fig.add_subplot(gs[3, 0])
    from sklearn.metrics import roc_curve
    fpr_rf, tpr_rf, _ = roc_curve(targets, rf_probs)
    fpr_dl, tpr_dl, _ = roc_curve(targets, dl_probs)
    ax4.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'RF (AUC={res["rf_auc"]:.3f})')
    ax4.plot(fpr_dl, tpr_dl, color='firebrick', lw=2, label=f'CNN-BiLSTM (AUC={res["dl_auc"]:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title("D. Receiver Operating Characteristic (ROC)", fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.2)
    
    # Panel 5: Precision-Recall Curves
    ax5 = fig.add_subplot(gs[3, 1])
    prec_rf, rec_rf, _ = precision_recall_curve(targets, rf_probs)
    prec_dl, rec_dl, _ = precision_recall_curve(targets, dl_probs)
    ax5.plot(rec_rf, prec_rf, color='blue', lw=2, label='Random Forest')
    ax5.plot(rec_dl, prec_dl, color='firebrick', lw=2, label='CNN-BiLSTM')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title("E. Precision-Recall Curves", fontweight='bold')
    ax5.legend(loc='lower left', fontsize=9)
    ax5.grid(True, alpha=0.2)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Time (s)')
        
    fig.suptitle(f"Multi-Model Comparative Segmentation Analysis: {session_name}", fontweight='bold', fontsize=15, y=0.995)
    
    img_path = os.path.join(OUTPUT_DIR, f'koro_ml_dashboard_{session_name}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')  # Explicit 300 DPI for paper
    plt.close()
    print(f"    Saved representative session dashboard to -> {img_path} (300 DPI)")

# ------------------------------------------------------------------
# GENERATE AGGREGATE SUMMARY FIGURES (300 DPI FOR PAPER)
# ------------------------------------------------------------------
def generate_aggregate_paper_figures(results, mean_importances, importances_sub1, importances_sub2):
    """Generates the primary aggregate figures for the publication paper at 300 DPI."""
    print("\n[Evaluation] Generating aggregate figures for the publication paper at 300 DPI...")
    
    # Convert results to DataFrame
    res_df = pd.DataFrame(results)
    
    # Calculate Mean Errors (MAE)
    heur_on_mae = np.mean(np.abs(res_df['heur_onset'] - res_df['true_onset']))
    heur_off_mae = np.mean(np.abs(res_df['heur_offset'] - res_df['true_offset']))
    rf_on_mae = np.mean(np.abs(res_df['rf_onset'] - res_df['true_onset']))
    rf_off_mae = np.mean(np.abs(res_df['rf_offset'] - res_df['true_offset']))
    dl_on_mae = np.mean(np.abs(res_df['dl_onset'] - res_df['true_onset']))
    dl_off_mae = np.mean(np.abs(res_df['dl_offset'] - res_df['true_offset']))
    
    # ==================================================================
    # FIGURE 1: PERFORMANCE BAR CHART (IOU & MAE COMPARISON)
    # ==================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar Chart 1: Intersection-over-Union (IoU) Comparison
    methods = ['Heuristic Baseline\n(v3.0)', 'Classical ML\n(Random Forest)', 'Deep Learning\n(CNN-BiLSTM)']
    raw_ious = [res_df['heur_raw_iou'].mean(), res_df['rf_raw_iou'].mean(), res_df['dl_raw_iou'].mean()]
    corr_ious = [res_df['heur_corr_iou'].mean(), res_df['rf_corr_iou'].mean(), res_df['dl_corr_iou'].mean()]
    
    raw_std = [res_df['heur_raw_iou'].std(), res_df['rf_raw_iou'].std(), res_df['dl_raw_iou'].std()]
    corr_std = [res_df['heur_corr_iou'].std(), res_df['rf_corr_iou'].std(), res_df['dl_corr_iou'].std()]
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, raw_ious, width, yerr=raw_std, capsize=5,
                     color='steelblue', edgecolor='black', alpha=0.85, label='Raw IoU (Unshifted)')
    rects2 = ax1.bar(x + width/2, corr_ious, width, yerr=corr_std, capsize=5,
                     color='limegreen', edgecolor='black', alpha=0.85, label='Lag-Corrected IoU')
                     
    ax1.set_ylabel('Mean Intersection-over-Union (IoU)', fontweight='bold')
    ax1.set_title('A. Window Overlap Performance (IoU)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Label bars with values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # Bar Chart 2: Onset/Offset Mean Absolute Error (MAE) Comparison
    on_maes = [heur_on_mae, rf_on_mae, dl_on_mae]
    off_maes = [heur_off_mae, rf_off_mae, dl_off_mae]
    
    rects3 = ax2.bar(x - width/2, on_maes, width, color='gold', edgecolor='black', alpha=0.85, label='Onset MAE')
    rects4 = ax2.bar(x + width/2, off_maes, width, color='firebrick', edgecolor='black', alpha=0.85, label='Offset MAE')
    
    ax2.set_ylabel('Mean Absolute Error (seconds)', fontweight='bold')
    ax2.set_title('B. Boundary Detection Precision (MAE)', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylim(0, max(max(on_maes), max(off_maes)) * 1.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Label bars
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}s', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}s', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    f1_path = os.path.join(OUTPUT_DIR, 'paper_performance_bar_comparison.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')  # Explicit 300 DPI for paper
    plt.close()
    print(f"  Saved aggregate comparison bar chart to -> {f1_path} (300 DPI)")
    
    # ==================================================================
    # FEATURE IMPORTANCE CHART PLOTTER HELPER
    # ==================================================================
    def save_importance_plot(importances, title, filename, color):
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importances)
        
        ax.barh(range(len(FEATURES)), importances[indices], color=color, edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels([FEATURE_LABELS[i] for i in indices], fontweight='bold')
        ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, axis='x', alpha=0.3)
        
        for i, val in enumerate(importances[indices]):
            ax.text(val + 0.002, i, f'{val*100:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        f_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(f_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved feature importance chart to -> {f_path} (300 DPI)")

    # Save all three feature importance plots at 300 DPI
    save_importance_plot(mean_importances, 'Random Forest Feature Importance Analysis (Average Across Subjects)', 'paper_feature_importance.png', 'plum')
    save_importance_plot(importances_sub1, 'Random Forest Feature Importance Analysis (Subject: Sub_1_Prof_kan)', 'paper_feature_importance_Sub_1_Prof_kan.png', 'skyblue')
    save_importance_plot(importances_sub2, 'Random Forest Feature Importance Analysis (Subject: Sub_2_Rajveer)', 'paper_feature_importance_Sub_2_Rajveer.png', 'lightgreen')

    # ==================================================================
    # FIGURE 3: BLAND-ALTMAN AGREEMENT PLOT FOR MODEL B (CNN-BiLSTM)
    # ==================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    
    def plot_bland_altman(ax, m1, m2, title):
        mean = (m1 + m2) / 2
        diff = m1 - m2
        md = np.mean(diff)
        sd = np.std(diff, ddof=1)
        
        ax.scatter(mean, diff, s=100, c='firebrick', edgecolors='black', alpha=0.8, zorder=5)
        ax.axhline(md, color='red', ls='-', lw=2, label=f'Mean Bias = {md:.2f}s')
        ax.axhline(md + 1.96 * sd, color='gray', ls='--', lw=1.5, label=f'+1.96 SD = {md + 1.96*sd:.2f}s')
        ax.axhline(md - 1.96 * sd, color='gray', ls='--', lw=1.5, label=f'–1.96 SD = {md - 1.96*sd:.2f}s')
        ax.fill_between(ax.get_xlim(), md - 1.96 * sd, md + 1.96 * sd, alpha=0.06, color='red')
        
        ax.set_xlabel('Mean of Prediction & Ground Truth (seconds)')
        ax.set_ylabel('Difference (Prediction – Ground Truth) (seconds)')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Annotate each point with session index
        for idx, (x_coord, y_coord) in enumerate(zip(mean, diff)):
            ax.annotate(f'S{idx+1}', (x_coord, y_coord), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, fontweight='bold')

    plot_bland_altman(ax3a, res_df['dl_onset'].values, res_df['true_onset'].values, 'A. Bland-Altman Agreement: Korotkoff Onset')
    plot_bland_altman(ax3b, res_df['dl_offset'].values, res_df['true_offset'].values, 'B. Bland-Altman Agreement: Korotkoff Offset')
    
    fig3.suptitle("Bland-Altman Agreement Analysis ( CNN-BiLSTM Seq Model vs. Acoustic Ground Truth )", fontweight='bold', fontsize=15, y=0.995)
    plt.tight_layout()
    f3_path = os.path.join(OUTPUT_DIR, 'paper_bland_altman_agreement.png')
    plt.savefig(f3_path, dpi=300, bbox_inches='tight')  # Explicit 300 DPI
    plt.close()
    print(f"  Saved Bland-Altman agreement plot to -> {f3_path} (300 DPI)")

# ------------------------------------------------------------------
# WRITE DETAILED STATISTICAL TEXT REPORT
# ------------------------------------------------------------------
def write_statistical_report(results):
    res_df = pd.DataFrame(results)
    
    # Calculate durations
    res_df['true_dur'] = res_df['true_offset'] - res_df['true_onset']
    res_df['heur_dur'] = res_df['heur_offset'] - res_df['heur_onset']
    res_df['rf_dur'] = res_df['rf_offset'] - res_df['rf_onset']
    res_df['dl_dur'] = res_df['dl_offset'] - res_df['dl_onset']
    
    lines = [
        "===========================================================",
        "        ADVANCED ML PIPELINE KOROTKOFF REPORT v1.0",
        "===========================================================",
        f"Sessions Processed: {len(res_df)} / 20",
        "",
        "COMPARATIVE PERFORMANCE ANALYSIS (MEAN +/- SD):",
        "-----------------------------------------------------------",
        f"Heuristic Baseline (v3.0):",
        f"  Mean Raw IoU       : {res_df['heur_raw_iou'].mean():.4f} +/- {res_df['heur_raw_iou'].std():.4f}",
        f"  Mean Lag-Corr IoU  : {res_df['heur_corr_iou'].mean():.4f} +/- {res_df['heur_corr_iou'].std():.4f}",
        f"  Mean Onset MAE     : {np.mean(np.abs(res_df['heur_onset'] - res_df['true_onset'])):.3f} seconds",
        f"  Mean Offset MAE    : {np.mean(np.abs(res_df['heur_offset'] - res_df['true_offset'])):.3f} seconds",
        f"  Mean Duration Error: {np.mean(np.abs(res_df['heur_dur'] - res_df['true_dur'])):.3f} seconds",
        "",
        f"Model 1: Random Forest Classifier (Classical ML):",
        f"  Mean Raw IoU       : {res_df['rf_raw_iou'].mean():.4f} +/- {res_df['rf_raw_iou'].std():.4f}",
        f"  Mean Lag-Corr IoU  : {res_df['rf_corr_iou'].mean():.4f} +/- {res_df['rf_corr_iou'].std():.4f}",
        f"  Mean Onset MAE     : {np.mean(np.abs(res_df['rf_onset'] - res_df['true_onset'])):.3f} seconds",
        f"  Mean Offset MAE    : {np.mean(np.abs(res_df['rf_offset'] - res_df['true_offset'])):.3f} seconds",
        f"  Mean Pointwise F1  : {res_df['rf_f1'].mean():.4f} +/- {res_df['rf_f1'].std():.4f}",
        f"  Mean Pointwise AUC : {res_df['rf_auc'].mean():.4f} +/- {res_df['rf_auc'].std():.4f}",
        f"  Mean Duration Error: {np.mean(np.abs(res_df['rf_dur'] - res_df['true_dur'])):.3f} seconds",
        "",
        f"Model 2: PyTorch 1D CNN-BiLSTM Model (Deep Learning):",
        f"  Mean Raw IoU       : {res_df['dl_raw_iou'].mean():.4f} +/- {res_df['dl_raw_iou'].std():.4f}",
        f"  Mean Lag-Corr IoU  : {res_df['dl_corr_iou'].mean():.4f} +/- {res_df['dl_corr_iou'].std():.4f}",
        f"  Mean Onset MAE     : {np.mean(np.abs(res_df['dl_onset'] - res_df['true_onset'])):.3f} seconds",
        f"  Mean Offset MAE    : {np.mean(np.abs(res_df['dl_offset'] - res_df['true_offset'])):.3f} seconds",
        f"  Mean Pointwise F1  : {res_df['dl_f1'].mean():.4f} +/- {res_df['dl_f1'].std():.4f}",
        f"  Mean Pointwise AUC : {res_df['dl_auc'].mean():.4f} +/- {res_df['dl_auc'].std():.4f}",
        f"  Mean Duration Error: {np.mean(np.abs(res_df['dl_dur'] - res_df['true_dur'])):.3f} seconds",
        "",
        "LOSO FOLD EVALUATION STATS & DURATION CONFIRMATION PER SUBJECT:",
        "-----------------------------------------------------------",
    ]
    
    for sub, sub_df in res_df.groupby('subject'):
        lines += [
            f"Subject '{sub}' Summary:",
            f"  Heuristic Lag-Corr IoU  : {sub_df['heur_corr_iou'].mean():.4f} +/- {sub_df['heur_corr_iou'].std():.4f}",
            f"  Random Forest IoU       : {sub_df['rf_corr_iou'].mean():.4f} +/- {sub_df['rf_corr_iou'].std():.4f}",
            f"  CNN-BiLSTM DL IoU       : {sub_df['dl_corr_iou'].mean():.4f} +/- {sub_df['dl_corr_iou'].std():.4f}",
            f"  CNN-BiLSTM Onset MAE    : {np.mean(np.abs(sub_df['dl_onset'] - sub_df['true_onset'])):.3f}s",
            f"  CNN-BiLSTM Offset MAE   : {np.mean(np.abs(sub_df['dl_offset'] - sub_df['true_offset'])):.3f}s",
            f"  Mean True Korotkoff Dur : {sub_df['true_dur'].mean():.2f} +/- {sub_df['true_dur'].std():.2f}s",
            f"  Mean Heuristic Dur      : {sub_df['heur_dur'].mean():.2f} +/- {sub_df['heur_dur'].std():.2f}s",
            f"  Mean Random Forest Dur  : {sub_df['rf_dur'].mean():.2f} +/- {sub_df['rf_dur'].std():.2f}s",
            f"  Mean CNN-BiLSTM Dur     : {sub_df['dl_dur'].mean():.2f} +/- {sub_df['dl_dur'].std():.2f}s",
            "",
            f"  Per-Session Duration Details for '{sub}':",
            f"    {'Session Name':28s} | {'True Dur':8s} | {'Heur Dur':8s} | {'RF Dur':8s} | {'DL Dur':8s}",
            "    " + "-" * 68
        ]
        for _, row in sub_df.iterrows():
            lines.append(f"    {row['session_name']:28s} | {row['true_dur']:8.2f}s | {row['heur_dur']:8.2f}s | {row['rf_dur']:8.2f}s | {row['dl_dur']:8.2f}s")
        lines.append("")
        
    lines += ["==========================================================="]
    
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved detailed text report to -> {REPORT_FILE}")
    print('\n'.join(lines[:35]))

# ------------------------------------------------------------------
# MAIN PIPELINE RUNNER
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  KOROTKOFF ADVANCED ML TRAINING & PIPELINE")
    print("=" * 70)
    
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset CSV file not found: {DATASET_CSV}. Please run koro_ml_features.py first.")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded dataset of shape: {df.shape}")
    print(f"Subjects available: {df['subject'].unique()}")
    
    baseline_heuristic = parse_heuristic_baseline()
    
    # Run the Leave-One-Subject-Out (LOSO) Cross-Validation
    results, mean_importances, importances_sub1, importances_sub2, last_test_df = run_loso_pipeline(df, baseline_heuristic)
    
    # Generate representative session dashboards (one from Sub_1, one from Sub_2)
    res_sub1 = [r for r in results if r['subject'] == 'Sub_1_Prof_kan'][0]
    res_sub2 = [r for r in results if r['subject'] == 'Sub_2_Rajveer'][0]
    
    print("\n[Visualisation] Generating publication-quality dashboards for representative sessions...")
    generate_session_dashboard(res_sub1, df, FEATURES)
    generate_session_dashboard(res_sub2, df, FEATURES)
    
    # Generate aggregate academic paper figures at 300 DPI
    generate_aggregate_paper_figures(results, mean_importances, importances_sub1, importances_sub2)
    
    # Write statistical text report
    write_statistical_report(results)
    
    print("\n" + "=" * 70)
    print("  ADVANCED ML PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
