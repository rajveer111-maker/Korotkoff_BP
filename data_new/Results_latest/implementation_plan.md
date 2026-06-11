# Improved Korotkoff Duration Detection Analysis

## Background

Your current pipeline in `D:\Bioview\My_RF_work_v1` uses a 6-algorithm consensus model (`koro_window_consensus.py`) and RF-vs-Stethoscope cross-validation (`temp_analysis_2.py`) to detect the Korotkoff sound window during cuff deflation. Results show:

- **Mean IoU 0.78** across 3 sessions (range 0.62–0.95)
- **RF leads acoustic by 1.25–1.58s** (mechanical-acoustic lead time)
- **HR agreement < 1 BPM** between RF and stethoscope
- **RF SNR: +11.75 dB**, Acoustic SNR: +14.20 dB

### Current Limitations Identified

1. **Fixed 60th-percentile threshold** in `rmg_koro_on_off_detection.py` — not adaptive to recording SNR
2. **Static Gaussian duration prior** (μ=10s, σ=3s) in `find_sustained_window()` — penalizes legitimate short/long deflation durations
3. **No deflation slope modeling** — the cuff pressure is declining, but the detector doesn't use this monotonic structure
4. **Per-session variability** — Session C (IoU=0.62, r=0.54) significantly worse than Session B (IoU=0.95)
5. **No per-beat Korotkoff tracking** — only detects the overall window, not individual K-sounds
6. **Cross-correlation lag correction** done post-hoc rather than integrated into detection
7. **No confidence/uncertainty quantification** — binary PASS/FAIL without graded confidence

---

## Proposed Changes

### Component 1: Adaptive Energy-Based Detection (`koro_adaptive_detector.py`)

> [!IMPORTANT]
> This is the core algorithmic improvement — replacing fixed thresholds with data-driven adaptive detection.

#### [NEW] [koro_improved_analysis.py](file:///D:/Bioview/My_RF_work_v1/koro_improved_analysis.py)

**Key improvements:**

1. **Noise-Adaptive Thresholding**: Instead of fixed percentile (60th), estimate the noise floor from the first 8s and last 8s of the recording (known non-Korotkoff regions), then set the detection threshold at `noise_mean + k×noise_std` where `k` is adaptively tuned per-recording based on the noise distribution.

2. **Change-Point Detection (CUSUM)**: Replace the brute-force sliding window with a CUSUM (Cumulative Sum) change-point detector to find the precise onset and offset transitions. This directly identifies where the energy level shifts, rather than scoring every possible window.

```python
# CUSUM change-point for onset/offset
def cusum_detect(energy, threshold_h, drift_v):
    """Detect step changes in energy using Page's CUSUM algorithm"""
    s_pos, s_neg = np.zeros(len(energy)), np.zeros(len(energy))
    for i in range(1, len(energy)):
        s_pos[i] = max(0, s_pos[i-1] + energy[i] - drift_v)
        s_neg[i] = max(0, s_neg[i-1] - energy[i] - drift_v)
    onset = np.where(s_pos > threshold_h)[0][0] if np.any(s_pos > threshold_h) else None
    offset = np.where(s_pos[onset:] < threshold_h * 0.3)[0]  # fade-out
    return onset, offset
```

3. **Deflation-Aware Temporal Model**: The cuff deflation produces a predictable time structure — Korotkoff sounds appear during a specific pressure range (systolic → diastolic). Model this as an increasing-then-decreasing energy envelope (bell-shaped), not a step function.

4. **Multi-Resolution Energy Tracking**: Compute energy at 3 time scales (200ms, 500ms, 1.5s) and require agreement across scales to reject transient spikes and motion artifacts.

---

### Component 2: Per-Beat Korotkoff Sound Detection

#### Part of [koro_improved_analysis.py](file:///D:/Bioview/My_RF_work_v1/koro_improved_analysis.py)

Instead of only detecting the overall window, identify **individual Korotkoff sound events**:

1. **Beat-Synchronous Gating**: Use the detected HR waveform (0.5–3 Hz) to create expected beat windows. Korotkoff sounds are phase-locked to systole.

2. **Per-Beat Energy Score**: For each detected heartbeat, compute the 10–49 Hz energy within a ±50ms window around systolic peak. Classify each beat as "Korotkoff-active" or "silent".

3. **Onset = first consecutive K-active beat**, **Offset = last consecutive K-active beat** (with tolerance for 1–2 missed beats).

```
Beat Timeline:
  HR:   |--♥--|--♥--|--♥--|--♥--|--♥--|--♥--|--♥--|--♥--|--♥--|
  Koro: | ✗  | ✗  | ✓  | ✓  | ✓  | ✓  | ✓  | ✗  | ✗  |
                    ↑onset              ↑offset
  Duration = 5 beats × IBI = 5 × 1.0s = 5.0s
```

4. **Korotkoff Count**: Report the total number of detected K-sounds — this is a clinically meaningful metric (typical: 8–15 sounds per deflation).

---

### Component 3: Enhanced Cross-Validation with Lag Integration

#### Part of [koro_improved_analysis.py](file:///D:/Bioview/My_RF_work_v1/koro_improved_analysis.py)

1. **Pre-Detection Lag Alignment**: Before comparing RF vs stethoscope windows, compute the cross-correlation lag from the full recording energy envelopes, and time-shift the stethoscope signal to align with RF. Then re-detect on the aligned signals.

2. **Lag-Corrected IoU**: Report both raw IoU and lag-corrected IoU. The expected lead time (1.25–1.58s) should be factored out when assessing detection quality.

3. **Graded Confidence Score**: Replace binary PASS/FAIL with a continuous confidence score:
   ```
   confidence = w1×IoU_corrected + w2×(1-onset_diff/3) + w3×HR_agreement + w4×(n_methods/6)
   ```
   where `w1=0.35, w2=0.25, w3=0.2, w4=0.2`

---

### Component 4: Statistical Robustness Improvements

#### Part of [koro_improved_analysis.py](file:///D:/Bioview/My_RF_work_v1/koro_improved_analysis.py)

1. **Bootstrap Confidence Intervals**: For onset/offset times, run 100 bootstrap iterations (resampling energy curve with noise) to estimate ±CI for the detected boundaries.

2. **Remove Hard-Coded Duration Prior**: Replace the Gaussian penalty (μ=10s, σ=3s) with a learned prior from the data. If deflation duration is unknown, use a flat (uninformative) prior.

3. **Robustness to Session C Failure Mode**: Analyze why Session C had IoU=0.62 — likely due to onset mismatch (RF at 12s, steth at 10s). Add a "pre-onset energy ramp" detector that identifies gradual energy buildup before the sharp onset.

---

### Component 5: Improved 16-Panel Visualization Dashboard

#### Part of [koro_improved_analysis.py](file:///D:/Bioview/My_RF_work_v1/koro_improved_analysis.py)

New panels to add:
1. **Per-beat Korotkoff activation timeline** (binary heatmap)
2. **CUSUM change-point curves** showing onset/offset detection traces
3. **Confidence interval bands** on detected windows
4. **Lag-corrected overlay** (stethoscope shifted to align with RF)
5. **Energy ratio curve** (Korotkoff band / noise band) over time
6. **Beat-to-beat Korotkoff energy profile** (reveals crescendo-decrescendo pattern)

---

### Component 6: Batch Multi-Session Processing

#### [NEW] [batch_koro_improved.py](file:///D:/Bioview/My_RF_work_v1/batch_koro_improved.py)

Process all 3 paired sessions automatically and generate:
- Per-session dashboards
- Aggregate statistics table (mean, std, CI for all metrics)
- Bland-Altman plot for onset/offset agreement
- Summary comparison: old algorithm vs improved algorithm

---

## Open Questions

> [!IMPORTANT]
> **Deflation Rate**: Do you know the approximate cuff deflation rate (mmHg/s)? This would allow us to convert the Korotkoff duration (seconds) into a pressure range (systolic – diastolic), which is the clinically meaningful quantity.

> [!IMPORTANT]
> **Additional Data**: Are there more paired RF+stethoscope recordings beyond the 3 sessions (rec_koro_sthe, _1, _2)? More sessions would significantly strengthen the statistical validation.

> [!NOTE]
> **Phase Clip Aggressiveness**: The ultra-tight clip (±0.0002 rad) in `rf_hr_koro_final.py` limits max velocity to 53 mm/s. Should we test with a less aggressive clip to see if it improves Session C detection?

> [!NOTE]
> **Priority**: Should I focus on the core detection algorithm improvements first (Components 1-2) and deliver those before tackling visualization and batch processing? Or do you want everything together?

---

## Verification Plan

### Automated Tests
1. Run improved detector on all 3 sessions and compare IoU, onset/offset error vs current algorithm
2. Verify per-beat detection count is physiologically plausible (8–20 beats per deflation)
3. Verify bootstrap CIs are tighter than ±1s for onset and offset
4. Run batch processing and check aggregate statistics

### Manual Verification
- Visual inspection of new 16-panel dashboards
- Comparison of old vs new detected windows on STFT spectrograms
- Verify lag-corrected IoU improves over raw IoU for all sessions
