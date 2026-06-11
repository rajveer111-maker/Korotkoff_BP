"""
Batch Multi-Session Korotkoff Analysis v3.0
============================================
Processes all paired RF + Stethoscope sessions and generates:
  1. Per-session improved dashboards
  2. Aggregate statistics table
  3. Bland-Altman plot for onset/offset agreement
  4. Old vs New algorithm comparison
  5. Summary report

Usage:
  python batch_koro_improved.py
"""
import numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import the improved analysis module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import koro_improved_analysis as kia

DATA_DIR = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUTPUT_DIR = r'd:\Bioview\My_RF_work_v1\data_new\Results_latest'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SESSIONS = []
subjects = ['Sub_1_Prof_kan', 'Sub_2_Rajveer']

for sub in subjects:
    sub_dir = os.path.join(DATA_DIR, sub)
    if os.path.exists(sub_dir):
        for i in range(1, 11):
            rf_file = os.path.join(sub_dir, f'Rec_{i}.h5')
            audio_file = os.path.join(sub_dir, f'sthethoscope_rec{i:02d}.mp4')
            if not os.path.exists(audio_file) and i == 9 and sub == 'Sub_1_Prof_kan':
                audio_file = os.path.join(sub_dir, f'sthethoscope_rec9.mp4') # Fix typo in filename

            if os.path.exists(rf_file):
                SESSIONS.append({
                    'name': f'{sub}_Session_{i}',
                    'rf': rf_file,
                    'audio': audio_file,
                })

SUMMARY_IMG  = os.path.join(OUTPUT_DIR, 'koro_batch_summary_v3.png')
REPORT_FILE  = os.path.join(OUTPUT_DIR, 'koro_batch_report_v3.txt')


def bland_altman(m1, m2, ax, title='Bland-Altman', units='s'):
    """Generate Bland-Altman plot on given axes."""
    m1 = np.array(m1, dtype=float)
    m2 = np.array(m2, dtype=float)
    mean = (m1 + m2) / 2
    diff = m1 - m2
    md   = np.mean(diff)
    sd   = np.std(diff, ddof=1) if len(diff) > 1 else 0

    ax.scatter(mean, diff, s=120, c='steelblue', edgecolors='black',
               zorder=5, linewidth=1.5)
    ax.axhline(md, color='red', ls='-', lw=2,
               label=f'Mean diff = {md:.2f} {units}')
    ax.axhline(md + 1.96 * sd, color='gray', ls='--', lw=1.5,
               label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax.axhline(md - 1.96 * sd, color='gray', ls='--', lw=1.5,
               label=f'–1.96 SD = {md - 1.96*sd:.2f}')
    ax.fill_between(ax.get_xlim(), md - 1.96 * sd, md + 1.96 * sd,
                    alpha=0.08, color='gray')
    ax.set_xlabel(f'Mean of RF & Steth ({units})', fontsize=11)
    ax.set_ylabel(f'Difference RF – Steth ({units})', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate each point with session name
    for i, (x, y) in enumerate(zip(mean, diff)):
        ax.annotate(f'S{i+1}', (x, y), textcoords='offset points',
                    xytext=(8, 5), fontsize=9, fontweight='bold')


def run_batch():
    print("=" * 70)
    print("  BATCH KOROTKOFF ANALYSIS v3.0")
    print("=" * 70)

    all_results = []

    for idx, session in enumerate(SESSIONS):
        print(f"\n{'-' * 70}")
        print(f"  Processing {session['name']} ({idx + 1}/{len(SESSIONS)})")
        print(f"{'-' * 70}")

        # Check if files exist
        if not os.path.exists(session['rf']):
            print(f"  [SKIP] RF file not found: {session['rf']}")
            continue

        # Set paths in the module
        kia.RF_PATH    = session['rf']
        kia.AUDIO_PATH = session['audio']
        base = os.path.basename(session['rf']).replace('.h5', '')
        kia.OUTPUT_IMG = os.path.join(OUTPUT_DIR,
                                       f'koro_improved_v3_{base}.png')

        try:
            results = kia.run()
            results['session_name'] = session['name']
            all_results.append(results)
        except Exception as e:
            print(f"  [ERROR] {session['name']} failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n[ERROR] No sessions processed successfully!")
        return

    # ══════════════════════════════════════════════════════════════
    # SUMMARY DASHBOARD (6 panels)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  Generating batch summary dashboard...")
    print(f"{'=' * 70}")

    fig = plt.figure(figsize=(24, 30))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30)

    # ── Panel 1: Window Comparison (Horizontal bars) ────────────
    ax1 = fig.add_subplot(gs[0, 0])
    session_names = [r['session_name'] for r in all_results]
    y_pos = np.arange(len(session_names))
    bar_height = 0.3

    for i, r in enumerate(all_results):
        # RF bar
        ax1.barh(i + bar_height/2, r['rf_duration'], left=r['rf_onset'],
                 height=bar_height, color='gold', edgecolor='black',
                 alpha=0.8, label='RF' if i == 0 else '')
        ax1.text(r['rf_onset'] + r['rf_duration']/2, i + bar_height/2,
                 f"{r['rf_onset']:.1f}–{r['rf_offset']:.1f}s",
                 ha='center', va='center', fontsize=8, fontweight='bold')

        # Stethoscope bar (if available)
        if 'steth_onset' in r:
            ax1.barh(i - bar_height/2, r['steth_duration'],
                     left=r['steth_onset'], height=bar_height,
                     color='steelblue', edgecolor='black', alpha=0.8,
                     label='Steth' if i == 0 else '')
            ax1.text(r['steth_onset'] + r['steth_duration']/2,
                     i - bar_height/2,
                     f"{r['steth_onset']:.1f}–{r['steth_offset']:.1f}s",
                     ha='center', va='center', fontsize=8, fontweight='bold',
                     color='white')

        # Per-beat bar (if available)
        if r.get('per_beat_onset') is not None:
            pbd = r['per_beat_offset'] - r['per_beat_onset']
            ax1.barh(i, pbd, left=r['per_beat_onset'], height=bar_height * 0.4,
                     color='limegreen', edgecolor='darkgreen', alpha=0.6,
                     label='Per-Beat' if i == 0 else '')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(session_names, fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('1. Detected Windows Across Sessions', fontweight='bold',
                  fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, axis='x', alpha=0.3)

    # ── Panel 2: Metrics Comparison Table ───────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Build table data
    col_labels = ['Metric']
    for r in all_results:
        col_labels.append(r['session_name'])
    col_labels.append('Mean ± SD')

    rows = []
    # RF Window
    rf_durs = [r['rf_duration'] for r in all_results]
    rows.append(['RF Duration (s)'] +
                [f"{r['rf_duration']:.1f}" for r in all_results] +
                [f"{np.mean(rf_durs):.1f} ± {np.std(rf_durs):.1f}"])

    # K-count
    kcounts = [r['k_count'] for r in all_results]
    rows.append(['K-Sound Count'] +
                [f"{r['k_count']}" for r in all_results] +
                [f"{np.mean(kcounts):.1f} ± {np.std(kcounts):.1f}"])

    # SNR
    snrs = [r['snr_db'] for r in all_results]
    rows.append(['RF SNR (dB)'] +
                [f"{r['snr_db']:.1f}" for r in all_results] +
                [f"{np.mean(snrs):.1f} ± {np.std(snrs):.1f}"])

    # HR
    hrs = [r['hr_rf_bpm'] for r in all_results]
    rows.append(['RF HR (BPM)'] +
                [f"{r['hr_rf_bpm']:.0f}" for r in all_results] +
                [f"{np.mean(hrs):.0f} ± {np.std(hrs):.1f}"])

    # Methods agreeing
    meths = [r['n_methods_agree'] for r in all_results]
    rows.append(['Methods Agree'] +
                [f"{r['n_methods_agree']}/6" for r in all_results] +
                [f"{np.mean(meths):.1f}/6"])

    # Cross-validation metrics (if available)
    has_steth = all([('steth_onset' in r) for r in all_results])
    if has_steth:
        ious = [r.get('raw_iou', 0) for r in all_results]
        rows.append(['Raw IoU'] +
                    [f"{r.get('raw_iou', 0):.3f}" for r in all_results] +
                    [f"{np.mean(ious):.3f} ± {np.std(ious):.3f}"])

        iou_corrs = [r.get('lag_corrected_iou', 0) for r in all_results]
        rows.append(['Lag-Corr IoU'] +
                    [f"{r.get('lag_corrected_iou', 0):.3f}" for r in all_results] +
                    [f"{np.mean(iou_corrs):.3f} ± {np.std(iou_corrs):.3f}"])

        confs = [r.get('confidence', 0) for r in all_results]
        rows.append(['Confidence'] +
                    [f"{r.get('confidence', 0):.3f}" for r in all_results] +
                    [f"{np.mean(confs):.3f} ± {np.std(confs):.3f}"])

        hr_diffs = [abs(r['hr_rf_bpm'] - r.get('hr_aud_bpm', r['hr_rf_bpm']))
                    for r in all_results]
        rows.append(['HR Diff (BPM)'] +
                    [f"{d:.1f}" for d in hr_diffs] +
                    [f"{np.mean(hr_diffs):.1f} ± {np.std(hr_diffs):.1f}"])

    table = ax2.table(cellText=rows, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Color alternating rows
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i + 1, j].set_facecolor('#ecf0f1')
            else:
                table[i + 1, j].set_facecolor('white')

    ax2.set_title('2. Aggregate Statistics', fontweight='bold', fontsize=13,
                  pad=20)

    # ── Panel 3: Bland-Altman Onset ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if has_steth:
        rf_onsets   = [r['rf_onset'] for r in all_results]
        steth_onsets = [r['steth_onset'] for r in all_results]
        bland_altman(rf_onsets, steth_onsets, ax3,
                     title='3. Bland-Altman: Onset Times', units='s')
    else:
        ax3.text(0.5, 0.5, 'No stethoscope data', ha='center', va='center',
                 fontsize=14, transform=ax3.transAxes)
        ax3.set_title('3. Bland-Altman: Onset (N/A)', fontweight='bold')

    # ── Panel 4: Bland-Altman Offset ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if has_steth:
        rf_offsets   = [r['rf_offset'] for r in all_results]
        steth_offsets = [r['steth_offset'] for r in all_results]
        bland_altman(rf_offsets, steth_offsets, ax4,
                     title='4. Bland-Altman: Offset Times', units='s')
    else:
        ax4.text(0.5, 0.5, 'No stethoscope data', ha='center', va='center',
                 fontsize=14, transform=ax4.transAxes)
        ax4.set_title('4. Bland-Altman: Offset (N/A)', fontweight='bold')

    # ── Panel 5: IoU Improvement (Old vs New) ───────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if has_steth:
        x = np.arange(len(session_names))
        w = 0.3
        raw_ious  = [r.get('raw_iou', 0) for r in all_results]
        corr_ious = [r.get('lag_corrected_iou', 0) for r in all_results]

        bars1 = ax5.bar(x - w/2, raw_ious, w, color='steelblue',
                        edgecolor='black', label='Raw IoU')
        bars2 = ax5.bar(x + w/2, corr_ious, w, color='limegreen',
                        edgecolor='black', label='Lag-Corrected IoU')

        for bar, val in zip(bars1, raw_ious):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, corr_ious):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax5.axhline(0.5, color='red', ls='--', lw=1.5, label='Min threshold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(session_names, fontsize=11)
        ax5.set_ylabel('IoU', fontsize=11)
        ax5.set_ylim(0, 1.1)
        ax5.set_title('5. IoU: Raw vs Lag-Corrected', fontweight='bold',
                      fontsize=13)
        ax5.legend(fontsize=9)
        ax5.grid(True, axis='y', alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No stethoscope data', ha='center', va='center',
                 fontsize=14, transform=ax5.transAxes)
        ax5.set_title('5. IoU Comparison (N/A)', fontweight='bold')

    # ── Panel 6: Confidence & Summary ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    lines = [
        "BATCH ANALYSIS SUMMARY v3.0",
        "=" * 52,
        f"Sessions Processed: {len(all_results)} / {len(SESSIONS)}",
        f"",
    ]

    for r in all_results:
        lines.append(f"-- {r['session_name']} --")
        lines.append(f"  RF Window   : {r['rf_onset']:.2f}s – "
                     f"{r['rf_offset']:.2f}s ({r['rf_duration']:.1f}s)")
        if 'steth_onset' in r:
            lines.append(f"  Steth Window: {r['steth_onset']:.2f}s – "
                         f"{r['steth_offset']:.2f}s ({r['steth_duration']:.1f}s)")
            lines.append(f"  IoU (raw/corr): {r.get('raw_iou',0):.3f} / "
                         f"{r.get('lag_corrected_iou',0):.3f}")
            lines.append(f"  Confidence  : {r.get('confidence',0):.3f} "
                         f"[{r.get('conf_label','N/A')}]")
        lines.append(f"  K-Sounds    : {r['k_count']}")
        lines.append(f"  HR (RF)     : {r['hr_rf_bpm']:.0f} BPM")
        lines.append(f"  SNR         : {r['snr_db']:.1f} dB")
        if r.get('onset_ci'):
            lines.append(f"  Onset 95%CI : [{r['onset_ci'][0]:.2f}, "
                         f"{r['onset_ci'][2]:.2f}]s")
        lines.append(f"")

    # Overall statistics
    lines.append("=" * 52)
    lines.append("AGGREGATE STATISTICS:")
    lines.append(f"  Mean RF Duration : {np.mean(rf_durs):.1f} ± "
                 f"{np.std(rf_durs):.1f} s")
    lines.append(f"  Mean K-Sounds    : {np.mean(kcounts):.1f} ± "
                 f"{np.std(kcounts):.1f}")
    lines.append(f"  Mean SNR         : {np.mean(snrs):.1f} ± "
                 f"{np.std(snrs):.1f} dB")
    lines.append(f"  Mean HR          : {np.mean(hrs):.0f} ± "
                 f"{np.std(hrs):.1f} BPM")
    if has_steth:
        lines.append(f"  Mean Raw IoU     : {np.mean(ious):.3f} ± "
                     f"{np.std(ious):.3f}")
        lines.append(f"  Mean Corr IoU    : {np.mean(iou_corrs):.3f} ± "
                     f"{np.std(iou_corrs):.3f}")
        lines.append(f"  Mean Confidence  : {np.mean(confs):.3f} ± "
                     f"{np.std(confs):.3f}")
        improvement = np.mean(iou_corrs) - np.mean(ious)
        lines.append(f"  IoU Improvement  : +{improvement:.3f} "
                     f"(lag correction)")
    lines.append("=" * 52)

    ax6.text(0.03, 0.97, '\n'.join(lines), fontsize=9, family='monospace',
             fontweight='bold', va='top', transform=ax6.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       alpha=0.8, edgecolor='black', linewidth=1.5))

    fig.suptitle('Batch Korotkoff Analysis Summary v3.0',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(SUMMARY_IMG, dpi=150, bbox_inches='tight')
    print(f"\n  Batch summary saved -> {SUMMARY_IMG}")

    # ── Write text report ────────────────────────────────────────
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Text report saved -> {REPORT_FILE}")

    return all_results


if __name__ == '__main__':
    run_batch()
