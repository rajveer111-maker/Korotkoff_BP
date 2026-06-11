"""
Batch RF Korotkoff Analysis — Sub_1_Prof_kan & Sub_2_Rajveer
=============================================================
Processes all 10 H5 recordings per subject.
Outputs:
  1. Sub_1_Prof_kan_analysis.png  — 10-row per-recording strip
  2. Sub_2_Rajveer_analysis.png   — 10-row per-recording strip
  3. cross_subject_comparison.png — statistical comparison
All at 300 DPI.
"""

import h5py, os, glob, numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, welch
from scipy.fft import next_fast_len
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt

# ── CONFIG ─────────────────────────────────────────────────────────
BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'

SUBJECTS = {
    'Sub_1_Prof_kan': os.path.join(BASE, 'Sub_1_Prof_kan'),
    'Sub_2_Rajveer':  os.path.join(BASE, 'Sub_2_Rajveer'),
}

FS      = 10_000
FC      = 0.9e9
LAMBDA  = (299_792_458.0 / FC) * 1000   # mm
SCALE   = LAMBDA / (4 * np.pi)          # mm/rad  ≈ 26.51

KORO_ON_APPROX  = 24.0    # approximate Korotkoff start (adaptive below)
KORO_DUR        = 17.5    # s
DEFL_FALLBACK   = 20.0    # s

# ── HELPERS ────────────────────────────────────────────────────────
def smooth(x, w):
    k = max(1, int(w))
    return np.convolve(x, np.ones(k)/k, mode='same')

def iq_condition(iq):
    ic = iq.real - iq.real.mean()
    qc = iq.imag - iq.imag.mean()
    p1,p2,p3 = np.mean(ic**2), np.mean(qc**2), np.mean(ic*qc)
    s = p3/np.sqrt(p1*p2+1e-20)
    c = np.sqrt(max(1-s**2,1e-10))
    a = np.sqrt(p2/(p1+1e-20))
    qc2 = (qc-s*ic)/(a*c+1e-15) if abs(np.degrees(np.arcsin(np.clip(s,-1,1))))<90 else qc
    return ic+1j*qc2

def robust_phase(iq):
    dp = np.angle(iq[1:]*np.conj(iq[:-1]))
    h,b = np.histogram(dp, bins=512)
    co  = b[np.argmax(h)]+(b[1]-b[0])/2
    dc  = dp-co
    iqr = np.percentile(dc,75)-np.percentile(dc,25)
    dc  = np.clip(dc,-max(3*iqr,0.017),max(3*iqr,0.017))
    ph  = np.insert(np.cumsum(dc),0,0.0)
    return signal.detrend(ph, type='linear')

def detect_defl(mag, t, fs, lo=8, hi=35, fb=20.0):
    sl = int(lo*fs); sh = int(min(hi*fs,len(mag)))
    if sh<=sl+fs: return fb
    tr  = smooth(np.abs(mag), int(fs*2))
    dt  = np.diff(tr[sl:sh])
    dts = smooth(np.abs(dt), max(1,int(fs*0.5)))
    if dts.max()<1e-12: return fb
    td  = t[sl+np.argmax(dts)]
    return float(td) if lo<=td<=hi else fb

def process_rec(h5path):
    with h5py.File(h5path,'r') as f:
        data = f['data'][:]
    i_raw,q_raw = data[0,:],data[1,:]
    N = len(i_raw)
    t = np.arange(N)/FS

    iq   = iq_condition(-i_raw+1j*q_raw)
    ph   = robust_phase(iq)

    # Korotkoff band velocity (10-200 Hz)
    sos_k = butter(4,[10,200],btype='band',fs=FS,output='sos')
    pk    = sosfiltfilt(sos_k, ph)
    vk    = np.append(np.diff(pk)*FS,0)*SCALE   # mm/s

    # Heartbeat displacement (0.4-3 Hz)
    sos_h = butter(4,[0.4,3.0],btype='band',fs=FS,output='sos')
    dh    = sosfiltfilt(sos_h, ph)*SCALE         # mm

    # Adaptive deflation start
    defl  = detect_defl(vk, t, FS)
    # Korotkoff window: defl+3.5s onset, KORO_DUR length
    k_on  = defl + 3.5
    k_off = k_on + KORO_DUR
    k_off = min(k_off, t[-1]-2.0)

    mask_k    = (t>=k_on)  & (t<=k_off)
    mask_base = (t>=5.0)   & (t<=15.0)

    vk_k    = vk[mask_k]
    vk_base = vk[mask_base]

    rms_k    = float(np.sqrt(np.mean(vk_k**2)))    if len(vk_k)>0    else 0.0
    rms_base = float(np.sqrt(np.mean(vk_base**2))) if len(vk_base)>0 else 1e-6
    peak_k   = float(np.max(np.abs(vk_k)))         if len(vk_k)>0    else 0.0
    snr      = rms_k/(rms_base+1e-20)
    dur      = float(k_off-k_on)

    # HR from heartbeat displacement
    pks,_ = signal.find_peaks(-dh, distance=int(FS*0.5),
                               prominence=np.std(dh[mask_k if mask_k.any() else mask_base])*0.5)
    if len(pks)>1:
        iv  = np.diff(t[pks])
        viv = iv[(iv>0.4)&(iv<1.5)]
        hr  = float(60/np.median(viv)) if len(viv)>0 else 0.0
    else:
        hr  = 0.0

    # PSD of Korotkoff velocity in window
    if len(vk_k) > FS:
        f_p,p_p = welch(vk_k, fs=FS, nperseg=min(len(vk_k),int(FS*2)))
    else:
        f_p = np.array([0]); p_p = np.array([0])

    return dict(
        t=t, vk=vk, dh=dh, ph=ph,
        defl=defl, k_on=k_on, k_off=k_off, dur=dur,
        rms_k=rms_k, rms_base=rms_base, peak_k=peak_k, snr=snr, hr=hr,
        f_psd=f_p, p_psd=p_p, rec_dur=t[-1]
    )

# ── PER-SUBJECT STRIP PLOT ─────────────────────────────────────────
BGFIG = '#0d1117'; BGAX = '#161b22'
GOLD='#FFD700'; CYAN='#00FFFF'; LIME='#39FF14'
CORAL='#FF6B6B'; PURP='#BD93F9'; WHT='#F8F8F2'

def sax(ax, title, xlabel='Time (s)', ylabel=''):
    ax.set_facecolor(BGAX)
    ax.set_title(title, color=WHT, fontsize=8.5, fontweight='bold', pad=4)
    ax.set_xlabel(xlabel, color=WHT, fontsize=7.5)
    ax.set_ylabel(ylabel, color=WHT, fontsize=7.5)
    ax.tick_params(colors=WHT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.5, alpha=0.7)

def plot_subject(subj_name, rec_list, results, out_path):
    n = len(rec_list)
    # 4 columns per recording: vk waveform | PSD | HR disp | stats text
    fig, axes = plt.subplots(n, 4, figsize=(26, 3.2*n), dpi=300)
    fig.patch.set_facecolor(BGFIG)
    plt.subplots_adjust(hspace=0.55, wspace=0.30,
                        left=0.05, right=0.97, top=0.96, bottom=0.02)

    for i,(rname,res) in enumerate(zip(rec_list, results)):
        t   = res['t']
        ds  = max(1, len(t)//8000)
        t_p = t[::ds]; vk_p = res['vk'][::ds]; dh_p = res['dh'][::ds]

        # Col 0: Korotkoff velocity waveform
        ax = axes[i,0]; sax(ax, f'{rname}  |  vk(t) 10-200 Hz', ylabel='mm/s')
        ax.plot(t_p, vk_p, color=LIME, lw=0.4, alpha=0.85)
        ax.axvspan(res['k_on'], res['k_off'], color=GOLD, alpha=0.15, label=f"Koro {res['k_on']:.1f}-{res['k_off']:.1f}s")
        ax.axvline(res['defl'], color=CYAN, ls=':', lw=1.2, label=f"Defl {res['defl']:.1f}s")
        ax.legend(fontsize=6, facecolor='#21262d', labelcolor=WHT, loc='upper right')

        # Col 1: PSD
        ax = axes[i,1]; sax(ax, f'{rname}  |  PSD (Koro window)', xlabel='Frequency (Hz)', ylabel='dB/Hz')
        fm = res['f_psd'] <= 250
        if fm.any() and len(res['p_psd'])>1:
            ax.plot(res['f_psd'][fm], 10*np.log10(res['p_psd'][fm]+1e-20), color=LIME, lw=1.0)
            ax.axvspan(10, 200, color=GOLD, alpha=0.08, label='10-200 Hz')
            ax.legend(fontsize=6, facecolor='#21262d', labelcolor=WHT)

        # Col 2: Heartbeat displacement
        ax = axes[i,2]; sax(ax, f'{rname}  |  HR Displacement (0.4-3 Hz)', ylabel='mm')
        ax.plot(t_p, dh_p, color=CYAN, lw=0.5, alpha=0.85)
        ax.axvspan(res['k_on'], res['k_off'], color=GOLD, alpha=0.12)
        ax.fill_between(t_p, dh_p, alpha=0.12, color=CYAN)

        # Col 3: Stats card
        ax = axes[i,3]; ax.set_facecolor(BGAX); ax.axis('off')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        stats = (
            f"Recording : {rname}\n"
            f"Duration  : {res['rec_dur']:.1f} s\n"
            f"Defl onset: {res['defl']:.2f} s\n"
            f"Koro win  : {res['k_on']:.2f}-{res['k_off']:.2f} s\n"
            f"Koro dur  : {res['dur']:.2f} s\n"
            f"─────────────────────\n"
            f"RMS vk (Koro)  : {res['rms_k']:.1f} mm/s\n"
            f"RMS vk (base)  : {res['rms_base']:.1f} mm/s\n"
            f"Peak |vk| (Koro): {res['peak_k']:.1f} mm/s\n"
            f"SNR Koro/Base  : {res['snr']:.2f} x\n"
            f"Heart Rate     : {res['hr']:.1f} BPM"
        )
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=8,
                family='monospace', color=WHT, va='top',
                bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.9, lw=0))

    fig.suptitle(f'RF Korotkoff Analysis — {subj_name}  |  USRP B210 @ 0.9 GHz  |  10 Recordings',
                 color=WHT, fontsize=13, fontweight='bold', y=0.985)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BGFIG)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

# ── CROSS-SUBJECT COMPARISON ───────────────────────────────────────
def plot_comparison(all_res, out_path):
    sub_names = list(all_res.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=300)
    fig.patch.set_facecolor(BGFIG)
    plt.subplots_adjust(hspace=0.45, wspace=0.32,
                        left=0.07, right=0.96, top=0.92, bottom=0.07)

    colors_sub = [LIME, CORAL]
    metrics = {s: {'rms_k':[], 'rms_base':[], 'snr':[], 'hr':[], 'peak_k':[], 'dur':[]}
               for s in sub_names}

    for s, res_list in all_res.items():
        for r in res_list:
            metrics[s]['rms_k'].append(r['rms_k'])
            metrics[s]['rms_base'].append(r['rms_base'])
            metrics[s]['snr'].append(r['snr'])
            metrics[s]['hr'].append(r['hr'])
            metrics[s]['peak_k'].append(r['peak_k'])
            metrics[s]['dur'].append(r['dur'])

    rec_idx = np.arange(1, 11)

    # Panel 1: RMS vk per recording
    ax = axes[0,0]; sax(ax, 'RMS Korotkoff Velocity per Recording', xlabel='Recording #', ylabel='RMS vk (mm/s)')
    for s,c in zip(sub_names, colors_sub):
        ax.plot(rec_idx, metrics[s]['rms_k'], 'o-', color=c, lw=1.8, ms=6, label=s)
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT)

    # Panel 2: SNR per recording
    ax = axes[0,1]; sax(ax, 'Koro/Baseline SNR per Recording', xlabel='Recording #', ylabel='SNR (x)')
    for s,c in zip(sub_names, colors_sub):
        ax.plot(rec_idx, metrics[s]['snr'], 's-', color=c, lw=1.8, ms=6, label=s)
    ax.axhline(1.0, color='gray', ls='--', lw=1); ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT)

    # Panel 3: Heart Rate per recording
    ax = axes[0,2]; sax(ax, 'Estimated Heart Rate per Recording', xlabel='Recording #', ylabel='HR (BPM)')
    for s,c in zip(sub_names, colors_sub):
        hr_valid = [h if h>30 else np.nan for h in metrics[s]['hr']]
        ax.plot(rec_idx, hr_valid, '^-', color=c, lw=1.8, ms=6, label=s)
    ax.set_ylim([40, 110]); ax.legend(fontsize=8, facecolor='#21262d', labelcolor=WHT)

    # Panel 4: Box plot RMS comparison
    ax = axes[1,0]; ax.set_facecolor(BGAX)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.tick_params(colors=WHT, labelsize=8)
    bdata = [metrics[s]['rms_k'] for s in sub_names]
    bp = ax.boxplot(bdata, labels=sub_names, patch_artist=True, widths=0.5)
    for patch,c in zip(bp['boxes'], colors_sub):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for el in bp['medians']: el.set_color(WHT); el.set_linewidth(2)
    ax.set_title('RMS vk Distribution (Koro Window)', color=WHT, fontsize=9, fontweight='bold')
    ax.set_ylabel('RMS vk (mm/s)', color=WHT, fontsize=8)
    ax.grid(True, color='#21262d', lw=0.5, alpha=0.7)

    # Panel 5: Box plot SNR comparison
    ax = axes[1,1]; ax.set_facecolor(BGAX)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.tick_params(colors=WHT, labelsize=8)
    bdata2 = [metrics[s]['snr'] for s in sub_names]
    bp2 = ax.boxplot(bdata2, labels=sub_names, patch_artist=True, widths=0.5)
    for patch,c in zip(bp2['boxes'], colors_sub):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for el in bp2['medians']: el.set_color(WHT); el.set_linewidth(2)
    ax.set_title('SNR Distribution (Koro/Baseline)', color=WHT, fontsize=9, fontweight='bold')
    ax.set_ylabel('SNR (x)', color=WHT, fontsize=8)
    ax.grid(True, color='#21262d', lw=0.5, alpha=0.7)

    # Panel 6: Summary table
    ax = axes[1,2]; ax.set_facecolor(BGAX); ax.axis('off')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    lines = ["CROSS-SUBJECT SUMMARY", "="*36]
    for s in sub_names:
        m = metrics[s]
        lines += [
            f"\n{s}:",
            f"  RMS vk  : {np.mean(m['rms_k']):.1f} +/- {np.std(m['rms_k']):.1f} mm/s",
            f"  SNR     : {np.mean(m['snr']):.2f} +/- {np.std(m['snr']):.2f} x",
            f"  HR      : {np.nanmean([h for h in m['hr'] if h>30]):.1f} BPM (mean)",
            f"  Peak vk : {np.mean(m['peak_k']):.1f} mm/s (mean)",
            f"  Koro dur: {np.mean(m['dur']):.2f} s (mean)",
        ]
    lines.append("\n"+"="*36)
    ax.text(0.04, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8.5, family='monospace', color=WHT, va='top',
            bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.92, lw=0))
    ax.set_title('Statistical Summary', color=WHT, fontsize=9, fontweight='bold', pad=4)

    fig.suptitle('Cross-Subject RF Korotkoff Comparison  |  Sub_1_Prof_kan vs Sub_2_Rajveer',
                 color=WHT, fontsize=13, fontweight='bold', y=0.97)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BGFIG)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

# ── MAIN ───────────────────────────────────────────────────────────
all_results = {}

for subj_name, subj_dir in SUBJECTS.items():
    print(f"\nProcessing {subj_name} ...")
    h5_files = sorted(glob.glob(os.path.join(subj_dir, 'Rec_*.h5')),
                      key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
    results = []
    for h5f in h5_files:
        rname = os.path.splitext(os.path.basename(h5f))[0]
        print(f"  {rname} ...", end='', flush=True)
        res = process_rec(h5f)
        results.append(res)
        print(f"  SNR={res['snr']:.2f}x  RMS={res['rms_k']:.1f}mm/s  HR={res['hr']:.1f}BPM")

    rec_names = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
    out_img = os.path.join(OUT, f'{subj_name}_analysis.png')
    plot_subject(subj_name, rec_names, results, out_img)
    all_results[subj_name] = results

print("\nGenerating cross-subject comparison ...")
comp_img = os.path.join(OUT, 'cross_subject_comparison.png')
plot_comparison(all_results, comp_img)

print("\nAll done.")
