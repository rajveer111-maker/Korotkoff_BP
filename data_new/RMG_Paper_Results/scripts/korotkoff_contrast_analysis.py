"""
Korotkoff Region Contrast Analysis - Best Preprocessing
Shows WHAT CHANGES in Korotkoff vs Normal using:
- Rolling energy (dB) contrast
- CUSUM change-point detection
- TKEO burst energy: quiet vs Korotkoff zoom
- Amplitude distribution (kurtosis)
- Fused RF beat detection vs GT Stethoscope
"""
import h5py, os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'font.weight':'bold',
    'axes.labelsize':12,'axes.labelweight':'bold',
    'axes.titlesize':11.5,'axes.titleweight':'bold',
    'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':9,'lines.linewidth':1.3,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'#E0E0E0','grid.linewidth':0.6,
})

BASE = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'

# ---- helpers -----------------------------------------------------------------
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order,[lo,hi],btype='band',fs=fs,output='sos')
    return sosfiltfilt(sos, x)

def notch_chain(x, freqs, fs, Q=35):
    for f0 in freqs:
        b, a = signal.iirnotch(f0, Q, fs)
        x = signal.filtfilt(b, a, x)
    return x

def robust_phase(i_c, q_c):
    iq   = i_c + 1j*q_c
    dphi = np.angle(iq[1:]*np.conj(iq[:-1]))
    hist, bins = np.histogram(dphi, bins=512)
    co   = bins[np.argmax(hist)]+(bins[1]-bins[0])/2
    dphi -= co
    iqr  = np.percentile(dphi,75)-np.percentile(dphi,25)
    dphi = np.clip(dphi, -max(3*iqr,0.01), max(3*iqr,0.01))
    return signal.detrend(np.insert(np.cumsum(dphi),0,0.0), type='linear')

def fit_circle(x, y):
    A = np.column_stack([x,y,np.ones_like(x)])
    B = -(x**2+y**2)
    res,*_ = np.linalg.lstsq(A,B,rcond=None)
    return -res[0]/2, -res[1]/2

def tkeo_op(x):
    out = (x**2).copy()
    out[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    return np.maximum(out, 0)

def roll_energy_db(x, win_sec, fs):
    k = max(1, int(win_sec*fs))
    e = np.convolve(x**2, np.ones(k)/k, mode='same')
    return 10*np.log10(e + 1e-20)

def cusum(x, k_factor=0.5):
    mu  = np.mean(x[:len(x)//4])
    sig = np.std(x[:len(x)//4]) + 1e-12
    k   = k_factor * sig
    cs  = np.zeros(len(x))
    for i in range(1, len(x)):
        cs[i] = max(0, cs[i-1] + (x[i]-mu) - k)
    return cs / (np.max(cs)+1e-12)

def shade(ax, k_on, k_off, defl, xl=None):
    ax.axvspan(k_on, k_off, color='#FFF8E1', alpha=0.9, zorder=0)
    ax.axvline(k_on,  color='#F39C12', lw=1.5, ls='--', zorder=3)
    ax.axvline(k_off, color='#F39C12', lw=1.5, ls='--', zorder=3)
    ax.axvline(defl,  color='#999',    lw=0.8, ls=':',  zorder=2)
    if xl: ax.set_xlim(xl)

# ---- main --------------------------------------------------------------------
def run(sub_select):
    cfgs = {
        1: dict(name='Subject 1 (Prof. Kan)', rec='Rec 06',
                rf=os.path.join(BASE,'Sub_1_Prof_kan','Rec_6.h5'),
                wav=os.path.join(BASE,'Sub_1_Prof_kan','sthethoscope_rec06.wav'),
                out=r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\figures\supplementary\korotkoff_contrast_Sub1.png',
                k_on=27.75, k_off=43.50, defl=18.3, t_max=52.0, lag=1.7083,
                notches=[100.71,201.43,302.14,402.86]),
        2: dict(name='Subject 2 (Rajveer)', rec='Rec 04',
                rf=os.path.join(BASE,'Sub_2_Rajveer','Rec_4.h5'),
                wav=os.path.join(BASE,'Sub_2_Rajveer','sthethoscope_rec04.wav'),
                out=r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\figures\supplementary\korotkoff_contrast_Sub2.png',
                k_on=27.375, k_off=42.00, defl=18.6, t_max=51.0, lag=2.6042,
                notches=[50.0,64.0,100.6,201.2]),
    }
    c = cfgs[sub_select]
    k_on,k_off,defl,t_max,lag = c['k_on'],c['k_off'],c['defl'],c['t_max'],c['lag']
    FS = 10_000; DEC=10; fs=FS//DEC
    SCALE = (299792458./0.9e9*1000)/(4*np.pi)

    # ---- Load & best-preprocess RF ------------------------------------------
    print(f"\n[{c['name']}] Loading RF...")
    with h5py.File(c['rf'],'r') as f: raw=f['data'][:]
    i_raw, q_raw = -raw[0], raw[1]
    xc, yc = fit_circle(i_raw, q_raw)
    i_c, q_c = i_raw-xc, q_raw-yc

    phi = robust_phase(i_c, q_c)
    sos_lp = butter(4,300.,btype='low',fs=FS,output='sos')
    mag = np.abs(sosfiltfilt(sos_lp, i_c+1j*q_c))

    phi = notch_chain(phi, c['notches'], FS)
    mag = notch_chain(mag, c['notches'], FS)

    # Optimized bands: Phase 30-80 Hz velocity, Magnitude 20-50 Hz velocity
    phi_vk = np.append(np.diff(bpf(phi,30,80,FS))*FS, 0)*SCALE
    mag_vk = np.append(np.diff(bpf(mag,20,50,FS))*FS, 0)
    t_rf   = np.arange(len(phi_vk))/FS

    # Decimate to 1kHz
    phi_ds = decimate(phi_vk, DEC, ftype='fir')
    mag_ds = decimate(mag_vk, DEC, ftype='fir')
    t_ds   = np.arange(len(phi_ds))/fs

    # Normalised TKEO envelopes
    def norm_tkeo(x, t):
        tk = np.convolve(tkeo_op(x), np.ones(int(0.15*fs))/int(0.15*fs), mode='same')
        mk = (t>=k_on)&(t<=k_off)
        mb = (t>=22.0)&(t<=k_on-2.)
        b  = np.percentile(tk[mb],5)
        return np.clip((tk-b)/(np.max(tk[mk])+1e-12),0,None)

    mag_n = norm_tkeo(mag_ds, t_ds)
    phi_n = norm_tkeo(phi_ds, t_ds)
    fused = np.sqrt((mag_n**2 + phi_n**2)/2)

    # Rolling energy dB (0.3 s window - short enough to see beat bursts)
    mag_edb = roll_energy_db(mag_ds, 0.3, fs)
    phi_edb = roll_energy_db(phi_ds, 0.3, fs)

    # Baseline energy level
    mask_b = (t_ds>=defl+3)&(t_ds<=k_on-2.)
    base_m = np.percentile(mag_edb[mask_b], 50)
    base_p = np.percentile(phi_edb[mask_b], 50)
    mag_rise = mag_edb - base_m   # dB above baseline
    phi_rise = phi_edb - base_p

    # CUSUM on rolling energy
    mag_cs = cusum(mag_edb)
    phi_cs = cusum(phi_edb)

    # ---- Load stethoscope GT ------------------------------------------------
    print(f"[{c['name']}] Loading stethoscope...")
    fs_a, aud = wavfile.read(c['wav'])
    aud = aud.astype(np.float64)/32768.
    if aud.ndim > 1: aud = aud.mean(axis=1)
    aud_bp  = bpf(aud, 50., 1000., fs_a)
    aud_env = np.convolve(tkeo_op(aud_bp),
                          np.ones(int(0.15*fs_a))/int(0.15*fs_a), mode='same')
    t_a = np.arange(len(aud_env))/fs_a + lag
    aud_env[(t_a<defl+3)|(t_a>k_off+1.2)] = 0.
    mk_a = (t_a>=k_on)&(t_a<=k_off)
    mb_a = (t_a>=22.)&(t_a<=k_on-2.)
    b_a  = np.percentile(aud_env[mb_a],5)
    steth_n = np.clip((aud_env-b_a)/(np.max(aud_env[mk_a])+1e-12),0,None)

    # ---- Quiet vs Korotkoff windows (TKEO energy) ---------------------------
    # Use post-deflation recovery window (k_off + 2.0 to k_off + 7.0) as quiet baseline
    t_qs, t_qe = k_off+2.0, k_off+7.0
    t_ks, t_ke = k_on+1.0, k_on+6.0
    mq = (t_ds>=t_qs)&(t_ds<=t_qe)
    mk2= (t_ds>=t_ks)&(t_ds<=t_ke)

    # Short TKEO (50ms) for burst capture
    mag_tk = np.convolve(tkeo_op(mag_ds), np.ones(int(0.05*fs))/int(0.05*fs), mode='same')
    phi_tk = np.convolve(tkeo_op(phi_ds), np.ones(int(0.05*fs))/int(0.05*fs), mode='same')

    mq_tk = mag_tk[mq]; mk_tk = mag_tk[mk2]
    pq_tk = phi_tk[mq]; pk_tk = phi_tk[mk2]

    p95_mq = np.percentile(mq_tk,95); p95_mk = np.percentile(mk_tk,95)
    p95_pq = np.percentile(pq_tk,95); p95_pk = np.percentile(pk_tk,95)
    ratio_m = p95_mk/(p95_mq+1e-12)
    ratio_p = p95_pk/(p95_pq+1e-12)

    print(f"  Mag P95-burst: Quiet(Recovery)={p95_mq:.2e}  Koro={p95_mk:.2e}  Ratio={ratio_m:.2f}x")
    print(f"  Phi P95-burst: Quiet(Recovery)={p95_pq:.2f}  Koro={p95_pk:.2f}  Ratio={ratio_p:.2f}x")

    # Kurtosis
    def kurt(x): return float(np.mean(((x-np.mean(x))/(np.std(x)+1e-12))**4))
    km_q = kurt(mq_tk); km_k = kurt(mk_tk)
    kp_q = kurt(pq_tk); kp_k = kurt(pk_tk)

    # ---- FIGURE 6x2 ---------------------------------------------------------
    CM='#1A6FC4'; CP='#C0392B'; CF='#1B7F4E'; CGT='#0A7251'
    xl = (defl-1, t_max)

    fig = plt.figure(figsize=(22,28), dpi=300, facecolor='#FFFFFF')
    gs  = gridspec.GridSpec(6, 2, hspace=0.55, wspace=0.28,
                            left=0.07, right=0.97, top=0.95, bottom=0.03)

    # ROW 0: Raw velocity (full timeline) - shows HOW SIMILAR they look
    ax = fig.add_subplot(gs[0,0])
    shade(ax, k_on, k_off, defl, xl)
    mv = mag_ds/(np.std(mag_ds)+1e-12)
    ax.plot(t_ds, mv, color=CM, lw=0.35, alpha=0.55, rasterized=True)
    ax.set_ylim(-6,6)
    ax.set_title('(A) RF Magnitude Velocity 20-50 Hz  [full recording]\nRaw amplitude LOOKS similar in both regions')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (sigma)')
    ax.text(k_on+0.5, 5.2, 'Korotkoff\nWindow', color='#F39C12', fontsize=9, fontweight='bold', zorder=5)

    ax = fig.add_subplot(gs[0,1])
    shade(ax, k_on, k_off, defl, xl)
    pv = phi_ds/(np.std(phi_ds)+1e-12)
    ax.plot(t_ds, pv, color=CP, lw=0.35, alpha=0.55, rasterized=True)
    ax.set_ylim(-6,6)
    ax.set_title('(B) RF Phase Velocity 30-80 Hz  [full recording]\nRaw amplitude LOOKS similar in both regions')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (sigma)')
    ax.text(k_on+0.5, 5.2, 'Korotkoff\nWindow', color='#F39C12', fontsize=9, fontweight='bold', zorder=5)

    # ROW 1: Rolling energy dB - THE KEY CONTRAST
    ax = fig.add_subplot(gs[1,0])
    shade(ax, k_on, k_off, defl, xl)
    ax.plot(t_ds, mag_edb, color=CM, lw=0.8, alpha=0.85, label='Mag energy (dB)')
    ax.axhline(base_m, color='#777', lw=1.0, ls='--', label=f'Baseline {base_m:.0f} dB')
    ax.fill_between(t_ds, base_m, mag_edb,
                    where=(t_ds>=k_on)&(t_ds<=k_off), color=CM, alpha=0.3, label='Energy above baseline')
    ax.set_title('(C) RF Magnitude  Rolling Energy 0.3s window  [dB]\nEnergy RISES clearly inside Korotkoff window')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Signal Energy (dB)')
    ax.legend(frameon=False, ncol=2)

    ax = fig.add_subplot(gs[1,1])
    shade(ax, k_on, k_off, defl, xl)
    ax.plot(t_ds, phi_edb, color=CP, lw=0.8, alpha=0.85, label='Phase energy (dB)')
    ax.axhline(base_p, color='#777', lw=1.0, ls='--', label=f'Baseline {base_p:.0f} dB')
    ax.fill_between(t_ds, base_p, phi_edb,
                    where=(t_ds>=k_on)&(t_ds<=k_off), color=CP, alpha=0.3, label='Energy above baseline')
    ax.set_title('(D) RF Phase  Rolling Energy 0.3s window  [dB]\nEnergy RISES clearly inside Korotkoff window')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Signal Energy (dB)')
    ax.legend(frameon=False, ncol=2)

    # ROW 2: Energy rise above baseline + CUSUM
    ax = fig.add_subplot(gs[2,0])
    shade(ax, k_on, k_off, defl, xl)
    ax.fill_between(t_ds, 0, mag_rise, where=mag_rise>0, color=CM, alpha=0.4, label='Mag energy rise')
    ax.fill_between(t_ds, 0, phi_rise, where=phi_rise>0, color=CP, alpha=0.35, label='Phase energy rise')
    ax.plot(t_ds, mag_rise, color=CM, lw=0.6)
    ax.plot(t_ds, phi_rise, color=CP, lw=0.6, ls='--')
    ax.axhline(0, color='#444', lw=0.8)
    ax.axhline(6, color='#444', lw=0.6, ls=':', label='+6 dB marker')
    ax.set_title('(E) Energy Rise Above Baseline (dB)\nHighlights Korotkoff window without thresholding')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('dB above baseline')
    ax.legend(frameon=False, ncol=3)

    ax = fig.add_subplot(gs[2,1])
    shade(ax, k_on, k_off, defl, xl)
    ax.plot(t_ds, mag_cs, color=CM, lw=1.0, label='Mag CUSUM')
    ax.plot(t_ds, phi_cs, color=CP, lw=1.0, ls='--', label='Phase CUSUM')
    t_xl_mask = (t_a>=xl[0])&(t_a<=xl[1])
    if np.sum(t_xl_mask) > 10:
        steth_rs = np.interp(t_ds, t_a[t_xl_mask], steth_n[t_xl_mask])
        ax.plot(t_ds, steth_rs, color=CGT, lw=0.8, alpha=0.7, label='GT Steth')
    ax.set_title('(F) CUSUM Change Detection\nCUMSUM rises sharply at Korotkoff onset, flat elsewhere')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('CUSUM (norm.)')
    ax.legend(frameon=False, ncol=3)

    # ROW 3: TKEO burst energy zoom  quiet vs Korotkoff
    t_qw = t_ds[mq] - t_qs
    t_kw = t_ds[mk2]- t_ks
    s_m  = np.percentile(mq_tk,99)+1e-12
    s_p  = np.percentile(pq_tk,99)+1e-12

    ax = fig.add_subplot(gs[3,0])
    ax.fill_between(t_qw, 0, mq_tk/s_m, color='#888', alpha=0.55, label=f'Recovery Baseline  P95={p95_mq:.2e}')
    ax.fill_between(t_kw, 0, mk_tk/s_m, color=CM,    alpha=0.65, label=f'Korotkoff Active  P95={p95_mk:.2e}')
    ax.set_title(f'(G) RF Mag TKEO Burst Energy  5s window comparison\nKorotkoff clicks are {ratio_m:.2f}x higher energy (P95)')
    ax.set_xlabel('Relative Time (s)'); ax.set_ylabel('Norm. TKEO Burst Energy')
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[3,1])
    ax.fill_between(t_qw, 0, pq_tk/s_p, color='#888', alpha=0.55, label=f'Recovery Baseline  P95={p95_pq:.1f}')
    ax.fill_between(t_kw, 0, pk_tk/s_p, color=CP,    alpha=0.65, label=f'Korotkoff Active  P95={p95_pk:.1f}')
    ax.set_title(f'(H) RF Phase TKEO Burst Energy  5s window comparison\nKorotkoff clicks are {ratio_p:.2f}x higher energy (P95)')
    ax.set_xlabel('Relative Time (s)'); ax.set_ylabel('Norm. TKEO Burst Energy')
    ax.legend(frameon=False)

    # ROW 4: Amplitude histogram (TKEO energy) - shows heavy tail in Korotkoff
    bins = np.linspace(0, 1, 80)
    ax = fig.add_subplot(gs[4,0])
    ax.hist(mq_tk/s_m, bins=bins, color='#999', alpha=0.55, density=True, label='Quiet recovery baseline')
    ax.hist(mk_tk/s_m, bins=bins, color=CM,    alpha=0.55, density=True, label='Korotkoff active')
    ax.set_title(f'(I) RF Magnitude TKEO  Energy Distribution\nKurtosis: Quiet={km_q:.0f}  Korotkoff={km_k:.0f}  [higher=more clicks]')
    ax.set_xlabel('Norm. Burst Energy'); ax.set_ylabel('Probability Density')
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[4,1])
    ax.hist(pq_tk/s_p, bins=bins, color='#999', alpha=0.55, density=True, label='Quiet recovery baseline')
    ax.hist(pk_tk/s_p, bins=bins, color=CP,    alpha=0.55, density=True, label='Korotkoff active')
    ax.set_title(f'(J) RF Phase TKEO  Energy Distribution\nKurtosis: Quiet={kp_q:.0f}  Korotkoff={kp_k:.0f}  [higher=more clicks]')
    ax.set_xlabel('Norm. Burst Energy'); ax.set_ylabel('Probability Density')
    ax.legend(frameon=False)

    # ROW 5: Fused RF beat detection vs GT (full width)
    ax = fig.add_subplot(gs[5,:])
    shade(ax, k_on, k_off, defl, xl)
    ax.fill_between(t_ds, 0, fused, alpha=0.3, color=CF)
    ax.plot(t_ds, fused, color=CF, lw=1.0, label='RF Fused TKEO (Mag+Phase)')
    t_xl = (t_a>=xl[0])&(t_a<=xl[1])
    ax.plot(t_a[t_xl], steth_n[t_xl], color=CGT, lw=0.9, ls='--', alpha=0.8, label='GT Stethoscope TKEO')
    mk3 = (t_ds>=k_on)&(t_ds<=k_off)
    fk  = fused.copy(); fk[~mk3] = 0
    thr = np.percentile(fk[mk3], 50)
    beats, _ = find_peaks(fk, height=thr, distance=int(0.5*fs), prominence=thr*0.5)
    for bt in t_ds[beats]:
        ax.axvline(bt, color=CF, lw=1.0, alpha=0.65, ymin=0, ymax=0.88)
    ax.set_title(f'(K) RF Fused vs GT Stethoscope  [{len(beats)} Korotkoff beats detected]\n'
                 'Vertical lines = detected cardiac pulses inside Korotkoff window')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalised Energy')
    ax.set_ylim(-0.05, 1.25)
    ax.legend(frameon=False, ncol=3)

    fig.suptitle(
        f"Korotkoff Region Contrast Analysis  |  {c['name']}  |  {c['rec']}\n"
        f"RF Magnitude & Phase: What CHANGES in Korotkoff vs Normal Region",
        fontsize=14, fontweight='bold', y=0.978
    )
    plt.savefig(c['out'], dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    print(f"  Saved: {c['out']}")
    plt.close()

run(1)
run(2)
print("Done.")
