"""
Ultra Detailed RF Analysis: Magnitude vs Phase
Korotkoff vs Normal Region — Both Subjects
Saves figures + CSV report to RMG_Paper_Results/ultra_analysis/
"""
import h5py, os
import numpy as np
from scipy import signal, stats
from scipy.signal import butter, sosfiltfilt, decimate, welch, hilbert, find_peaks
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

BASE  = r'd:\Bioview\My_RF_work_v1\data_new\data_latest'
OUT   = r'd:\Bioview\My_RF_work_v1\data_new\RMG_Paper_Results\ultra_analysis'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'font.weight':'bold',
    'axes.labelsize':11,'axes.labelweight':'bold',
    'axes.titlesize':10.5,'axes.titleweight':'bold',
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'#E0E0E0','grid.linewidth':0.5,
})

# ── helpers ──────────────────────────────────────────────────────────────────
def bpf(x, lo, hi, fs, order=4):
    sos = butter(order,[lo,hi],btype='band',fs=fs,output='sos')
    return sosfiltfilt(sos,x)

def notch_chain(x, freqs, fs, Q=35):
    for f0 in freqs:
        b,a = signal.iirnotch(f0,Q,fs); x=signal.filtfilt(b,a,x)
    return x

def robust_phase(ic,qc):
    iq=ic+1j*qc; dp=np.angle(iq[1:]*np.conj(iq[:-1]))
    h,b=np.histogram(dp,bins=512); co=b[np.argmax(h)]+(b[1]-b[0])/2
    dp-=co; iqr=np.percentile(dp,75)-np.percentile(dp,25)
    dp=np.clip(dp,-max(3*iqr,0.01),max(3*iqr,0.01))
    return signal.detrend(np.insert(np.cumsum(dp),0,0.0),type='linear')

def fit_circle(x,y):
    A=np.column_stack([x,y,np.ones_like(x)]); B=-(x**2+y**2)
    r,*_=np.linalg.lstsq(A,B,rcond=None); return -r[0]/2,-r[1]/2

def tkeo(x):
    o=(x**2).copy(); o[1:-1]=x[1:-1]**2-x[:-2]*x[2:]; return np.maximum(o,0)

def features(x, fs, label):
    """Compute all statistical + spectral features of a 1-D segment."""
    f = {}
    # Time-domain stats
    f['rms']      = float(np.sqrt(np.mean(x**2)))
    f['mean_abs'] = float(np.mean(np.abs(x)))
    f['std']      = float(np.std(x))
    f['skew']     = float(stats.skew(x))
    f['kurt']     = float(stats.kurtosis(x, fisher=False))  # excess kurtosis
    f['p5']       = float(np.percentile(x,5))
    f['p95']      = float(np.percentile(x,95))
    f['p95_p5']   = f['p95']-f['p5']   # dynamic range
    # TKEO
    tk = tkeo(x)
    f['tkeo_mean']= float(np.mean(tk))
    f['tkeo_p95'] = float(np.percentile(tk,95))
    f['tkeo_max'] = float(np.max(tk))
    # Spectral
    nperseg = min(len(x), int(fs*1.0))
    freq,pxx = welch(x,fs=fs,nperseg=nperseg)
    # Band powers (normalised to total)
    def bpow(lo,hi):
        m=(freq>=lo)&(freq<=hi); return float(np.trapz(pxx[m],freq[m]))
    total = bpow(1,500) + 1e-20
    f['psd_1_5hz']   = bpow(1,5)/total
    f['psd_5_30hz']  = bpow(5,30)/total
    f['psd_30_60hz'] = bpow(30,60)/total
    f['psd_60_120hz']= bpow(60,120)/total
    f['psd_120_180hz']= bpow(120,180)/total
    f['spectral_entropy'] = float(-np.sum((pxx/(np.sum(pxx)+1e-20))*
                                          np.log(pxx/(np.sum(pxx)+1e-20)+1e-12)))
    # Peak frequency
    m30_180=(freq>=30)&(freq<=180)
    f['peak_freq_30_180'] = float(freq[m30_180][np.argmax(pxx[m30_180])])
    return f

def shade(ax,k_on,k_off,defl,xl=None):
    ax.axvspan(k_on,k_off,color='#FFF8E1',alpha=0.9,zorder=0)
    ax.axvline(k_on, color='#F39C12',lw=1.2,ls='--',zorder=3)
    ax.axvline(k_off,color='#F39C12',lw=1.2,ls='--',zorder=3)
    ax.axvline(defl, color='#999',lw=0.7,ls=':',zorder=2)
    if xl: ax.set_xlim(xl)

# ── main ─────────────────────────────────────────────────────────────────────
CFGS = {
    1: dict(name='Subject 1 (Prof. Kan)', rec='Rec 06',
            rf=os.path.join(BASE,'Sub_1_Prof_kan','Rec_6.h5'),
            wav=os.path.join(BASE,'Sub_1_Prof_kan','sthethoscope_rec06.wav'),
            k_on=27.53, k_off=43.33, defl=18.0, t_max=52.0, lag=1.7083,
            notches=[100.71,201.43,302.14,402.86]),
    2: dict(name='Subject 2 (Rajveer)', rec='Rec 04',
            rf=os.path.join(BASE,'Sub_2_Rajveer','Rec_4.h5'),
            wav=os.path.join(BASE,'Sub_2_Rajveer','sthethoscope_rec04.wav'),
            k_on=27.38, k_off=42.00, defl=18.6, t_max=51.0, lag=2.6042,
            notches=[50.0,64.0,100.6,201.2]),
}

all_rows = []

for sid, c in CFGS.items():
    k_on,k_off,defl,t_max,lag = c['k_on'],c['k_off'],c['defl'],c['t_max'],c['lag']
    FS=10_000; DEC=10; fs=1000
    SCALE=(299792458./0.9e9*1000)/(4*np.pi)

    print(f"\n{'='*60}\n{c['name']}\n{'='*60}")

    # Load + preprocess
    with h5py.File(c['rf'],'r') as f: raw=f['data'][:]
    i_raw,q_raw=-raw[0],raw[1]
    xc,yc=fit_circle(i_raw,q_raw); ic,qc=i_raw-xc,q_raw-yc
    phi=robust_phase(ic,qc)
    sos_lp=butter(4,300.,btype='low',fs=FS,output='sos')
    mag=np.abs(sosfiltfilt(sos_lp,ic+1j*qc))
    phi=notch_chain(phi,c['notches'],FS)
    mag=notch_chain(mag,c['notches'],FS)

    # Korotkoff velocity (30-180 Hz)
    phi_vk=np.append(np.diff(bpf(phi,30,180,FS))*FS,0)*SCALE
    mag_vk=np.append(np.diff(bpf(mag,30,180,FS))*FS,0)
    # Compliance (0.4-3 Hz)
    phi_cp=decimate(bpf(phi,0.4,3.0,FS),DEC,ftype='fir')*SCALE
    mag_cp=decimate(bpf(mag,0.4,3.0,FS),DEC,ftype='fir')
    t_ds=np.arange(len(phi_cp))/fs
    t_rf=np.arange(len(phi_vk))/FS

    # Decimate vk to 1kHz
    phi_vk_ds=decimate(phi_vk,DEC,ftype='fir')
    mag_vk_ds=decimate(mag_vk,DEC,ftype='fir')

    # Stethoscope GT
    fs_a,aud=wavfile.read(c['wav'])
    aud=aud.astype(np.float64)/32768.
    if aud.ndim>1: aud=aud.mean(axis=1)
    aud_bp=bpf(aud,50.,1000.,fs_a)
    t_a=np.arange(len(aud_bp))/fs_a+lag

    # Window masks (1kHz)
    t_clean_s=defl+3.0; t_clean_e=k_off+1.5
    # Normal = 5s window just before Korotkoff
    m_norm=(t_ds>=k_on-6.0)&(t_ds<=k_on-1.0)
    # Korotkoff = full window
    m_koro=(t_ds>=k_on)&(t_ds<=k_off)
    # Steth window masks
    m_norm_a=(t_a>=k_on-6.0)&(t_a<=k_on-1.0)
    m_koro_a=(t_a>=k_on)&(t_a<=k_off)

    # Feature extraction for all 4 signal × 2 region combinations
    segs = {
        'Phi_VK_Normal'  : (phi_vk_ds[m_norm], fs),
        'Phi_VK_Koro'    : (phi_vk_ds[m_koro], fs),
        'Mag_VK_Normal'  : (mag_vk_ds[m_norm], fs),
        'Mag_VK_Koro'    : (mag_vk_ds[m_koro], fs),
        'Phi_CP_Normal'  : (phi_cp[m_norm], fs),
        'Phi_CP_Koro'    : (phi_cp[m_koro], fs),
        'Mag_CP_Normal'  : (mag_cp[m_norm], fs),
        'Mag_CP_Koro'    : (mag_cp[m_koro], fs),
        'Steth_Normal'   : (aud_bp[m_norm_a], fs_a),
        'Steth_Koro'     : (aud_bp[m_koro_a], fs_a),
    }
    feat = {k: features(v[0],v[1],k) for k,v in segs.items()}

    # Print key comparisons
    for sig in ['Phi_VK','Mag_VK','Phi_CP','Mag_CP']:
        fn,fk=feat[f'{sig}_Normal'],feat[f'{sig}_Koro']
        print(f"\n  {sig}")
        print(f"    RMS         Norm={fn['rms']:.4f}   Koro={fk['rms']:.4f}   ratio={fk['rms']/(fn['rms']+1e-12):.2f}x")
        print(f"    Kurtosis    Norm={fn['kurt']:.1f}   Koro={fk['kurt']:.1f}")
        print(f"    TKEO-P95    Norm={fn['tkeo_p95']:.4f}   Koro={fk['tkeo_p95']:.4f}   ratio={fk['tkeo_p95']/(fn['tkeo_p95']+1e-12):.2f}x")
        print(f"    PSD 30-60Hz Norm={fn['psd_30_60hz']:.4f}   Koro={fk['psd_30_60hz']:.4f}")
        print(f"    Peak freq   Norm={fn['peak_freq_30_180']:.1f}Hz   Koro={fk['peak_freq_30_180']:.1f}Hz")

    # Collect CSV rows
    for seg,f_dict in feat.items():
        row = {'subject': c['name'], 'segment': seg}
        row.update(f_dict)
        all_rows.append(row)

    # ── FIGURE: 8×2 ultra analysis ──────────────────────────────────────────
    CM='#1A6FC4'; CP='#C0392B'; CG='#1B7F4E'; CS='#0A7251'
    CK='#F39C12'; xl=(defl-1,t_max)

    fig=plt.figure(figsize=(24,32),dpi=300,facecolor='#FFFFFF')
    gs=gridspec.GridSpec(8,2,hspace=0.52,wspace=0.28,
                         left=0.07,right=0.97,top=0.96,bottom=0.02)

    # ROW 0: Full velocity waveforms
    for col,(sig,color,label) in enumerate([(phi_vk_ds,CP,'Phase Velocity'),(mag_vk_ds,CM,'Magnitude Velocity')]):
        ax=fig.add_subplot(gs[0,col])
        shade(ax,k_on,k_off,defl,xl)
        nv=sig/(np.std(sig)+1e-12)
        ax.plot(t_ds,nv,color=color,lw=0.35,alpha=0.6,rasterized=True)
        ax.set_ylim(-6,6); ax.set_title(f'({"A" if col==0 else "B"}) RF {label} 30-180 Hz [Full]')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Norm. Amplitude (σ)')
        ax.text(k_on+0.5,5.2,'Korotkoff',color=CK,fontsize=8,fontweight='bold',zorder=5)

    # ROW 1: Short-time TKEO energy (beat-level resolution 50ms)
    for col,(sig,color,label,fsig) in enumerate([
            (phi_vk_ds,CP,'Phase'),
            (mag_vk_ds,CM,'Magnitude')]):
        ax=fig.add_subplot(gs[1,col])
        shade(ax,k_on,k_off,defl,xl)
        tk=np.convolve(tkeo(sig),np.ones(int(0.05*fs))/int(0.05*fs),mode='same')
        mk=(t_ds>=k_on)&(t_ds<=k_off)
        mb=(t_ds>=22.)&(t_ds<=k_on-2.)
        base=np.percentile(tk[mb],5)
        tkn=np.clip((tk-base)/(np.max(tk[mk])+1e-12),0,None)
        ax.fill_between(t_ds,0,tkn,color=color,alpha=0.4)
        ax.plot(t_ds,tkn,color=color,lw=0.6)
        ax.set_title(f'({"C" if col==0 else "D"}) RF {label} — TKEO Burst Energy [50ms window]')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Norm. TKEO')

    # ROW 2: Compliance pulses (0.4-3 Hz)
    for col,(sig,color,label) in enumerate([(phi_cp,CP,'Phase'),(mag_cp,CM,'Magnitude')]):
        ax=fig.add_subplot(gs[2,col])
        shade(ax,k_on,k_off,defl,xl)
        ax.plot(t_ds,sig/(np.std(sig)+1e-12),color=color,lw=0.9,alpha=0.85)
        ax.set_title(f'({"E" if col==0 else "F"}) RF {label} — Compliance Pulse 0.4-3 Hz')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Norm. Displacement')

    # ROW 3: Rolling energy dB contrast
    def roll_edb(x,w,f): return 10*np.log10(np.convolve(x**2,np.ones(int(w*f))/int(w*f),mode='same')+1e-20)
    for col,(sig,color,label) in enumerate([(phi_vk_ds,CP,'Phase'),(mag_vk_ds,CM,'Magnitude')]):
        ax=fig.add_subplot(gs[3,col])
        shade(ax,k_on,k_off,defl,xl)
        edb=roll_edb(sig,0.3,fs)
        mb2=(t_ds>=defl+3)&(t_ds<=k_on-2.)
        base_e=np.percentile(edb[mb2],50)
        ax.plot(t_ds,edb,color=color,lw=0.7,alpha=0.85)
        ax.axhline(base_e,color='#555',lw=1.0,ls='--',label=f'Baseline {base_e:.0f} dB')
        ax.fill_between(t_ds,base_e,edb,where=(t_ds>=k_on)&(t_ds<=k_off),color=color,alpha=0.3,label='Koro excess')
        ax.set_title(f'({"G" if col==0 else "H"}) RF {label} — Rolling Energy [0.3s, dB]')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Energy (dB)')
        ax.legend(frameon=False,ncol=2)

    # ROW 4: PSD comparison Normal vs Korotkoff (both channels)
    nperseg=int(fs*1.5)
    for col,(vk_ds,color,label) in enumerate([(phi_vk_ds,CP,'Phase'),(mag_vk_ds,CM,'Magnitude')]):
        ax=fig.add_subplot(gs[4,col])
        fn,pk=welch(vk_ds[m_norm],fs=fs,nperseg=min(nperseg,np.sum(m_norm)))
        fk,pk_k=welch(vk_ds[m_koro],fs=fs,nperseg=min(nperseg,np.sum(m_koro)))
        mf=(fn>=5)&(fn<=200)
        ax.semilogy(fn[mf],pk[mf],color='#888',lw=1.0,ls='--',label='Normal (pre-Korotkoff)')
        ax.semilogy(fk[mf],pk_k[mf],color=color,lw=1.0,label='Korotkoff window')
        ax.fill_between(fn[mf],pk[mf],pk_k[mf],where=pk_k[mf]>pk[mf],color=color,alpha=0.2,label='Spectral gain')
        ax.set_title(f'({"I" if col==0 else "J"}) RF {label} — PSD: Normal vs Korotkoff')
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (log scale)')
        ax.legend(frameon=False)

    # ROW 5: Spectrogram (Phase + Magnitude side by side) — Korotkoff zoom
    zoom_s=max(0,k_on-5); zoom_e=min(t_max,k_off+2)
    for col,(vk_all,color,label) in enumerate([(phi_vk,CP,'Phase'),(mag_vk,CM,'Magnitude')]):
        ax=fig.add_subplot(gs[5,col])
        mzoom=(t_rf>=zoom_s)&(t_rf<=zoom_e)
        seg=vk_all[mzoom]
        f_sg,t_sg,Sxx=signal.spectrogram(seg,fs=FS,nperseg=1024,noverlap=900,scaling='density')
        mf2=(f_sg>=10)&(f_sg<=200)
        im=ax.pcolormesh(t_sg+zoom_s,f_sg[mf2],10*np.log10(Sxx[mf2]+1e-20),
                         shading='gouraud',cmap='inferno',vmin=-60,vmax=20)
        ax.axvline(k_on, color='white',lw=1.5,ls='--')
        ax.axvline(k_off,color='white',lw=1.5,ls='--')
        ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
        ax.set_title(f'({"K" if col==0 else "L"}) RF {label} — Spectrogram [10-200 Hz, zoom]')
        plt.colorbar(im,ax=ax,label='dB',pad=0.02)

    # ROW 6: Feature comparison bar charts
    feat_names=['rms','tkeo_p95','kurt','psd_30_60hz','psd_60_120hz']
    feat_labels=['RMS','TKEO P95','Kurtosis','PSD 30-60Hz','PSD 60-120Hz']
    for col,(sig_pfx,color,label) in enumerate([('Phi_VK',CP,'Phase'),('Mag_VK',CM,'Magnitude')]):
        ax=fig.add_subplot(gs[6,col])
        fn_v=[feat[f'{sig_pfx}_Normal'][k] for k in feat_names]
        fk_v=[feat[f'{sig_pfx}_Koro'][k]   for k in feat_names]
        x=np.arange(len(feat_names))
        # Normalise each feature to max for visual comparison
        mx=[max(a,b,1e-12) for a,b in zip(fn_v,fk_v)]
        fn_n=[v/m for v,m in zip(fn_v,mx)]
        fk_n=[v/m for v,m in zip(fk_v,mx)]
        ax.bar(x-0.2,fn_n,0.38,label='Normal',color='#AAA',alpha=0.8)
        ax.bar(x+0.2,fk_n,0.38,label='Korotkoff',color=color,alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(feat_labels,rotation=25,ha='right')
        ax.set_title(f'({"M" if col==0 else "N"}) RF {label} — Feature Comparison\nNorm vs Korotkoff (normalised to max)')
        ax.set_ylabel('Relative Value'); ax.legend(frameon=False)
        # Annotate ratio
        for i,(n,k) in enumerate(zip(fn_v,fk_v)):
            r=k/(n+1e-12)
            ax.text(i+0.2,fk_n[i]+0.02,f'{r:.1f}x',ha='center',fontsize=7,color=color)

    # ROW 7: Cross-modality comparison in Korotkoff (RF Phase + Mag + Steth)
    ax=fig.add_subplot(gs[7,:])
    shade(ax,k_on,k_off,defl,xl)
    def norm_env(x,mk2,mb2):
        tk=np.convolve(tkeo(x),np.ones(int(0.15*fs))/int(0.15*fs),mode='same')
        b=np.percentile(tk[mb2],5)
        return np.clip((tk-b)/(np.max(tk[mk2])+1e-12),0,None)
    mb3=(t_ds>=22.)&(t_ds<=k_on-2.)
    phi_env=norm_env(phi_vk_ds,m_koro,mb3)
    mag_env=norm_env(mag_vk_ds,m_koro,mb3)
    # Steth
    tk_a=np.convolve(tkeo(aud_bp),np.ones(int(0.15*fs_a))/int(0.15*fs_a),mode='same')
    mb_a2=(t_a>=22.)&(t_a<=k_on-2.)
    mk_a2=(t_a>=k_on)&(t_a<=k_off)
    ba2=np.percentile(tk_a[mb_a2],5)
    steth_e=np.clip((tk_a-ba2)/(np.max(tk_a[mk_a2])+1e-12),0,None)
    ax.fill_between(t_ds,0,phi_env,alpha=0.3,color=CP)
    ax.fill_between(t_ds,0,mag_env,alpha=0.25,color=CM)
    ax.plot(t_ds,phi_env,color=CP,lw=0.9,label='RF Phase TKEO')
    ax.plot(t_ds,mag_env,color=CM,lw=0.9,ls='--',label='RF Magnitude TKEO')
    t_xl=(t_a>=xl[0])&(t_a<=xl[1])
    ax.plot(t_a[t_xl],steth_e[t_xl],color=CS,lw=0.9,ls=':',alpha=0.9,label='GT Stethoscope TKEO')
    ax.set_xlim(xl); ax.set_ylim(-0.05,1.35)
    ax.set_title('(O) Cross-Modality: RF Phase vs RF Magnitude vs GT Stethoscope — TKEO Energy Envelopes')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Norm. TKEO Energy')
    ax.legend(frameon=False,ncol=3)

    fig.suptitle(f"Ultra Detailed RF Analysis: Magnitude vs Phase\n{c['name']} | {c['rec']}"
                 f"  |  Korotkoff: {k_on:.2f}–{k_off:.2f} s  ({k_off-k_on:.1f} s)",
                 fontsize=14,fontweight='bold',y=0.978)
    outf=os.path.join(OUT,f'ultra_analysis_Sub{sid}.png')
    plt.savefig(outf,dpi=300,facecolor='#FFFFFF',bbox_inches='tight')
    print(f"  Saved: {outf}")
    plt.close()

# ── Save CSV report ──────────────────────────────────────────────────────────
csv_path=os.path.join(OUT,'ultra_analysis_features.csv')
if all_rows:
    keys=['subject','segment']+[k for k in all_rows[0] if k not in ('subject','segment')]
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    print(f"\nCSV saved: {csv_path}")

print("\nAll done.")
