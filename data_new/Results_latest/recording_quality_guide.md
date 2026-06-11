# Recording Quality Guide for Korotkoff Detection

## Your Recording Comparison

| Recording | Duration | Carrier Offset | Phase Noise | IQ SNR | Quality |
|-----------|----------|----------------|-------------|--------|---------|
| `rec_koro_may11.h5` | 42.8s | **+1.1 Hz** | **0.011 rad/s** | **3.6** | ✅ **Best** |
| `rec_koro11_1.h5` | 36.0s | **+2.5 Hz** | 0.048 rad/s | 0.7 | ✅ Good |
| `rec_may12.h5` | 33.4s | **-5.9 Hz** | 0.076 rad/s | 0.7 | ⚠️ OK |
| `rec_may12_1.h5` | 29.9s | ❌ -484.7 Hz | 0.798 rad/s | 0.4 | ❌ Bad |
| `rec_may12_2.h5` | 26.2s | ❌ -375.3 Hz | 0.945 rad/s | 0.5 | ❌ Bad |

## What Makes a Good Recording

### The #1 Issue: Carrier Frequency Offset

The **carrier offset** is the dominant quality factor. Good recordings have < ±10 Hz offset; bad ones have hundreds of Hz.

**Root cause**: The USRP B210's TX and RX share the same local oscillator, so in CW radar mode the offset should be near zero. A large offset (375-485 Hz) means:
- The USRP clock wasn't properly locked/synced before recording started
- There was a significant TX/RX frequency mismatch during initialization

### Key Metrics to Target

| Metric | Good | Bad |
|--------|------|-----|
| Carrier offset | < ±10 Hz | > ±50 Hz |
| Phase noise (dphi std) | < 0.1 rad/sample | > 0.5 rad/sample |
| IQ SNR (mean/std of |IQ|) | > 1.0 | < 0.5 |

---

## Parameters to Set for Better Results

### 1. IF Frequency (`if_freq`) — Currently: 2000 Hz

> [!IMPORTANT]
> This is likely the main culprit for the large carrier offsets.

- `if_freq = [2000]` means the TX transmits at `carrier_freq + 2000 Hz = 0.9 GHz + 2 kHz`
- The received signal is mixed down, giving a **2 kHz IF tone** in the baseband
- If the system doesn't perfectly compensate this offset in software, you get a **residual carrier offset** in the IQ data

**Recommendation**: Try `if_freq = [0]` for **zero-IF (homodyne)** operation. This eliminates the IF offset entirely, giving a clean DC-centered IQ signal. The good recordings (`rec_koro_may11`, `rec_koro11_1`) likely used `if_freq = [0]` or a very small value.

### 2. TX/RX Gain — Currently: TX=40, RX=45

| Parameter | Current | Recommended Range |
|-----------|---------|-------------------|
| `tx_gain` | 40 dB | **30-40 dB** |
| `rx_gain` | 45 dB | **40-50 dB** |

- If RX gain is too high → saturation → phase noise spikes
- If too low → poor SNR → noisy phase
- The best recording (`rec_koro_may11`, IQ SNR=3.6) had significantly better amplitude stability
- **Start at TX=35, RX=45** and adjust until the IQ constellation looks clean (no saturation)

### 3. Recording Duration — Currently: ~25-35s

**Recommended: 35-45 seconds** with this protocol:
```
[0-5s]     Setup / cuff fully inflated, arm still (baseline)
[5-10s]    Begin slow cuff deflation  
[10-30s]   Korotkoff window (expect sounds here)
[30-35s]   Cuff fully deflated (quiet baseline)
[35-45s]   Post-recording baseline (no cuff)
```
- First 5s and last 5s are used as noise reference
- The middle 25s is the active measurement zone

### 4. Antenna/Probe Positioning

- Place antenna **directly over the brachial artery** (inner elbow)
- Distance: **2-5 cm** from skin
- Keep the arm **completely still** during recording
- Any bulk arm movement creates massive phase excursions

### 5. Before Each Recording: Warm-Up

> [!TIP]
> Run a 5-second "dummy" recording and discard it before the real one. This allows the USRP oscillator to stabilize and lock.

---

## Checklist Before Recording

```
□ USRP powered on for > 2 minutes (oscillator warm-up)
□ Run 1 dummy recording (5s) and discard
□ Verify IF frequency setting (try if_freq = [0])
□ TX gain = 35, RX gain = 45
□ Antenna positioned over brachial artery, 2-5cm distance
□ Cuff fully inflated before starting
□ Subject arm completely still
□ Record for 40+ seconds
□ Slow, steady cuff deflation (~2-3 mmHg/sec)
```

## Quick Test After Recording

Run this to check recording quality:
```python
import h5py, numpy as np
with h5py.File('your_recording.h5','r') as f: data=f['data'][:]
iq = (data[0,:]-data[0,:].mean()) + 1j*(data[1,:]-data[1,:].mean())
dphi = np.diff(np.unwrap(np.angle(iq)))
offset_hz = np.median(dphi)*10000/(2*np.pi)
noise = np.std(dphi)
snr = np.mean(np.abs(iq))/np.std(np.abs(iq))
print(f"Carrier offset: {offset_hz:+.1f} Hz {'✅' if abs(offset_hz)<10 else '❌'}")
print(f"Phase noise:    {noise:.4f} rad/s {'✅' if noise<0.1 else '❌'}")
print(f"IQ SNR:         {snr:.1f} {'✅' if snr>1.0 else '⚠️'}")
```
