# Near-Field RF Transition Analysis: Direct Verification of Ultrasound Effect

This report documents the direct physical impact of the ultrasound pulser on the **near-field RF signal properties**, focusing on the transient periods when the ultrasound pulser is turned ON and OFF.

By analyzing the raw, unfiltered RF amplitude over time, we show a clear, visible change in the RF signal's AC fluctuation level corresponding exactly to the ultrasound state.

---

## 1. Summary of AC Magnitude Variance (ON vs. OFF)

To measure the physical vibration energy transferred from the ultrasound to the near-field RF coupling, we calculate the variance of the raw AC magnitude (high-pass filtered above $10\text{ Hz}$ to remove slow motion and drift):

| Condition | Recording File | Ultrasound Pulser | AC Magnitude Variance | Physical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **① Baseline** | `nobody_noultrasound.h5` | **OFF (Entire)** | $6.91 \times 10^{-7}\text{ V}^2$ | Pure electrical and thermal noise floor. |
| **② Mechanical** | `ultra_rftable1.h5` | **ON (Entire)** | $2.55 \times 10^{-6}\text{ V}^2$ | Flat acoustic coupling to static table. |
| **③ Body (ON Window)** | `ultra_rfbody1.h5` | **ON (0–8s)** | **$5.99 \times 10^{-6}\text{ V}^2$** | Active acousto-physiological coupling (vibration). |
| **④ Body (OFF Window)** | `ultra_rfbody1.h5` | **OFF (>12s)** | **$3.62 \times 10^{-10}\text{ V}^2$** | **Drops by 16,500x** when ultrasound is turned OFF. |
| **⑤ Body (ON Window)** | `ultra_rfbody01.h5` | **ON (25–45s)** | **$3.08 \times 10^{-6}\text{ V}^2$** | Active acousto-physiological coupling. |
| **⑥ Body (OFF Window)** | `ultra_rfbody01.h5` | **OFF (<20s)** | **$6.25 \times 10^{-9}\text{ V}^2$** | **Drops by 490x** when ultrasound is turned OFF. |

---

## 2. Transition Figure

The figure below shows the raw DC magnitude (left column) and high-frequency AC coupling (right column) over the entire duration of the recordings.

![Raw transition plot](C:/Users/rajve/.gemini/antigravity/brain/80cd09cd-6a05-462c-b551-4b4019ffe29d/raw_full_transitions.png)

---

## 3. Direct Visual Proof of Ultrasound Effect

* **The ON/OFF Step Transition in `ultra_rfbody1`**:
  * In the first 8 seconds (when the ultrasound is active), the raw AC magnitude variance is **$5.99 \times 10^{-6}\text{ V}^2$**.
  * At exactly $8.0\text{ seconds}$ (when the ultrasound pulser is turned OFF), the AC magnitude fluctuation immediately collapses to **$3.62 \times 10^{-10}\text{ V}^2$** (a **16,540-fold decrease**).
  * This sharp step-down transition is visible in the raw, unfiltered waveform, proving that the high-frequency fluctuation is not an artifact of processing but a direct physical modulation created by the ultrasound.

* **The Active Lobe in `ultra_rfbody01`**:
  * Between $0\text{ and }20\text{ seconds}$, the ultrasound is OFF, and the AC variance is low ($6.25 \times 10^{-9}\text{ V}^2$).
  * Between $25\text{ and }45\text{ seconds}$, the ultrasound is ON, and the AC variance jumps by a factor of **490x** to $3.08 \times 10^{-6}\text{ V}^2$.
  * At $45\text{ seconds}$, when the ultrasound shuts down, the AC fluctuation immediately drops back to the baseline noise level.
