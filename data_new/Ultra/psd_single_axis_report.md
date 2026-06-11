# Near-Field RF Shared-Axis PSD Analysis: Shared Y-Scale Comparison

This report details the Power Spectral Density (PSD) of the raw complex near-field RF signals, comparing the three conditions on a **single shared axis** to expose the true physical changes.

By plotting all three signals on the same axes without independent auto-scaling, we reveal a massive difference in signal amplitude and modulation behavior that was previously hidden.

---

## 1. Shared-Axis Power Spectrum Figure

The figure below shows the raw PSD over the full band ($-600\text{ to }600\text{ Hz}$, Panel A) and zoomed into the carrier region ($95\text{ to }105\text{ Hz}$, Panel B) on a unified Y-axis scale.

![Shared axis PSD plot](C:/Users/rajve/.gemini/antigravity/brain/80cd09cd-6a05-462c-b551-4b4019ffe29d/psd_single_axis.png)

---

## 2. Direct Visual & Physical Differences Exposed

### Difference 1: The Massive $60\text{ dB}$ Power Drop (Human Absorption)
When plotted on the same scale, the **US ON, Body** curve sits at the very bottom of the plot, showing a baseline power level of **$10^{-13}\text{ to }10^{-12}\text{ a.u./Hz}$**. 
* The **Table** ($10^{-6}\text{ to }10^{-5}\text{ a.u./Hz}$) and **No-US** ($10^{-7}\text{ to }10^{-6}\text{ a.u./Hz}$) controls sit **$60\text{ dB}$ higher** (1 million times more power).
* **Physical Reason**: Human tissue absorbs the near-field RF energy, drastically attenuating the backscattered wave compared to the highly reflective, metal-rich environment of the static laboratory.

### Difference 2: The 100 Hz Grid Hum vs. 100.71 Hz Carrier Peak
* **In the No-US baseline**: The peak at $100.7\text{ Hz}$ represents the environmental power grid switching noise (100 Hz hum second harmonic) leaking directly into the RF receiver circuitry.
* **In the Table control**: The peak at $100.7\text{ Hz}$ is **$3.5\text{ times stronger}$** ($3.90 \times 10^{-4}$ vs. $1.17 \times 10^{-4}\text{ a.u./Hz}$), showing the addition of the active acoustic carrier bouncing off the metal table.
* **In the Body condition**: The electrical hum and carrier are attenuated by the body blockage, leaving a low-power, broadened carrier peak.

### Difference 3: Spectral Line Broadening in the Body
Zooming in on the $95\text{ to }105\text{ Hz}$ region reveals that:
* The **Table** control shows an extremely sharp, narrow peak, reflecting a static propagation path.
* The **Body** condition shows a broadened peak with elevated sidebands, representing dynamic frequency modulation (FM) caused by physiological tissue movement.
