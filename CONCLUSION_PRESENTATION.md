---
title: "Rice Growth-Stage, Cropping Intensity & Production from S1+S2 Fusion"
subtitle: "MOGPR Sentinel-1 + Sentinel-2 — Conclusions (Java, 2024-2026)"
author: "Dr. Firman Hadi"
date: "June 2026"
aspectratio: 169
---

# The question

**Can we map rice growth stage — and from it cropping intensity and production —
better than with Sentinel-1 alone?**

- The TA-2026 reports built a strong **S1-only** paddy map (95% CV, ~3.0 M ha).
- But their **growth-stage** model was *not validated* (32% vs field data) and they
  parked **S1 + S2 fusion** as long-term future work.
- This work builds and validates that fusion.

# Approach

- **MOGPR** fuses Sentinel-1 VH (SAR, all-weather) + Sentinel-2 NDVI (optical) into one
  gap-filled curve per pixel.
- MOGPR is a **feature generator**, not the final model.
- A trained **MiniROCKET + LightGBM** classifier reads the fused curve -> 6 / 3 growth phases.
- Production = area in generative phase (4+5) x 5.8 t/ha, per 12-day period, Java-wide, 50 m.

# Finding 1 — Fusion beats SAR-only for phase

Generative-phase F1 (leave-one-region-out):

| Method | F1 |
|---|---|
| Rule-based on SAR curve | ~22% |
| **VH-only** | **52%** |
| **VH + MOGPR optical** | **67-68%** |
| idmai VH-CNN (in-sample) | 64% |

- **Optical is the dominant signal** for phase; SAR alone cannot carry it.
- This is the validated phase model the reports lacked.

# Finding 2 — Multi-season training is non-negotiable

A model trained only on the wet season **collapses** on the dry season:

| Training | Dry-season accuracy | Dry-season generative F1 |
|---|---|---|
| Wet-only | 36.9% | 60.7% |
| **Wet + dry (multi-season)** | **78.6%** | **91.6%** |

- Fixed using existing dry-season field labels (no new data needed).
- Independently reproduces the S1-only literature's central finding — on the fused signal.

# Finding 3 — Production (method demonstration)

| Year | Cropping intensity | Production |
|---|---|---|
| 2024 | 2.18 | **28.8 Mt** |
| 2025 | 2.22 | **29.4 Mt** |
| 2024-2025 combined | 2.22 | **~29.5 Mt/yr** |

- Harvest area ~5.0 M ha **= close to BPS luas panen (~5.34 M)** — an independent sanity check.
- Detected-paddy denominator (2.28 M ha), flat 5.8 t/ha.

# Finding 4 — A real interannual signal

![Triple-cropping (IP=3) concentrates in the Bengawan Solo corridor](output/production/ci3_highlight_2024_2025.png){width=85%}

- 2025 > 2024 (wetter, post-El-Nino) — confirmed real, not artefact.
- ~Half of Java's triple-cropping sits in the **Bengawan Solo** basin (water-secure year-round).

# Java cropping intensity — combined 2024-2025

![Annual cropping intensity, 24-month combined count](output/production/java_combined_2024_2025/java_cropping_intensity_combined.png){width=95%}

# Honest limits

- **Method demonstration, not official numbers:** detected-mask denominator + flat yield.
- Accuracy is **optimistic** — validated on a random (not spatial) split, same-campaign labels.
- 6-class is weak (~51%); robust products are **3-class** and the **generative** class.

# Conclusion

- **S1 + S2 MOGPR fusion materially improves rice growth-stage mapping over S1-only**, and turns
  it into a working **cropping-intensity and production** product for Java.
- It delivers — ahead of schedule — the fusion the program had parked as future work.
- The S1 binary-paddy results stand; this **adds** a validated phase -> IP -> production capability.

# Next step — the decisive one

**TA-2027 multi-DI field validation** (Klambu, Rentang, Jatiluhur, Semarang, E. Java, rainfed):

- Spatial leave-one-DI-out test (not random) + a dry-season campaign.
- Validate IP against tracked actual harvests; production vs KSA/BPS.
- Converts "promising demonstration" -> "operationally trustworthy."
- **Pilot already prepared:** 100 lag-adjusted Semarang points awaiting survey.
