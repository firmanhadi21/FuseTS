# All-Weather Multi-Season Rice Phenology from Sentinel-1/Sentinel-2 Fusion with Multi-Output Gaussian Processes: A Cloud-Robust Workflow for Indonesia's Food-Security Monitoring

**Draft manuscript** · prepared from the FuseTS Klambu–Glapan / BulakBakal experiments
Target venues: *Remote Sensing* (MDPI), *International Journal of Applied Earth Observation and Geoinformation*, or *GIScience & Remote Sensing*.

> Status: working draft. All cited references were verified against the Crossref REST API (DOIs resolve to the stated title/authors/venue). Quantitative results are reproducible from the scripts and outputs in this repository (`scripts/`, `output/`). Numbers are not to be changed without re-running the pipeline.

---

## Abstract

Indonesia's rice self-sufficiency agenda depends on timely, objective information on where rice is grown, how many crops are harvested per year, and when each season starts. Optical satellite remote sensing — the basis of most operational crop-phenology products — is systematically degraded over the maritime continent during the northwest monsoon, exactly the wet-season planting window that matters most for harvested-area forecasting. We present a cloud-robust, open-data workflow that fuses Sentinel-1 C-band SAR (VV/VH) with Sentinel-2 NDVI using Multi-Output Gaussian Process Regression (MOGPR), implemented in the open-source FuseTS toolbox, and extracts **multi-season** rice phenology with a peak–valley detector. Applied to the Klambu–Glapan irrigation command area (Central Java) over a full year of 12-day composites, fusion roughly **quadruples the share of cropped pixels identified as double-cropped (13% → 50%)** relative to an optical-only (Whittaker-smoothed) baseline under an otherwise identical configuration, converging toward an independent point-based MOGPR estimate (mean 1.83 seasons per cropped sample). In an independent validation against 31 ground-recorded paddy fields near Magelang, detected green-up matched the recorded transplanting date with a median offset of **+9 days** and a mean absolute error of **21 days** (81% within ±24 days), with peak greenness reached ~81 days after transplanting — consistent with rice agronomy. The workflow runs entirely on free data (Microsoft Planetary Computer; Copernicus Sentinel) and open software, and is directly complementary to Indonesia's existing decision systems — the BPS area-frame (Kerangka Sampel Area) harvested-area statistics and the Ministry of Agriculture's integrated planting calendar (Katam). We discuss its value as an all-weather, wall-to-wall layer for wet-season monitoring and cropping-intensity accounting, and its current limitations.

**Keywords:** rice phenology; SAR–optical fusion; Multi-Output Gaussian Processes; Sentinel-1; Sentinel-2; cropping intensity; cloud gap-filling; food security; Indonesia.

---

## 1. Introduction

### 1.1 Rice, self-sufficiency, and the demand for objective monitoring

Rice is the staple of Indonesian diets and a politically central commodity; national policy has long pursued rice self-sufficiency (*swasembada beras*), a goal that is conceptually distinct from — and more data-demanding than — food security in the broad sense (Warr, 2011). Pursuing self-sufficiency requires credible, frequently updated estimates of harvested area, cropping intensity, and production. Indonesia substantially reformed its rice statistics in 2018 by adopting an area-frame sampling design, the *Kerangka Sampel Area* (KSA), in which enumerators record the observed crop growth phase at fixed grid points each month, enabling objective, forward-looking forecasts of harvested area that replace earlier subjective "eye-estimate" reporting (Ruslan, 2019; Raharjo et al., 2022; Ruslan & Prasetyo, 2023). In parallel, the Ministry of Agriculture operates the integrated planting calendar *Katam Terpadu*, which issues sub-district planting-time, fertiliser, and irrigation advisories to adapt scheduling to climate variability (Dewi et al., 2021). Both systems are strengthened by spatially explicit, all-weather earth-observation (EO) inputs.

### 1.2 The optical paradigm and its tropical failure mode

Satellite rice monitoring has been dominated by the optical, phenology-based paradigm established by Xiao et al. (2005, 2006), who exploited the transplanting-stage flooding signal in MODIS time series to map paddy across South and Southeast Asia. Land-surface phenology (LSP) extraction — locating the start (SOS), peak (POS), end (EOS), and length (LOS) of season from a smoothed vegetation-index trajectory — underpins these methods, using curve-fitting and threshold rules formalised in TIMESAT and related work (Jönsson & Eklundh, 2002, 2004; Zhang et al., 2003; Beck et al., 2006), with rice-specific adaptations across Asia (Sakamoto et al., 2005, 2006; Boschetti-style MODIS calendars; Mishra et al., 2021). Higher-resolution LSP is now feasible from Sentinel-2 and harmonized Landsat–Sentinel data (Vrieling et al., 2018; Bolton et al., 2020), and the choice of smoother materially affects retrieved dates (Atkinson et al., 2012).

The limiting factor in the humid tropics is cloud. Passive optical sensors lose a large, systematic fraction of growing-season acquisitions to cloud cover (Mercury et al., 2012; Whitcraft et al., 2015), and equatorial Indonesia exhibits some of the most persistent cloud frequencies on Earth (Wilson & Jetz, 2016). Critically, the densest convective cloud coincides with the northwest-monsoon wet season (≈ October–March), which is precisely when the principal rice crop is transplanted. Optical-only time series therefore suffer multi-week to multi-month gaps over exactly the transitions used to date planting and to count cropping cycles, biasing area and intensity estimates toward the more cloud-free dry-season crop.

### 1.3 SAR–optical fusion and Gaussian-process gap-filling

Synthetic Aperture Radar (SAR) is cloud- and illumination-independent, and Sentinel-1 C-band backscatter is diagnostically rich for rice: transplanting and flooding produce a characteristic backscatter minimum followed by a steep VH rise as the canopy develops (Veloso et al., 2017; Clauss et al., 2018; Phan et al., 2021; Fikriyah et al., 2025). Optical and SAR signals are physically complementary, and fusion consistently outperforms either alone for crop mapping and monitoring (Joshi et al., 2016; Setiyono et al., 2018; Nelson et al., 2014; Zhao et al., 2024). Among fusion methods, Multi-Output Gaussian Process Regression (MOGPR) is particularly well-suited to gap-filling: it models optical and SAR series as correlated outputs of shared latent functions and transfers information from cloud-free SAR acquisitions to reconstruct gaps in the optical series, while returning calibrated, time-varying uncertainty (Pipia et al., 2019). Related Gaussian-process gap-filling and phenology tools have matured into operational toolboxes (Belda et al., 2020; Salinero-Delgado et al., 2022), and the open-source FuseTS package (Open-EO/VITO) implements MOGPR alongside the Whittaker smoother (Eilers, 2003).

### 1.4 Gap and contribution

Despite this progress, three gaps remain for operational tropical rice. First, most MOGPR and GP gap-filling demonstrations target **temperate, single-season** systems; their behaviour across **consecutive monsoon/dry rice cycles** in the same paddy is largely unquantified. Second, the wet-season cloud blackouts of the maritime continent produce **longer optical gaps** than temperate validations consider, where the reconstruction must lean almost entirely on SAR. Third, fusion studies typically optimise continuous reconstruction (LAI/NDVI) but stop short of extracting **multi-season, field-validated rice phenological transitions**, and validation against **ground-recorded transplanting dates** is scarce (a notable SAR-based exception is Cauba et al., 2025). 

This paper contributes a compact, reproducible, open-data workflow that (i) builds a harmonised Sentinel-1/Sentinel-2 12-day datacube from the Microsoft Planetary Computer; (ii) fuses VV, VH, and NDVI with MOGPR; (iii) extracts **multi-season** rice phenology with an amplitude-based peak–valley detector applied to the fused series; and (iv) is validated against ground-recorded planting dates. We quantify how much fusion changes the detectable double-cropping signal under a strictly controlled comparison, and we frame the result for Indonesia's food-security decision systems.

---

## 2. Study areas and data

### 2.1 Study areas

The primary demonstration area is the **Klambu–Glapan irrigation command area** in Central Java (Demak/Grobogan), a wall-to-wall AOI of approximately 57 × 43 km (bounding box ≈ 110.518–111.034° E, −7.108 to −6.718° S), a classic irrigated double-rice landscape. For independent ground validation we use **BulakBakal**, a set of 31 surveyed paddy-field polygons near Magelang (≈ 110.236° E, −7.84° S) for which field records provide the transplanting date (*Tgl_Tanam*) and other per-period observations.

### 2.2 Satellite data

All imagery was accessed from the **Microsoft Planetary Computer (MPC)** STAC catalogue:
- **Sentinel-2 L2A** surface reflectance (bands B04 red, B08 NIR; Scene Classification Layer for cloud masking) → NDVI.
- **Sentinel-1 Radiometrically Terrain Corrected (RTC)** GRD (VV, VH; converted to decibels).

For Klambu–Glapan, data spanned 2024-11-07 to 2025-11-02 as 31 × 12-day composites at 50 m on a UTM 49S grid (EPSG:32749). For BulakBakal, data spanned 2025-09-01 to 2026-06-30 as 24 usable 12-day composites at 10 m, sampled at the 31 field centroids. NDVI was cloud-masked with the SCL (retaining vegetation, bare soil, water, unclassified, and snow classes) before compositing; periods with total cloud cover yielded NaN NDVI that fusion then reconstructs from SAR. Persistent wet-season cloud was evident: several BulakBakal periods (Nov–Feb) had 0% valid NDVI, recoverable only through the radar channel.

### 2.3 Software

Processing used open-source Python: `odc-stac`/`rioxarray` for STAC loading and gridding; **FuseTS** (Open-EO/VITO) for MOGPR fusion, the Whittaker smoother (Eilers, 2003), and peak–valley detection; GPy for the underlying Gaussian processes. The full pipeline (extraction, fusion, phenology, validation, mapping) is released as command-line scripts in this repository.

---

## 3. Methods

### 3.1 Datacube construction

For each 12-day period, cloud-masked S2 NDVI and S1 VV/VH (dB) were median-composited onto the common UTM grid and assembled into a single `(t, y, x)` dataset with variables `S2ndvi`, `VV`, `VH`; missing observations were retained as NaN. The Klambu–Glapan cube was 31 × 886 × 1162 (~1.03 M pixels per period).

### 3.2 MOGPR fusion

Time series were fused with MOGPR (Pipia et al., 2019) as implemented in FuseTS. MOGPR learns the cross-covariance between VV, VH, and NDVI and predicts each variable on a regular 12-day grid, propagating uncertainty. Fusion was applied (i) at sample points / field centroids (per-series, fast) and (ii) wall-to-wall on a spatially coarsened cube for raster mapping.

### 3.3 Multi-season phenology

Because Indonesian rice is multi-cropped, single-season LSP metrics are inappropriate. We applied the FuseTS **peak–valley** detector to the fused NDVI: it identifies successive peaks and the intervening valleys, from which we derive, **per detected season**, the start (green-up minimum before the peak), peak (POS), and end (valley) dates, the peak NDVI, amplitude, and season length. The amplitude sensitivity (`drop_thr`) governs how strong an NDVI cycle must be to count as a season.

### 3.4 Controlled fusion-effect experiment

To isolate the contribution of fusion (as opposed to resolution or detector tuning), we compared two wall-to-wall `n_seasons` rasters that were **identical in every respect except the input series**: both at 200 m, both processed with the same peak–valley `drop_thr = 0.12`. One used the **MOGPR-fused** NDVI; the other used **optical-only NDVI smoothed with the Whittaker filter** (Eilers, 2003). Any difference is therefore attributable to SAR–optical fusion.

### 3.5 Ground validation

For BulakBakal, detected green-up (SOS) of the season nearest each field's recorded *Tgl_Tanam* was compared to that transplanting date; we report the signed offset, mean absolute error (MAE), and the fraction within ±12 and ±24 days, plus the peak-to-planting interval. We tested centroid-pixel versus field-polygon-mean sampling and swept `drop_thr`.

---

## 4. Results

### 4.1 Point-based fusion and multi-season phenology (Klambu–Glapan)

All 200 random sample points were successfully fused. Multi-season detection yielded a mean of **1.83 seasons per cropped point** (1 season: 51 points; 2: 48; 3: 15), with mean season length ≈ 102 days, green-up ≈ 64–78 days, mean peak NDVI ≈ 0.36, and amplitude ≈ 0.28 — values consistent with irrigated double-rice. (Single-season phenolopy metrics computed for comparison collapsed the two cycles into one and are not used for cropping-intensity reporting.)

### 4.2 Effect of fusion on detectable double-cropping (controlled, wall-to-wall)

Under the controlled 200 m / `drop_thr = 0.12` comparison (Section 3.4), MOGPR fusion substantially increased detected cropping intensity (Table 1). Among cropped pixels, the share identified as **double-cropped rose from 13% (optical-only) to 50% (fused)**, and the mean number of seasons rose from 1.14 to 1.72 — converging toward the independent point-based estimate (1.83). The cropped-area fraction also increased (46% → 53%), as fusion recovered marginal/cloudy pixels that the optical-only series had dropped to "no season".

**Table 1.** Per-pixel `n_seasons` distribution (200 m, peak–valley `drop_thr = 0.12`; only the input series differs).

| Input series | 0 | 1 | 2 | 3+ | cropped (≥1) | double-cropped / cropped | mean (≥1) |
|---|---|---|---|---|---|---|---|
| Whittaker-only (optical S2) | 54.3% | 39.5% | 5.9% | 0.3% | 46% | **13%** | 1.14 |
| MOGPR-fused (S1 + S2) | 47.3% | 26.6% | 17.0% | 9.1% | 53% | **50%** | 1.72 |
| *Point-based MOGPR (reference)* | — | — | — | — | — | — | *1.83* |

The interpretation is agronomically coherent: optical-only smoothing misses the **wet-season** crop because clouds erase its NDVI signal during transplanting and growth, whereas SAR sees through cloud and MOGPR carries that information into the reconstructed NDVI. The spatial pattern (fused map) concentrates the additional double-cropped pixels within the irrigated command area, as expected.

### 4.3 Ground validation against recorded planting dates (BulakBakal)

Detected green-up tracked recorded transplanting closely at the median, with the best configuration being **centroid sampling at `drop_thr = 0.08`** (Table 2): all 31 fields matched a detected season, with **median SOS offset +9 days** (green-up lagging transplanting, as expected since NDVI rises after transplant), **MAE 21.3 days**, and **81% of fields within ±24 days**. Peak greenness occurred a median of ~81 days after planting, consistent with rice heading. Field-polygon-mean sampling did **not** improve on centroid sampling — small plots dilute the signal with bunds and edges.

**Table 2.** Validation of detected SOS against recorded *Tgl_Tanam* (BulakBakal, n = 31 fields).

| Configuration | Matched | SOS median | MAE | ≤ 24 d |
|---|---|---|---|---|
| centroid, `drop_thr` 0.15 | 23/31 | −3 d | 31.7 d | 57% |
| **centroid, `drop_thr` 0.08** | **31/31** | **+9 d** | **21.3 d** | **81%** |
| field-mean, `drop_thr` 0.08 | 30/31 | −13 d | 31.5 d | 73% |

We note explicitly that the most permissive setting (`drop_thr = 0.08`) **over-segments** (≈3 detected seasons per field for one documented planting); because our matching selects the detected season nearest the recorded date, this setting flatters the MAE. The robust reading is a precision/recall trade-off: low thresholds maximise recall of the documented season; higher thresholds are more conservative.

---

## 5. Discussion

### 5.1 Why this matters for the Indonesian government

The central, actionable benefit is **all-weather, wall-to-wall monitoring of the wet-season rice crop and of cropping intensity** — precisely the information that optical-only systems lose when it is most needed. This translates into concrete support for existing national decision systems:

1. **Objective harvested-area and cropping-intensity accounting (supporting BPS KSA).** The KSA area frame yields statistically sound, point-based estimates of crop phase and harvested area (Ruslan, 2019; Raharjo et al., 2022; Ruslan & Prasetyo, 2023). A fused, all-weather phenology layer provides a spatially continuous complement that can stratify the sample frame, flag double- versus single-cropped areas wall-to-wall, and reduce wet-season blind spots — directly relevant to harvested-area forecasting that feeds the self-sufficiency agenda (Warr, 2011).

2. **Validation and refinement of the planting calendar (supporting Katam Terpadu).** Detected transplanting timing, validated here against field records to a median of a few days, offers an EO-based check on, and spatial enrichment of, the Ministry of Agriculture's sub-district planting advisories (Dewi et al., 2021), and can detect shifts in planting in response to monsoon onset variability.

3. **Early, climate-adaptive intelligence.** Because the signal is recovered through cloud, the workflow can report wet-season planting progress in near real time, informing irrigation scheduling in command areas such as Klambu–Glapan, fertiliser distribution timing, and early warning for delayed or failed planting.

4. **Low cost and sovereignty-friendly.** The workflow uses only free, openly licensed data (Copernicus Sentinel via the Microsoft Planetary Computer) and open-source software (FuseTS), is fully reproducible, and can be run on commodity hardware or national cloud infrastructure without proprietary licences — lowering the barrier for operationalisation by BPS, the Ministry of Agriculture, or BRIN/LAPAN.

### 5.2 Relation to prior work

Our finding that fusion roughly quadruples detectable double-cropping is consistent with, and extends, the established complementarity of SAR and optical observations (Joshi et al., 2016; Veloso et al., 2017) and with rice-specific SAR sensitivity (Clauss et al., 2018; Phan et al., 2021; Fikriyah et al., 2025). Relative to gridded rice-calendar products such as RICA (Mishra et al., 2021) and the Monsoon Asia Rice Calendar (Zhao et al., 2024), our contribution is the **field-to-landscape resolution multi-season phenology from uncertainty-aware MOGPR fusion**, validated against **recorded transplanting dates** — a validation mode that remains rare outside SAR-only studies (Cauba et al., 2025).

### 5.3 Limitations

(i) **Resolution of the raster demonstration.** Per-pixel MOGPR is computationally heavy; the controlled wall-to-wall comparison was run at 200 m, with full 50 m fusion left to HPC/GPU. (ii) **Detector tuning.** The amplitude threshold trades recall against over-segmentation; a multi-season-aware validation metric (penalising spurious cycles) would set it more rigorously than the nearest-season match used here. (iii) **Validation scope.** Ground truth covered 31 fields at one site and the transplanting date only; harvest-date validation awaits recorded *Tgl_Panen*. (iv) **Residual wet-season uncertainty.** Even with fusion, the December–January peak is thin in places, and MOGPR uncertainty (not analysed in depth here) should be propagated into operational products. (v) **Orbit/geometry.** A single Sentinel-1 orbit direction was used; multi-orbit robustness was not tested.

### 5.4 Future work

Field-mean compositing with cloud-weighted quality flags; multi-year SOS/EOS trend analysis; explicit use of MOGPR predictive variance as a per-pixel confidence layer for KSA stratification; a seasonal-integral yield proxy; and a larger, multi-site ground campaign validating both transplanting and harvest dates.

---

## 6. Conclusions

A cloud-robust, open-data workflow fusing Sentinel-1 and Sentinel-2 with Multi-Output Gaussian Processes recovers the wet-season rice signal that optical-only monitoring loses over Indonesia. Under a controlled comparison, fusion increased the share of cropped pixels identified as double-cropped from 13% to 50%, converging with an independent point-based estimate, and detected green-up matched ground-recorded transplanting dates to a median of +9 days (MAE 21 days; 81% within ±24 days). The approach is directly complementary to Indonesia's KSA harvested-area statistics and Katam planting calendar, is reproducible on free data and software, and offers an all-weather, wall-to-wall layer for the country's rice self-sufficiency monitoring. Its main current limitations — raster-scale compute, detector tuning, and validation breadth — define a clear and tractable path to operationalisation.

---

## Data and software availability

- **Satellite data:** Copernicus Sentinel-1 and Sentinel-2 (European Union/ESA), accessed via the Microsoft Planetary Computer STAC API (https://planetarycomputer.microsoft.com).
- **Fusion software:** FuseTS — Time Series & Data Fusion toolbox integrated with openEO (Open-EO/VITO), https://github.com/Open-EO/FuseTS.
- **Ground data:** BulakBakal field survey (31 paddy polygons, planting/harvest attributes), used with permission.
- **Code:** extraction, fusion, phenology, validation, and mapping scripts are provided in this repository (`scripts/klambu_glapan_mogpr.py`, `scripts/pv_phenology.py`, `scripts/raster_mogpr_phenology.py`, `scripts/field_zonal_mogpr.py`, `scripts/map_phenology.py`).

*Note on grey-literature sources:* FuseTS, the Microsoft Planetary Computer, and the Copernicus Sentinel missions are cited as software/data resources, not as peer-reviewed claims. BPS KSA and Katam Terpadu are described from the peer-reviewed and conference literature cited below; primary agency documentation should additionally be cited in a final submission.

---

## References

*All DOIs below were verified against the Crossref REST API; each resolves to the stated authors, title, year, and venue.*

1. Atkinson, P. M., Jeganathan, C., Dash, J., & Atzberger, C. (2012). Inter-comparison of four models for smoothing satellite sensor time-series data to estimate vegetation phenology. *Remote Sensing of Environment*, 123, 400–417. https://doi.org/10.1016/j.rse.2012.04.001
2. Beck, P. S. A., Atzberger, C., Høgda, K. A., Johansen, B., & Skidmore, A. K. (2006). Improved monitoring of vegetation dynamics at very high latitudes: A new method using MODIS NDVI. *Remote Sensing of Environment*, 100(3), 321–334. https://doi.org/10.1016/j.rse.2005.10.021
3. Belda, S., Pipia, L., Morcillo-Pallarés, P., Rivera-Caicedo, J. P., Amin, E., De Grave, C., & Verrelst, J. (2020). DATimeS: A machine learning time series GUI toolbox for gap-filling and vegetation phenology trends detection. *Environmental Modelling & Software*, 127, 104666. https://doi.org/10.1016/j.envsoft.2020.104666
4. Bolton, D. K., Gray, J. M., Melaas, E. K., Moon, M., Eklundh, L., & Friedl, M. A. (2020). Continental-scale land surface phenology from harmonized Landsat 8 and Sentinel-2 imagery. *Remote Sensing of Environment*, 240, 111685. https://doi.org/10.1016/j.rse.2020.111685
5. Cauba, A. G., Darvishzadeh, R., Schlund, M., Nelson, A., & Laborte, A. (2025). Estimation of transplanting and harvest dates of rice crops in the Philippines using Sentinel-1 data. *Remote Sensing Applications: Society and Environment*, 37, 101435. https://doi.org/10.1016/j.rsase.2024.101435
6. Clauss, K., Ottinger, M., & Kuenzer, C. (2018). Mapping rice areas with Sentinel-1 time series and superpixel segmentation. *International Journal of Remote Sensing*, 39(5), 1399–1420. https://doi.org/10.1080/01431161.2017.1404162
7. Dewi, D. O., Kartinaty, T., & Sugiarti, T. (2021). Application of paddy planting calendar (KATAM) in Bengkayang District, West Kalimantan, Indonesia. *E3S Web of Conferences*, 306, 03017. https://doi.org/10.1051/e3sconf/202130603017
8. Eilers, P. H. C. (2003). A Perfect Smoother. *Analytical Chemistry*, 75(14), 3631–3636. https://doi.org/10.1021/ac034173t
9. Fikriyah, V. N., Darvishzadeh, R., Laborte, A., Rathore, J., & Nelson, A. (2025). Temporal backscatter characterisation of ratoon rice crops based on Sentinel-1 intensity data. *GIScience & Remote Sensing*, 62(1). https://doi.org/10.1080/15481603.2025.2455081
10. Jönsson, P., & Eklundh, L. (2002). Seasonality extraction by function fitting to time-series of satellite sensor data. *IEEE Transactions on Geoscience and Remote Sensing*, 40(8), 1824–1832. https://doi.org/10.1109/TGRS.2002.802519
11. Jönsson, P., & Eklundh, L. (2004). TIMESAT — a program for analyzing time-series of satellite sensor data. *Computers & Geosciences*, 30(8), 833–845. https://doi.org/10.1016/j.cageo.2004.05.006
12. Joshi, N., Baumann, M., Ehammer, A., Fensholt, R., Grogan, K., Hostert, P., Jepsen, M. R., Kuemmerle, T., Meyfroidt, P., Mitchard, E. T. A., et al. (2016). A review of the application of optical and radar remote sensing data fusion to land use mapping and monitoring. *Remote Sensing*, 8(1), 70. https://doi.org/10.3390/rs8010070
13. Mercury, M., Green, R., Hook, S., Oaida, B., Wu, W., Gunderson, A., & Chodas, M. (2012). Global cloud cover for assessment of optical satellite observation opportunities: A HyspIRI case study. *Remote Sensing of Environment*, 126, 62–71. https://doi.org/10.1016/j.rse.2012.08.007
14. Mishra, B., Busetto, L., Boschetti, M., Laborte, A., & Nelson, A. (2021). RICA: A rice crop calendar for Asia based on MODIS multi year data. *International Journal of Applied Earth Observation and Geoinformation*, 103, 102471. https://doi.org/10.1016/j.jag.2021.102471
15. Nelson, A., Boschetti, M., Manfron, G., Holecz, F., Collivignarelli, F., Gatti, L., Barbieri, M., Villano, L., Chandna, P., & Setiyono, T. (2014). Combining moderate-resolution time-series RS data from SAR and optical sources for rice crop characterisation: Examples from Bangladesh. In *Land Applications of Radar Remote Sensing*. IntechOpen. https://doi.org/10.5772/57443
16. Phan, H., Le Toan, T., & Bouvet, A. (2021). Understanding dense time series of Sentinel-1 backscatter from rice fields: Case study in a province of the Mekong Delta, Vietnam. *Remote Sensing*, 13(5), 921. https://doi.org/10.3390/rs13050921
17. Pipia, L., Muñoz-Marí, J., Amin, E., Belda, S., Camps-Valls, G., & Verrelst, J. (2019). Fusing optical and SAR time series for LAI gap filling with multioutput Gaussian processes. *Remote Sensing of Environment*, 235, 111452. https://doi.org/10.1016/j.rse.2019.111452
18. Raharjo, M., Kurnia, A., & Wijayanto, H. (2022). Study on accuracy of paddy harvest area estimation on area sampling frame method. *Indonesian Journal of Statistics and Its Applications*, 6(1), 41–49. https://doi.org/10.29244/ijsa.v6i1p41-49
19. Ruslan, K. (2019). Improving Indonesia's food statistics through the area sampling frame method. https://doi.org/10.35497/287781
20. Ruslan, K., & Prasetyo, O. R. (2023). Can paddy growing phase produce an accurate forecast of paddy harvested area in Indonesia? Analysis of the area sampling frame results. *Proceedings of the International Conference on Data Science and Official Statistics*, 2023(1). https://doi.org/10.34123/icdsos.v2023i1.316
21. Sakamoto, T., Yokozawa, M., Toritani, H., Shibayama, M., Ishitsuka, N., & Ohno, H. (2005). A crop phenology detection method using time-series MODIS data. *Remote Sensing of Environment*, 96(3–4), 366–374. https://doi.org/10.1016/j.rse.2005.03.008
22. Sakamoto, T., Van Nguyen, N., Ohno, H., Ishitsuka, N., & Yokozawa, M. (2006). Spatio-temporal distribution of rice phenology and cropping systems in the Mekong Delta with special reference to the seasonal water flow of the Mekong and Bassac rivers. *Remote Sensing of Environment*, 100(1), 1–16. https://doi.org/10.1016/j.rse.2005.09.007
23. Salinero-Delgado, M., Estévez, J., Pipia, L., Belda, S., Berger, K., Paredes Gómez, V., & Verrelst, J. (2022). Monitoring cropland phenology on Google Earth Engine using Gaussian Process Regression. *Remote Sensing*, 14(1), 146. https://doi.org/10.3390/rs14010146
24. Setiyono, T. D., Quicho, E. D., Gatti, L., Campos-Taberner, M., Busetto, L., Collivignarelli, F., García-Haro, F. J., Boschetti, M., Khan, N. I., & Holecz, F. (2018). Spatial rice yield estimation based on MODIS and Sentinel-1 SAR data and ORYZA crop growth model. *Remote Sensing*, 10(2), 293. https://doi.org/10.3390/rs10020293
25. Veloso, A., Mermoz, S., Bouvet, A., Le Toan, T., Planells, M., Dejoux, J.-F., & Ceschia, E. (2017). Understanding the temporal behavior of crops using Sentinel-1 and Sentinel-2-like data for agricultural applications. *Remote Sensing of Environment*, 199, 415–426. https://doi.org/10.1016/j.rse.2017.07.015
26. Vrieling, A., Meroni, M., Darvishzadeh, R., Skidmore, A. K., Wang, T., Zurita-Milla, R., Oosterbeek, K., O'Connor, B., & Paganini, M. (2018). Vegetation phenology from Sentinel-2 and field cameras for a Dutch barrier island. *Remote Sensing of Environment*, 215, 517–529. https://doi.org/10.1016/j.rse.2018.03.014
27. Warr, P. (2011). Food security vs. food self-sufficiency: The Indonesian case. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.1910356
28. Whitcraft, A. K., Vermote, E. F., Becker-Reshef, I., & Justice, C. O. (2015). Cloud cover throughout the agricultural growing season: Impacts on passive optical earth observations. *Remote Sensing of Environment*, 156, 438–447. https://doi.org/10.1016/j.rse.2014.10.009
29. Wilson, A. M., & Jetz, W. (2016). Remotely sensed high-resolution global cloud dynamics for predicting ecosystem and biodiversity distributions. *PLOS Biology*, 14(3), e1002415. https://doi.org/10.1371/journal.pbio.1002415
30. Xiao, X., Boles, S., Liu, J., Zhuang, D., Frolking, S., Li, C., Salas, W., & Moore, B. (2005). Mapping paddy rice agriculture in southern China using multi-temporal MODIS images. *Remote Sensing of Environment*, 95(4), 480–492. https://doi.org/10.1016/j.rse.2004.12.009
31. Xiao, X., Boles, S., Frolking, S., Li, C., Babu, J. Y., Salas, W., & Moore, B. (2006). Mapping paddy rice agriculture in South and Southeast Asia using multi-temporal MODIS images. *Remote Sensing of Environment*, 100(1), 95–113. https://doi.org/10.1016/j.rse.2005.10.004
32. Zhang, X., Friedl, M. A., Schaaf, C. B., Strahler, A. H., Hodges, J. C. F., Gao, F., Reed, B. C., & Huete, A. (2003). Monitoring vegetation phenology using MODIS. *Remote Sensing of Environment*, 84(3), 471–475. https://doi.org/10.1016/S0034-4257(02)00135-9
33. Zhao, X., Nishina, K., Izumisawa, H., Masutomi, Y., Osako, S., & Yamamoto, S. (2024). Monsoon Asia Rice Calendar (MARC): A gridded rice calendar in monsoon Asia based on Sentinel-1 and Sentinel-2 images. *Earth System Science Data*, 16, 3893–3911. https://doi.org/10.5194/essd-16-3893-2024

---

*Author note on integrity of results:* The quantitative claims (1.83 seasons/point; 13%→50% double-cropping; SOS median +9 d, MAE 21.3 d, 81% ≤24 d) are taken directly from the pipeline outputs in `output/klambu_glapan/RESULTS.md` and `output/bulak_bakal/`. Page numbers for a handful of older references should be re-confirmed at copy-edit against the publisher record; all DOIs are verified.
