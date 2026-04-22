# Slide notes — spatio-temporal kriging presentation

Figures are under `presentation_figures/`. Drop each PNG on a slide and use these talking points.

---

## SLIDE 1 — Data overview

**Figures:** `slide1_station_map.png`, `slide1_time_series.png`, `slide1_distance_elevation.png`

**Talking points:**

- **29 hourly NOAA weather stations** around San Diego (fetched with the `meteostat` Python library).
- Two one-week windows matching the reference paper: **14–20 June 2020** and **14–20 September 2020**.
- Target: an HPWREN-like site at **33.1 °N, −116.6 °W, ~860 m elevation** (inferred from the Kaggle *Minute Weather* air-pressure signature).
- Stations span **sea level to ~2 000 m** elevation and **4 km to 170 km** from the target — a much more heterogeneous geometry than the paper's Taiwan greenhouse network.
- All 29 stations show the expected **diurnal cycle** but with station-dependent amplitude; colour encodes distance to target.

---

## SLIDE 2 — The variogram

**Figures:** `slide2_variogram_1d.png`, `slide2_st_variogram.png`

**Talking points:**

- Start with 1-D intuition: the **semivariogram γ(h)** measures how dissimilar pairs of observations are at lag $h$. Three classical models — spherical, exponential, Gaussian — all fit the empirical bins reasonably.
- Key parameters: **nugget** (micro-scale variance at lag 0), **sill** (long-range variance), **range** (distance where correlation is lost).
- To handle *both* space and time we use the paper's **sum-metric** model:

  $$\gamma(h, u) = n\,\mathbf{1}\{h{>}0 \text{ or } u{>}0\} + \gamma_s(h) + \gamma_t(u) + \gamma_j\!\left(\sqrt{h^2 + (ku)^2}\right).$$

- The right panel of `slide2_st_variogram.png` shows the fitted 2-D surface $\gamma(h, u)$ alongside the empirical one; they match well, which is what we want before feeding $\gamma$ into the kriging linear system.

---

## SLIDE 3 — Building the prediction (method)

**Figure:** `slide3_kriging_weights.png`

**Talking points:**

- Ordinary kriging predicts the target temperature as a **weighted sum** of station observations: $\hat Z(\mathbf{s}_0) = \sum_i w_i \, Z(\mathbf{s}_i)$, with weights summing to 1 (unbiasedness).
- Weights come directly from the variogram: closer / more-correlated stations get **larger positive** weights; far-away stations in uncorrelated directions can even get small **negative** weights (subtracting the regional trend).
- The figure shows the kriging weights across the 29 stations for a single prediction time; dot *size* encodes magnitude, dot *colour* encodes sign. The nearest inland stations dominate, coastal stations contribute small negative corrections.

---

## SLIDE 4 — Building the prediction (result)

**Figures:** `slide4_prediction_trajectory.png`, `slide4_prediction_zoom.png`

**Talking points:**

- The solid red line is the kriging estimate of the target's hourly temperature for the full 168-hour week, computed independently at each hour with the sum-metric variogram.
- Gray background: the observations from all 29 stations. The prediction sits inside their envelope, tracing the regional diurnal cycle.
- The red shaded band is the **95% kriging prediction interval**. It is narrow near peaks and troughs because many stations constrain those moments; it widens slightly between them.
- The second figure zooms into a 48-hour window to show the uncertainty band and the hourly prediction points clearly.

---

## SLIDE 5 — Measuring accuracy

**Figures:** `slide5_rmse_mae_bars.png`, `slide5_scatter_and_errors.png`, `slide5_final_summary.png`

**Talking points:**

- Validation: **leave-one-station-out** cross-validation on each of the 29 stations, for 168 hours each.
- Ordinary kriging (OK) vs **regression kriging (RK)** with an elevation trend removed before kriging — both trend and variogram are refit *inside* each LOO fold, so the evaluation has no data leakage.
- Summer 2020: OK RMSE ≈ 4.6 °C → RK RMSE ≈ 3.1 °C (≈ 33 % improvement). Autumn 2020: RK ≈ OK (the Sept heat-wave dominates variance uniformly across elevation).
- The predicted-vs-observed hexbin and the error-distribution histogram show most errors inside ±3 °C and centred near zero.
- The final summary bar chart collects **every** RMSE / MAE produced in the project — from the 6× synthetic gap down to the real-data LOO numbers and the HPWREN sensor benchmark.
