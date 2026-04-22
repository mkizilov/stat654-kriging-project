# Slide notes - spatio-temporal kriging presentation

All figures are in `presentation_figures/` at 200 DPI. Drop each PNG into a slide and use the talking points below.

---

## SLIDE 1 - Data overview

**Figures:** slide1_station_map.png, slide1_time_series.png, slide1_distance_elevation.png

- 29 real hourly NOAA weather stations around San Diego, fetched via Meteostat.
- Study window: 14-20 June 2020, matching the reference paper.
- Target: HPWREN-like mountain site at 33.1 N, -116.6 W, ~860 m.
- Stations span sea level to ~2000 m and 30 km to 170 km from target.
- Colour in the time series = distance to target.
- Histograms: ~80 km median station-to-target distance; most stations below 500 m elevation with only one >1000 m.

---

## SLIDE 2 - The variogram

**Figures:** slide2_variogram_1d.png, slide2_st_variogram.png

- The semivariogram γ(h) measures the average squared difference between pairs of observations at lag h. Rises from 0 at h=0 to a sill (overall variance); the distance it plateaus at is the range.
- The 1-D figure is a *controlled teaching example*: 60 samples drawn from a spherical Gaussian process with known range=2.5, sill=1.0. All three classical models (spherical/exponential/gaussian) fit the empirical bins within sampling noise.
- For space + time we use the paper's sum-metric model:
  γ(h,u) = n·1{h>0 or u>0} + γ_s(h) + γ_t(u) + γ_j(√(h² + (k·u)²))
- We fit this on *elevation-detrended residuals* because raw temperature variance in San Diego is dominated by elevation.
- The fitted sum-metric surface (right) reproduces the overall empirical shape (left).

---

## SLIDE 3 - Building the prediction (method)

**Figure:** slide3_kriging_weights.png

- To test the method objectively we hold out one real station, **Ramona / Rosemont (KRNM0)**, and predict its 168 hourly temperatures from the other 28 stations.
- Ordinary kriging chooses weights w_i so that Ẑ(target) = Σᵢ wᵢ Zᵢ, with Σᵢ wᵢ = 1 (unbiasedness) and minimum prediction variance.
- The figure shows the per-station weights for a single hour: dot size ∝ |weight|, colour = sign. The five biggest contributors are labelled.
- The nearest 4-5 stations carry the prediction, with the single closest neighbour (≈10 km away) getting w ≈ 0.46. Far stations get near-zero weight. Weights sum to exactly 1.

---

## SLIDE 4 - Building the prediction (result)

**Figures:** slide4_prediction_trajectory.png, slide4_prediction_zoom.png

- Black = actual observations at Ramona / Rosemont. Red dashed = leave-one-out kriging prediction built from the other 28 stations.
- Prediction tracks the diurnal cycle closely across the whole week; it is a bit smoother than reality (expected — kriging averages neighbours and does not reproduce site-specific fluctuations).
- **RMSE = 2.05 °C, MAE = 1.70 °C, R² = 0.81** for 168 hours at this held-out station.
- Zoom shows the same data hour by hour; prediction lags the sharpest night-time dips slightly because warmer neighbours pull the mean up.
- The pink band = 95% kriging prediction interval; the actual observations stay inside it for the whole week.

---

## SLIDE 5 - Measuring accuracy

**Figures:** slide5_rmse_mae_bars.png, slide5_scatter_and_errors.png, slide5_final_summary.png

- We repeat the leave-one-out for every station (not just Ramona), both seasons.
- Bars compare ordinary kriging (OK) vs regression kriging (RK). RK first fits a linear elevation trend, kriges the residuals, and adds the trend back at the target — refit fold-by-fold to avoid leakage.
- **Summer 2020:** RK cuts pooled RMSE from 4.58 to 3.11 °C (≈33% reduction). Fitted slope ≈ −5 °C / km matches the atmospheric lapse rate.
- **Autumn 2020:** the Sept 2020 heat-wave dominated variance uniformly across elevation, so RK gives no further improvement.
- Hexbin (RK, summer): predictions vs observations tight on the 1:1 line; **RMSE = 2.58 °C, R² = 0.83** across all 29 stations × 168 hours.
- Error distribution is centred near zero with std ≈ 2.6 °C.
- Final summary bar chart: every RMSE/MAE produced in the project, from 0.15 °C (synthetic dense-sensor) to 8.6 °C (HPWREN single-sensor with location uncertainty).
