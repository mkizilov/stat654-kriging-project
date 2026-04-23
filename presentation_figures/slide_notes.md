# Slide notes - spatio-temporal kriging presentation

All figures are in `presentation_figures/` at 200 DPI.
A detailed speaking transcript is in `slide_script.md`.

---

## SLIDE 1 - Data overview

**Figures:** slide1_station_map.png, slide1_time_series.png, slide1_distance_elevation.png

- 29 real hourly NOAA weather stations around San Diego, fetched via Meteostat.
- Study window: 14-20 June 2020, matching the reference paper.
- Validation plan: hold out one real station (Ramona / Rosemont, KRNM0) and predict it from the other 28.
- Target Ramona is inland at 425 m; the other 28 stations span sea level to ~2000 m and sit 16-170 km away (median 98 km).
- Time-series figure shows Ramona as a thick red line against the 28 neighbours (thin lines, coloured by distance to target).

---

## SLIDE 2 - The variogram

**Figures:** slide2_variogram_1d.png, slide2_st_variogram.png

- The semivariogram gamma(h) measures average squared difference between pairs at lag h.
- Rises from 0 at h=0 to a sill (overall variance); the distance it plateaus is the range.
- 1-D teaching example: 60 samples from a spherical GP with known range=2.5, sill=1.0 - all 3 parametric fits recover within sampling noise.
- For space+time: sum-metric model
  gamma(h,u) = n * 1{h>0 or u>0} + gamma_s(h) + gamma_t(u) + gamma_j(sqrt(h^2 + (ku)^2))
- Fitted on elevation-detrended residuals (temperature variance in San Diego is dominated by elevation).

---

## SLIDE 3 - Building the prediction (method)

**Figure:** slide3_kriging_weights.png

- Kriging picks weights w_i so that Zhat(Ramona) = sum_i w_i Z_i, with sum(w_i)=1 (unbiased) and minimum prediction variance.
- Weights come from the fitted variogram - closer/more-correlated stations get larger positive weights.
- Per-station weights for one hour: dot size proportional to |weight|, colour indicates sign.
- Top-5 contributors carry about 85% of the weight: 0.46, 0.22, 0.19, 0.16, 0.13. Far stations get near-zero weight.

---

## SLIDE 4 - Building the prediction (result)

**Figures:** slide4_prediction_trajectory.png, slide4_prediction_zoom.png

- Black = actual observations at the held-out Ramona station. Red dashed = leave-one-out prediction from the other 28 stations.
- RMSE = 2.05 C, MAE = 1.70 C, R2 = 0.81 over 168 hours.
- Prediction tracks the diurnal cycle closely for the whole week; a bit smoother than reality because it averages neighbours.
- Zoom shows hour-by-hour tracking. Observations stay inside the 95% kriging band.

---

## SLIDE 5 - Measuring accuracy

**Figures:** slide5_rmse_mae_bars.png, slide5_scatter_and_errors.png, slide5_final_summary.png

- LOO-CV on every station, both seasons.
- Ordinary kriging (OK) vs regression kriging (RK, trend on elevation refit per fold).
- Summer: RK cuts pooled RMSE from 4.58 to 3.11 C. Fitted slope approx -5 to -7 C/km matches the atmospheric lapse rate.
- Autumn: the Sept 2020 heat wave dominated variance uniformly across elevation, so RK gives no further improvement.
- Hexbin (RK, summer): predictions tight on the 1:1 line, R2 = 0.83, error std approx 2.6 C centred near zero.
- Final summary bar chart covers every experiment in the project.
