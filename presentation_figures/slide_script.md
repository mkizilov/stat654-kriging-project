# Presentation script — spatio-temporal kriging

Total plan: **~10 min of speaking** across 5 content slides + title + conclusion. Each section below gives (a) what to put **on** the slide and (b) what to **say** while it is shown.

---

## Title slide

**Put on the slide:**

> # Spatio-Temporal Kriging for Local Temperature Prediction
>
> STAT 654 class project · Mykyta Kizilov
>
> Based on **Kuo, Huang & Putra (2021), Sensors 21(5):1853**

**Say (≈20 s):**

> "Today I'll walk through how we can statistically predict the temperature at a site where we do not have a sensor, using hourly observations from surrounding weather stations. The method is **spatio-temporal kriging**. The project follows the setup of Kuo, Huang and Putra in their 2021 paper that compared public weather-station data with on-site greenhouse sensors."

---

## SLIDE 1 — Data overview

**Put on the slide:**

> ### 29 NOAA stations around San Diego, 14–20 Jun 2020
>
> - 29 real hourly stations (NOAA / Meteostat)
> - Study window matches the paper: 14–20 June 2020
> - Target: HPWREN-like mountain site at **33.1 °N, −116.6 °W, ~860 m**
> - Elevation range across network: **sea level → ~2 000 m**
> - Distance to target: **30–170 km** (median 82 km)
>
> *Images:* `slide1_station_map.png`, `slide1_time_series.png`, `slide1_distance_elevation.png`

**Say (≈75 s):**

> "I pulled real hourly temperature data from 29 NOAA weather stations around San Diego, using the free Meteostat Python library. The one-week window — 14th to 20th of June 2020 — is the same window the paper used, so our results are directly comparable.
>
> [point to the map] Here's the network. Each dot is a station, coloured by elevation: dark blue is sea level, green is a few hundred metres, and the single near-white dot at the top is a mountain peak at about 2 000 m. The red star is the target where we want to predict temperature — we chose a mountain-style location at around 860 m.
>
> [point to the time-series plot] All 29 stations show the expected daily cycle over 7 days, but with very different amplitudes. The coastal dark-purple stations stay fairly flat between 15 and 25 °C because of the Pacific marine layer. The inland yellow-green stations swing from about 5 °C at night to almost 40 °C at mid-day.
>
> [point to the histograms] Most stations sit below 500 m, and the median station is about 82 km from the target. That terrain heterogeneity is much larger than the flat Taiwan greenhouse region in the original paper — and it turns out to matter a lot for how well kriging performs."

---

## SLIDE 2 — The semivariogram

**Put on the slide:**

> ### Measuring correlation: the semivariogram γ(h, u)
>
> γ(h) = ½ · average of (Zᵢ − Zⱼ)²  over pairs at lag h
>
> - Rises from **0** to a **sill** (total variance)
> - Distance at which it plateaus = **range**
> - Sum-metric space+time model (paper eq. 5):
>   γ(h,u) = n·𝟙{h>0 ∨ u>0} + γ_s(h) + γ_t(u) + γ_j(√(h² + (k·u)²))
> - Fit on elevation-detrended residuals
>
> *Images:* `slide2_variogram_1d.png`, `slide2_st_variogram.png`

**Say (≈90 s):**

> "Before we krige anything we need a model of how temperatures are correlated across space and time. That's the semivariogram.
>
> [left plot] Here's the intuition on a 1-D toy example: 60 samples drawn from a known Gaussian process. The black dots are the empirical semivariogram — we just compute the squared difference between pairs of points at each lag h, bin them, and average. You can see the textbook shape: near 0 at small lag, rises, and flattens out at a sill. The distance at which it flattens is called the range.
>
> Three classical parametric models — spherical, exponential, gaussian — all fit well and all recover the true range of 2.5.
>
> [right plot] Now the same idea in 2-D on our real data: spatial lag h on the x-axis, temporal lag u on the y-axis. I use the **sum-metric** model from Kuo et al.: a sum of three component variograms plus a nugget, with an anisotropy scale k that couples space and time in units of kilometres per hour — you can see the equation on the slide.
>
> The left panel is the empirical estimate, the right is the fitted sum-metric model. They agree on the overall shape — low at the origin, growing with both lags.
>
> I fit on elevation-detrended residuals because otherwise the variance across San Diego is dominated by the elevation gradient, which would wash out the spatio-temporal signal we actually want to model."

---

## SLIDE 3 — How kriging combines stations

**Put on the slide:**

> ### Kriging = weighted average of observations
>
> Prediction:  Ẑ(target) = Σᵢ wᵢ · Zᵢ
>
> - Weights **must sum to 1** (unbiasedness)
> - Chosen to minimise prediction variance
> - Derived from the fitted variogram
> - Demo: hold out **Ramona / Rosemont** (KRNM0), predict from other 28 stations
> - Top-5 neighbours carry ≈85 % of the weight
>
> *Image:* `slide3_kriging_weights.png`

**Say (≈75 s):**

> "Ordinary kriging predicts the target as a weighted linear combination of the surrounding observations. Two rules: the weights sum to 1 for unbiasedness, and they are chosen to minimise the prediction variance. The weights themselves come from the fitted variogram — closer and more correlated stations get bigger weights.
>
> [point to the map] To make this concrete, I held out one real station — Ramona / Rosemont, an inland site at about 425 m — and asked kriging to predict its temperature using only the other 28 stations. This figure shows the per-station kriging weights for a single hour.
>
> Dot size is the magnitude of the weight, colour is the sign — red positive, blue negative. The five biggest contributors are labelled.
>
> The single closest station gets weight 0.46. The next four get 0.22, 0.19, 0.16, and 0.13. Together the top 5 carry about 85 % of the mass. Distant stations get nearly zero weight — they're too decorrelated to help. And the weights sum to exactly 1.
>
> So kriging is a principled weighted average whose weights come from the data's own spatial structure — it is not equal averaging, and it is not nearest neighbour."

---

## SLIDE 4 — Does it actually predict well?

**Put on the slide:**

> ### Leave-one-out prediction at Ramona / Rosemont
>
> - Target held out, predicted from the **other 28 stations**
> - 168 hours over 7 days
> - Prediction tracks the diurnal cycle
> - Observations stay inside the 95 % kriging band
>
> **Accuracy:** RMSE = **2.05 °C** · MAE = **1.70 °C** · R² = **0.81**
>
> *Images:* `slide4_prediction_trajectory.png`, `slide4_prediction_zoom.png`

**Say (≈80 s):**

> "So does it actually predict well? Here's the answer.
>
> [full-week plot] Black is what the station actually measured over the 7 days. Red dashed is the kriging prediction, built only from the other 28 stations — the target itself was completely held out. The prediction tracks the diurnal cycle really well: cool nights around 14 °C, warm afternoons in the mid-20s. Over all 168 hours, the RMSE is **2.05 °C**, MAE is 1.70, and R² is 0.81.
>
> [zoom plot] Here's a 48-hour zoom to see the hourly tracking. The prediction is a bit smoother than reality — it's an average of neighbours, so site-specific fluctuations get dampened. The night-time lows lag slightly because warmer neighbours pull the mean up. But the shape and the timing are right, and the 95 % kriging prediction band in pink covers the real observations throughout.
>
> This is the headline result: from only regional weather-station data, kriging recovers the hourly temperature trace at an arbitrary held-out site to about **2 °C RMSE**."

---

## SLIDE 5 — Measuring accuracy across the whole network

**Put on the slide:**

> ### How accurate on average, and what helps?
>
> - Leave-one-out on **all 29 stations**, two seasons
> - Ordinary kriging (OK) vs Regression kriging (RK, trend on elevation)
>
> | Season | OK RMSE | RK RMSE |
> |---|---|---|
> | Summer 2020 | 4.58 | **3.11** (−33 %) |
> | Autumn 2020 | 4.68 | 4.68 (heat wave) |
>
> - RK fitted lapse rate ≈ **−5 to −7 °C / km**  ←  matches atmospheric lapse rate
> - Pooled RK scatter (summer): **R² = 0.83**
>
> *Images:* `slide5_rmse_mae_bars.png`, `slide5_scatter_and_errors.png`, `slide5_final_summary.png`

**Say (≈90 s):**

> "Ramona was one example — now let's do it for every one of the 29 stations, both seasons, and compare two kriging flavours.
>
> [left plot, bars] Ordinary kriging in red: pooled RMSE of about 4.6 °C in both seasons. The reason is that ordinary kriging cannot represent the atmospheric lapse rate — if you try to predict a mountain station from coastal neighbours, you under-predict the elevation cooling.
>
> Regression kriging in green fixes this: it fits a linear trend on elevation first, kriges the residuals, and adds the trend back at the prediction point. Crucially, we refit both the trend and the variogram inside every leave-one-out fold, so there is no data leakage.
>
> In **summer** regression kriging cuts RMSE from 4.58 to 3.11 — a 33 % improvement. The fitted elevation slope comes out at about −5 to −7 °C per kilometre, which matches the textbook atmospheric environmental lapse rate. That's a nice independent physical sanity check.
>
> In **autumn** there was a heat wave in mid-September that added variance uniformly across elevations, so the elevation trend has nothing new to explain — RK and OK are essentially tied at 4.68 °C.
>
> [middle plot, hexbin] Here's the pooled scatter for regression kriging in summer. Across all 29 stations and 168 hours, predictions sit tight on the 1:1 line with R² = 0.83. The right panel is the error distribution — centred near zero with standard deviation 2.6 °C.
>
> [right plot, summary] This final bar chart is every experiment in the project at a glance. On the left, the synthetic gold standard — 0.15 °C RMSE with 44 dense sensors in a 1 km box. In the middle, real-data regression kriging at about 3 °C. On the right the HPWREN single-sensor benchmark, which is large and noisy because we had to **infer** that sensor's coordinates from its pressure signature."

---

## Conclusion slide

**Put on the slide:**

> ### Take-aways
>
> - Spatio-temporal kriging from public weather stations works: **~2 °C RMSE** at an inland held-out site
> - Elevation detrending cuts pooled RMSE by ~33 % in summer
> - Recovered lapse rate matches atmospheric physics (−5 to −7 °C / km)
> - But regional kriging **cannot replace** an on-site sensor when the site has its own microclimate
> - Code + notebook + data:
>   **github.com/mkizilov/stat654-kriging-project**

**Say (≈45 s):**

> "To wrap up. Spatio-temporal kriging with the sum-metric variogram is a viable, interpretable, reproducible method for predicting temperature at a site without a sensor. With 29 free public weather stations around San Diego, we get RMSE of about 2 °C at a held-out inland site, and **3 °C pooled across all 29 stations** when we include the elevation trend.
>
> Elevation matters a lot — plain OK gives 4.6 °C RMSE, regression kriging gives 3.1, and the fitted elevation slope reproduces the textbook lapse rate from data alone.
>
> The practical message from the paper holds on a different region and dataset: public weather stations are a reasonable substitute for on-site sensors when you don't have them, but only to the accuracy of a few degrees — for **site-specific microclimate** you still want a local sensor.
>
> The full code, tests, and notebook are on GitHub at the repo shown. Thanks — happy to take questions."

---

## Q&A preparation (not on slides — just in case)

Likely questions and one-liner answers:

1. **Why 29 stations and not 44 like the paper?**
   The paper used 24 weather stations + 44 local greenhouse sensors. We do not have a local sensor array in San Diego, so we use 29 NOAA stations and imitate the "local vs remote" comparison with a near-10 vs far-10 subset experiment in the main notebook.

2. **Why did you pick 33.1 °N, −116.6 °W as the target?**
   It is the inferred location of the HPWREN Minute Weather sensor from the Kaggle dataset the class was originally pointed to — chosen to match its ~860 m elevation from the pressure signature of 917 mbar.

3. **Why did ordinary kriging do so poorly in autumn 2020?**
   A heat wave dominated that week; variance was spatially uniform, so the elevation trend that helps in summer carries no additional information.

4. **What's the anisotropy scale k?**
   It is a fitted parameter in the sum-metric variogram that tells you how many kilometres of spatial distance equal one hour of temporal distance — essentially, at what scale do space and time become comparably dissimilar.

5. **How do you know the regression kriging LOO is not leaking?**
   Every fold refits the elevation trend, the per-hour mean, and the sum-metric variogram from scratch using only the n−1 training stations. The held-out station never contributes to any of those quantities.

6. **Why did the HPWREN comparison have negative R²?**
   Two reasons: (1) that sensor has a distinct microclimate with a very small diurnal swing, and (2) its coordinates are inferred from the pressure, not given in the dataset — a location error of several km on heterogeneous terrain shows up as a large temperature error.
