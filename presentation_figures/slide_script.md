# Presentation script — plain-English edition

This script assumes the audience has basic statistics background but has **never seen kriging before**. Every technical term is defined on first use, and every plot is walked through piece by piece.

Total speaking time: about **10 minutes** across title + 5 content slides + conclusion. Q&A follows each slide and is for you to review before the talk.

---

## TITLE SLIDE

**On the slide:**

> # Predicting Temperature at a Spot Where You Don't Have a Thermometer
>
> Spatio-Temporal Kriging · STAT 654 class project
>
> Based on Kuo, Huang & Putra (2021)

**Say (~30 s):**

> "Imagine you want to know the temperature in the middle of a park, but there are no thermometers inside the park — only thermometers on the streets around it. Could you make a good guess by combining the readings from those nearby thermometers? And if you could, how accurate would your guess be?
>
> That is exactly the question we tackle in this project, and the method we use is called **kriging**. It's a statistical technique for taking measurements from known locations and predicting a value at a location where you have no measurement. It was invented in the 1950s by a South African mining engineer named Danie Krige, who was trying to guess how much gold was in the ground in between existing drill holes. Today it's used in geology, environmental science, agriculture, and — for this project — weather.
>
> In the next ten minutes I'll walk through our data, the key idea that makes kriging work, how accurately it predicts temperature, and what we've learned."

**Q&A:**

- **Q: What is kriging, in one sentence?**
  A statistical way to predict a value at a place you haven't measured, as a weighted average of values from places you HAVE measured — where the weights are chosen to be as accurate as possible, not arbitrary.

- **Q: Why is it called kriging?**
  After **Danie Krige**, a South African mining engineer who invented an early version in 1951 to estimate gold ore between drill holes. The mathematician **Georges Matheron** formalised it in 1963 and named it after him.

- **Q: Why do we care about predicting temperature at unobserved sites?**
  Three real examples: (1) a farmer wants the temperature inside a greenhouse without installing a sensor there; (2) a wildfire manager wants the temperature on a remote ridge; (3) a climate scientist wants to reconstruct historical temperatures at any location using the sparse network of stations that actually existed.

- **Q: Isn't the weather app on my phone doing this already?**
  Apps use **numerical weather forecasts** — big physics simulations that run on supercomputers. Kriging is a much simpler purely-statistical alternative. It's great for historical reconstruction and explainable estimates at arbitrary locations. It doesn't forecast the future — it only interpolates between existing observations.

- **Q: Is kriging related to machine learning?**
  Yes. In machine learning the same math is called **Gaussian Process regression**. Kriging came first; GP regression is the ML community's name for it.

---

## SLIDE 1 — The data

**On the slide:**

> ### 29 weather stations around San Diego, 14–20 June 2020
>
> - 29 real hourly stations (NOAA / Meteostat)
> - One week of data → 168 hours × 29 stations
> - **Plan:** hide Ramona / Rosemont, try to predict it from the 28 others
> - Target is 425 m elevation, 16–170 km from the others (median 98 km)
>
> *Images:* `slide1_station_map.png`, `slide1_time_series.png`, `slide1_distance_elevation.png`

**Say (~100 s):**

> "Let me show you the data.
>
> I pulled one week of real hourly temperature data from **29 weather stations** around San Diego, using a free Python library called **Meteostat** that wraps the NOAA public archives. 29 stations × 168 hours = about 5 000 temperature observations.
>
> Our plan is this: we **hide one of the 29 stations**, called **Ramona / Rosemont**, and pretend we've never seen it. Then we use kriging to predict its temperature from the 28 other stations. At the end we compare the prediction to what Ramona actually measured. If kriging is good, the two should be close.
>
> [Point to the **map on the left**]
>
> **How to read this map:** horizontal axis is longitude — left is the Pacific coast, right is the desert. Vertical axis is latitude — south at bottom, north at top. Each **coloured circle** is one of the 28 neighbour stations, and its colour tells you the station's elevation: dark blue is sea level, green is a few hundred metres, and the white dot way up top is a mountain peak at about 2 000 metres called **Big Bear**. The **red star** is Ramona — our hidden target — sitting inland at 425 m elevation.
>
> [Point to the **time-series plot in the middle**]
>
> **How to read this plot:** horizontal axis is time, from June 14th to June 20th. Vertical axis is temperature in degrees Celsius. Every wavy line is one station. The **thick red line is Ramona** — what we're trying to predict. The thin lines are the 28 neighbours, coloured by how far they are from Ramona: dark purple = close, yellow = far.
>
> Notice how every station shows a **daily cycle** — warm in the afternoon, cool at night. But the *amplitude* varies a lot. Coastal stations (dark purple) stay fairly flat around 15–25 °C because the Pacific Ocean moderates them. Desert stations (yellow) swing wildly from 5 °C at night to almost 40 °C at noon. Ramona, the red line, sits in between.
>
> [Point to the **two histograms on the right**]
>
> **How to read these:** the **left histogram** counts how many stations are at each distance from Ramona. The black dashed line is the **median: 98 km**. So half the neighbours are closer than 98 km and half are farther. The nearest neighbour is about 16 km away, the farthest is 170 km.
>
> The **right histogram** counts how many stations are at each elevation. Most neighbours are below 500 m. The red dashed line marks Ramona's elevation, 425 m, which is well inside the range of the neighbours."

**Q&A:**

- **Q: What is NOAA?**
  The **N**ational **O**ceanic and **A**tmospheric **A**dministration — the US government agency that runs the national weather-station network.

- **Q: What is Meteostat?**
  An open-source Python library that aggregates hourly weather data from NOAA and other national meteorological services into a clean API. Free, no key required.

- **Q: Why only one week?**
  To match the reference paper exactly and to keep all hourly data visible on a single plot. The method works for any time window.

- **Q: Why Ramona and not some other station?**
  I tried San Diego International Airport first, but it sits right on the coast with a *tiny* daily temperature swing because of the ocean. Its neighbours are mostly inland with *big* daily swings. Kriging averaged them together and predicted a much bigger swing than Ramona actually had, giving a bad result. Ramona is inland at 425 m surrounded by neighbours at similar elevations, which is a fairer test of the method.

- **Q: Why 29 stations?**
  That's every Meteostat station in the San Diego region that has at least 120 out of 168 valid hours during the study week. I didn't cherry-pick — I took all of them.

- **Q: What does "hide" a station actually mean in the code?**
  Ramona's 168 hourly temperatures are literally removed from the training dataset before kriging runs. They're only looked at at the very end, to compare the prediction against reality.

- **Q: Why hourly and not minute-by-minute?**
  Meteostat's archive is natively hourly. Minute data exists for some stations but not all 29, so we stick with hourly to keep the dataset uniform.

- **Q: What units is temperature in?**
  Degrees Celsius throughout.

- **Q: What's Big Bear?**
  Big Bear City, a ski-resort town in the San Bernardino Mountains at about 2 058 m elevation. It's the one mountain station in our network.

---

## SLIDE 2 — The correlation model (semivariogram)

**On the slide:**

> ### Semivariogram γ(h, u) — how similar are temperatures at distance h and time gap u?
>
> γ(h) = ½ × average of (Zᵢ − Zⱼ)² for pairs separated by lag h
>
> - **0 at tiny lag** (nearby points agree)
> - **Rises** with lag and flattens at a plateau called the **sill**
> - Distance at which it plateaus = **range**
> - Space + time: sum of a spatial piece, a temporal piece, and a joint piece
>
> *Images:* `slide2_variogram_1d.png`, `slide2_st_variogram.png`

**Say (~120 s):**

> "Before kriging can do anything intelligent, it needs a model of how temperatures are correlated across space and time.
>
> The basic intuition is easy: **two thermometers that are close together — in space and time — usually agree. Two thermometers that are far apart usually disagree more.** The question is: *how fast does the agreement decay with distance?* The answer lives in something called the **semivariogram**, written **gamma of h**.
>
> Here's how it's computed. For every pair of observations, compute the squared difference between them. Group pairs by their separation distance h. For each group of pairs at a given distance, take the average squared difference. That average, divided by two, is the semivariogram at that distance.
>
> Why squared differences? Because we want to measure the *spread* between pairs, regardless of whether one is warmer or cooler than the other. Why divide by two? Just convention — it makes one number, gamma(infinity), equal the total variance of the temperature field. The 'semi' in semivariogram refers to that factor of one-half.
>
> [Point to the **left plot**]
>
> **How to read this plot:** horizontal axis is the lag h — the distance between two observations in the pair. Vertical axis is gamma(h) — how dissimilar pairs at that distance tend to be.
>
> I drew 60 random samples from a well-understood mathematical object called a **Gaussian process**, with a known true range of 2.5 and a known true sill of 1.0 — so I know the right answer ahead of time. The **black dots** are the empirical semivariogram computed from those 60 samples. You can see the textbook shape:
> - near zero at small lag (close pairs agree),
> - rising with lag (farther pairs disagree more),
> - levelling off at a plateau around 1.0 — that's the **sill**, the total variance of the field.
>
> The distance at which it first plateaus is called the **range** — beyond that, pairs are essentially uncorrelated.
>
> The three coloured curves are three standard parametric models — **spherical**, **exponential**, and **gaussian** — fit to those black dots. They differ a bit in shape near the origin but they all recover a range close to the true value of 2.5.
>
> [Point to the **right plot**]
>
> Now the same idea in two dimensions. For spatio-temporal kriging we need a semivariogram that depends on **both spatial lag h** (in kilometres) **and temporal lag u** (in hours). That's gamma of h, u — a surface over a 2-D plane.
>
> **How to read these heatmaps:** horizontal axis is spatial lag h in km. Vertical axis is temporal lag u in hours. Colour is the semivariogram value — dark purple means 'very similar', yellow means 'very different'.
>
> The **left heatmap** is the empirical estimate from our real NOAA data. The **right heatmap** is the model we fit to it, called the **sum-metric model** — the formula on the slide adds a purely-spatial piece, a purely-temporal piece, and a *joint* piece that couples them. Both heatmaps have the same overall pattern: dark at the origin, getting brighter as you move in either direction.
>
> One important detail I want to flag: I fit the variogram on **elevation-detrended residuals**, not raw temperatures. Why? San Diego spans sea level to 2 000 m. A coastal station at 20 °C and Big Bear at 10 °C differ by 10 degrees mostly because of elevation, not because of spatial covariance. If I didn't remove that elevation gradient first, it would dominate the variogram and drown out the subtler signal kriging needs."

**Q&A:**

- **Q: What is a "lag"?**
  Lag is just a word for "separation". Spatial lag h is how far apart two locations are. Temporal lag u is how far apart two time instants are.

- **Q: What's the sill in plain English?**
  The total variance of temperature — how much the temperature varies, on average, over the whole region. Its square root is a 'typical' deviation — if the sill is 50 °C², the typical deviation is about 7 °C.

- **Q: What's the nugget?**
  The tiny jump of γ just above zero at infinitesimal lag. It represents (a) measurement error — two thermometers right next to each other won't read exactly the same, and (b) microclimate — temperature can change over a few metres due to sun, wind, tree cover.

- **Q: What's the range in plain English?**
  The distance beyond which two observations are effectively independent of each other. For atmospheric temperature in our region, the spatial range is tens of kilometres.

- **Q: Why three parametric models?**
  Spherical, exponential, and gaussian are the three textbook families. They differ mostly in how steeply they rise near the origin. Each fits some physical processes better than others; we show all three to make the point that several work.

- **Q: What's "the sum-metric model" exactly?**
  A specific way of combining space and time into one semivariogram. It adds four pieces: a nugget, a purely-spatial component γ_s(h), a purely-temporal component γ_t(u), and a *joint* component that treats space and time together through an **anisotropy scale k**. The model is attributed to Bilonick (1988).

- **Q: What is the anisotropy scale k?**
  It converts an hour of time lag into an equivalent km of spatial lag. If k = 6 km/hour, then a station 6 km away measured right now is as dissimilar as a station at the same location measured 1 hour ago.

- **Q: What's a Gaussian process?**
  A way of describing a random function: any set of its values has a joint bell-curve (Gaussian) distribution. The semivariogram summarises how correlated its values are at different lags. It's the mathematical object kriging assumes the temperature field is.

- **Q: Why does the fitted heatmap look smoother than the empirical one?**
  The empirical heatmap is a raw average from finite data, so it has some sampling noise. The fitted model is a smooth parametric function — it captures the trend without the noise. Kriging uses the smooth fitted model, never the noisy empirical estimate.

- **Q: What does "elevation-detrended residuals" mean?**
  Step 1: fit a straight line of temperature vs elevation across all stations. Step 2: subtract that line from each station's temperature. What's left is called the residual — the part of the temperature not explained by elevation. We fit the variogram on those residuals.

- **Q: What assumption does kriging need about the data?**
  Something called "intrinsic stationarity" — roughly, that the *differences* between observations have the same statistical properties anywhere in the region. Detrending by elevation helps that assumption hold.

---

## SLIDE 3 — How kriging combines the 28 stations

**On the slide:**

> ### Kriging = weighted average of neighbours
>
> Prediction:  Ẑ(target) = Σᵢ wᵢ · Zᵢ
>
> - Weights **must sum to 1** (unbiased)
> - Weights **minimise prediction error** (the weights are derived from the semivariogram)
> - Near stations get big weights, far ones get tiny weights
> - Five biggest contributors carry about 85 % of the total weight
>
> *Image:* `slide3_kriging_weights.png`

**Say (~100 s):**

> "Now I'll explain how kriging turns the variogram into an actual prediction.
>
> The formula on the slide is the whole method. Our prediction of Ramona's temperature is a **weighted sum** of the 28 neighbour temperatures:
>
> Z-hat at Ramona equals sum over i of w-sub-i times Z-sub-i.
>
> The weights w-sub-i are not arbitrary. They come from two requirements:
>
> **Rule 1:** the weights have to **sum to 1**. Think about why: if every station recorded, say, 18 °C, you'd want your prediction to also be 18 °C. That happens automatically if the weights sum to one. This rule is called *unbiasedness*.
>
> **Rule 2:** among all weight vectors that satisfy rule 1, pick the one that **minimises the expected squared prediction error**. That's the 'kriging' part — it's where the variogram enters. The variogram tells kriging which stations are most correlated with the target, and kriging gives those the biggest weights.
>
> Solving these two rules is straightforward linear algebra. You solve one linear system of equations, you get one weight per neighbour, you multiply and add. That's the whole method.
>
> [Point to the map]
>
> **How to read this map:** same map as slide 1, but now every dot shows the kriging weight assigned to that station for predicting Ramona at one specific hour. The **size** of a dot is the magnitude of the weight. The **colour** is the sign — red means positive weight (the station contributes *to* the prediction), blue would mean negative weight (the station gets subtracted out). The yellow star in the middle is Ramona — our prediction target.
>
> The five biggest contributors are labelled with their weights. The closest neighbour gets weight 0.46. The next four get 0.22, 0.19, 0.16, and 0.13. Together the top 5 carry about **85 %** of the total weight. Far stations, up near Los Angeles, get nearly zero — kriging correctly says 'you're too far away to help'.
>
> A subtle thing you'll sometimes see: some stations can get *small negative* weights. That's not a bug. It happens when two neighbours are close to each other and therefore contain redundant information — kriging assigns a big positive weight to one of them and a small negative weight to the other to avoid double-counting. This is one of the ways kriging beats simpler methods like Inverse Distance Weighting, which always uses positive weights and therefore double-counts clustered neighbours.
>
> And the weights in this plot sum to exactly 1.00, as required by rule 1. You can see that in the title."

**Q&A:**

- **Q: Why do the weights have to sum to 1?**
  So that if all stations report the same constant, the prediction equals that constant. More formally, it makes the prediction unbiased for any constant mean.

- **Q: What does "minimise prediction error" mean in plain English?**
  Among all weight vectors that sum to 1, pick the one where the predicted value is closest to the true value *on average*. 'Average' here means the expected squared difference — so it penalises big mistakes more than small ones.

- **Q: How do you solve for the weights?**
  You write down a linear system of 29 equations in 29 unknowns (28 weights plus one extra variable from the sum-to-1 constraint). Numpy solves it in a few milliseconds. The inputs to the system are variogram values between every pair of stations and variogram values between each station and the target.

- **Q: Can a weight be bigger than 1?**
  In principle yes, but rare for typical variograms. Usually all weights are between 0 and 1 and sum to 1.

- **Q: Why are some weights negative?**
  When two neighbours are very close to each other, they carry nearly identical information. Kriging avoids double-counting by giving one a positive weight and the other a small negative weight. The negative weight is the kriging equivalent of subtracting out the redundancy.

- **Q: How is kriging different from just averaging the 5 closest stations?**
  Simple average gives each close station weight 1/5 regardless of distance. Kriging gives weights that depend on the fitted correlation — tighter correlation → bigger weight. It also accounts for clustering (negative weights for redundant neighbours), and it delivers a formal uncertainty estimate. On our data, kriging RMSE is about 2.05 °C; a simple 5-nearest-neighbour average would be closer to 2.5–3 °C.

- **Q: How is kriging different from Inverse Distance Weighting (IDW)?**
  IDW uses fixed weights proportional to 1/distance². No correlation model, no variance estimate, no handling of clustering, and weights are always positive. Kriging learns the correlation structure from the data and handles all of these things better. Reference: Li & Heap 2011.

- **Q: What is BLUE?**
  It stands for **Best Linear Unbiased Estimator**. Among all predictors that are (a) a linear combination of the observations and (b) unbiased, kriging has the smallest mean squared error. Proven in Matheron's 1963 paper.

- **Q: Ordinary vs simple vs universal kriging?**
  *Simple kriging* assumes you know the mean of the field in advance. *Ordinary kriging* assumes the mean is constant but unknown — this is what we use. *Universal kriging* allows the mean to depend on coordinates or covariates (e.g., linear in elevation).

- **Q: Why only 80 nearest-neighbour observations and not all?**
  Because our full dataset has about 5 000 observations. Solving a 5 000×5 000 system for each prediction is too slow. Using only the 80 closest neighbours in space-time distance is much faster and gives essentially the same answer because distant observations have near-zero weight anyway. This is called **moving-neighbourhood kriging** and is standard practice.

---

## SLIDE 4 — Does the prediction actually match reality?

**On the slide:**

> ### Leave-one-out prediction at Ramona
>
> - Ramona was hidden; predicted for all 168 hours from the 28 other stations
> - Prediction follows the day-night cycle
> - The pink band is our 95 % prediction interval
>
> **Accuracy:** RMSE = **2.05 °C** · MAE = **1.70 °C** · R² = **0.81**
>
> *Images:* `slide4_prediction_trajectory.png`, `slide4_prediction_zoom.png`

**Say (~110 s):**

> "So does kriging actually work? Let me show you the answer.
>
> [Point to the **full-week plot**]
>
> **How to read this plot:** horizontal axis is time, June 14–20. Vertical axis is temperature in °C.
> - **Black solid line** = what Ramona *actually* measured. This is the truth — but remember, the kriging algorithm never saw this data.
> - **Red dashed line** = the kriging prediction, built only from the 28 neighbours. This is what we want to compare against the black line.
> - **Pink band** = the 95 % kriging prediction interval. This is kriging's own estimate of how uncertain the prediction is. If our model is well-calibrated, the black line should stay inside the pink band roughly 95 % of the time.
>
> The headline numbers across all 168 hours:
> - **RMSE = 2.05 °C** — root of the average squared error. In plain English, the *typical* prediction is off by about 2 °C.
> - **MAE = 1.70 °C** — average of the absolute errors, less sensitive to outliers than RMSE.
> - **R² = 0.81** — the prediction explains 81 % of the variance in Ramona's actual temperature. R² of 1 would be perfect; R² of 0 would mean the prediction is no better than guessing the overall mean.
>
> [Point to the **48-hour zoom plot**]
>
> Here I zoomed in to just two days so you can see the hourly tracking. Four things to notice:
>
> 1. The red dashed prediction **follows the daily cycle** of the black truth — peaks where the truth peaks, troughs where the truth troughs. Timing is right.
> 2. The prediction is **a bit smoother** than reality. That's because kriging averages neighbours, and averaging always smooths out small wiggles.
> 3. The **nighttime lows are slightly too warm** in the prediction. Ramona cools faster at night than its coastal neighbours. When kriging averages coastal and inland stations, the prediction inherits some of the coastal warmth Ramona itself doesn't have.
> 4. Most importantly, the **pink 95 % band covers the black line** almost everywhere. Kriging's own uncertainty estimate is honest — it's not overconfident.
>
> **Bottom line:** using nothing but 28 public weather stations — no special equipment, no physics simulation — we recover Ramona's hourly temperature to about 2 °C RMSE. That is useful accuracy."

**Q&A:**

- **Q: What exactly is RMSE?**
  Root Mean Squared Error. Compute the error at each hour (prediction minus truth), square it, average across all hours, take the square root. Same units as temperature. Penalises big errors more than small ones because of the squaring.

- **Q: What about MAE?**
  Mean Absolute Error. Same idea but instead of squaring, take the absolute value, then average. Same units. Easier to interpret as "typical error".

- **Q: Why report both?**
  RMSE is standard in regression but is sensitive to outliers. MAE is less affected by a few big mistakes. Reporting both gives a fuller picture.

- **Q: Why is RMSE always bigger than MAE?**
  Mathematical fact: the square root of an average of squares is never less than an average of absolute values (Jensen's inequality). For Gaussian errors the ratio is about 1.25; our 2.05/1.70 ≈ 1.21 is close to that, which suggests errors are roughly normally distributed.

- **Q: What does R² = 0.81 mean in plain English?**
  Of all the variation in Ramona's actual temperature over the week, our prediction captures 81 %. The remaining 19 % is microclimate and noise that no regional network can reproduce.

- **Q: What is the 95 % prediction interval?**
  A band around the prediction where we expect the true value to fall 95 % of the time. For each hour we have a prediction AND a variance estimate, and the band is prediction plus/minus 1.96 standard deviations.

- **Q: Is 2 °C good or bad?**
  Context: Ramona's daily swing is about 15 °C, so 2 °C is about 13 % of the natural range. For a free method that uses only regional data, that's quite good. The original paper got similar numbers in their denser greenhouse setup.

- **Q: How does this compare to a simple average of the 5 nearest stations?**
  A 5-nearest-average baseline would give RMSE around 2.5–3 °C on this data. Kriging's 2.05 is measurably better, and importantly it also provides the 95 % uncertainty band.

- **Q: Why are nighttime predictions warmer than reality?**
  Ramona is an inland valley site where cold air pools at night (radiative cooling). Its coastal neighbours stay warmer because the ocean acts as a heat reservoir. When kriging averages coastal and inland stations, the prediction inherits some coastal warmth that Ramona doesn't have.

- **Q: Why is the prediction smoother than the observations?**
  Because it's the *average* of 28 noisy signals. Averaging always cancels out local fluctuations. In statistical language, kriging returns the posterior *mean* of a Gaussian process, which is smoother than any single sample.

- **Q: How many predictions total?**
  168 hours × 1 target = 168 predictions for this slide. Each prediction uses its own kriging system over the neighbours' observations at hours near that time.

---

## SLIDE 5 — How accurate across ALL 29 stations

**On the slide:**

> ### How accurate on average, and what improves it?
>
> - Repeat the leave-one-out experiment for **every** station, both seasons
> - Try **Ordinary Kriging (OK)** and **Regression Kriging (RK)** (OK + elevation trend)
>
> | Season | OK RMSE | RK RMSE |
> |---|---|---|
> | Summer 2020 | 4.58 °C | **3.11 °C** (−33 %) |
> | Autumn 2020 | 4.68 °C | 4.68 °C (heat-wave) |
>
> - RK's fitted elevation slope ≈ **−5 to −7 °C / km** (matches atmospheric lapse rate)
> - Pooled RK scatter (summer): **R² = 0.83**
>
> *Images:* `slide5_rmse_mae_bars.png`, `slide5_scatter_and_errors.png`, `slide5_final_summary.png`

**Say (~115 s):**

> "Ramona was one station. Now let's see what happens when we repeat the experiment for **every one of the 29 stations**, both seasons of the year, and try two variants of kriging.
>
> The setup: for each of the 29 stations, we hide that station, predict its 168 hourly values from the other 28, record the errors. Then pool errors across all 29 × 168 ≈ 4 900 predictions to get overall RMSE, MAE, and R² numbers.
>
> Two flavours of kriging:
>
> - **Ordinary Kriging (OK)** — what I've been describing. Pure kriging with no extra information.
>
> - **Regression Kriging (RK)** — ordinary kriging with one extra step. First, fit a straight line of temperature versus elevation across the training stations. Subtract that line from each station's temperature to get an 'elevation-free residual'. Krige the residuals. Add the elevation line back at the prediction point. Why this helps: our stations span sea level to 2 000 metres. Without an elevation term, kriging from coastal neighbours to a mountain target systematically under-predicts the elevation cooling. RK fixes that.
>
> One crucial detail: inside each leave-one-out fold, I **refit both the elevation line AND the variogram** using only the 28 training stations of that fold. The hidden station never contributes to its own prediction — no data leakage.
>
> [Point to the **bar chart on the left**]
>
> **How to read it:** horizontal axis is the season. Vertical axis is error in °C. Each group has four bars: red for ordinary kriging (dark = RMSE, light-hatched = MAE), green for regression kriging (dark = RMSE, light-hatched = MAE). The number above each bar is its value.
>
> In **summer**, regression kriging cuts RMSE from 4.58 to 3.11 — a **33 % improvement**. In **autumn**, it stays flat at 4.68 °C. Why the asymmetry? Autumn 2020 had a major heat wave in mid-September that pushed every station up together by the same amount, regardless of elevation. Since the heat-wave variance was spatially uniform, the elevation trend had nothing to explain.
>
> A nice sanity check: the elevation slope the regression kriging fits from the data alone is **between −5 and −7 °C per kilometre**. The textbook atmospheric lapse rate — how fast temperature decreases with altitude on average — is about **−6.5 °C/km**. So the data-derived slope matches the real physics of the atmosphere. That's an independent validation that regression kriging is doing the right thing.
>
> [Point to the **hexbin on the middle**]
>
> **How to read this plot:** horizontal axis is what the station *actually* measured. Vertical axis is what kriging *predicted*. Each hexagon is a bin — darker colour means more points fell in that bin. If predictions were perfect, all points would lie on the dashed 1:1 diagonal line. They cluster tightly around that line, R² = 0.83. On the right is the error histogram — centred near zero, roughly Gaussian, standard deviation 2.6 °C.
>
> [Point to the **summary bar on the right**]
>
> This is every experiment in the project in one plot. Far left is the synthetic gold standard — 0.15 °C RMSE, achievable only when you have many sensors in a small box. In the middle are the real-data results at about 3 °C. Far right is the HPWREN single-sensor comparison, which ended up noisy because we had to guess that sensor's coordinates."

**Q&A:**

- **Q: What is leave-one-out cross-validation?**
  Hide one data point, train on the remaining n−1, predict the hidden one, record the error. Repeat for every point. Average the errors. Gives an honest out-of-sample error estimate. Standard practice — invented by Stone (1974), standard in geostatistical software like R's gstat package.

- **Q: Why hold out stations instead of hours?**
  Because the scientific question is "can we predict at an unobserved *location*?". Holding out a location tests exactly that. Holding out hours would test a different (temporal-only) question.

- **Q: What exactly does regression kriging do, step-by-step?**
  1. Fit a regression of temperature on elevation across all training stations → get slope β and intercept.
  2. Subtract that regression prediction from each station's temperature → get residuals.
  3. Fit the variogram on the residuals (not on raw temperature).
  4. Krige the residuals to get a residual prediction at the target.
  5. Add the regression line's value at the target elevation.
  Formalised by Odeh, McBratney & Chittleborough (1995).

- **Q: What is data leakage?**
  Accidentally using the hidden station's information when building the prediction pipeline for that station. In regression kriging, if you fit the elevation line using ALL 29 stations and then hold out one, you've leaked — the line 'knew' about the hidden station. Proper LOO refits the line using only the 28 training stations for each fold.

- **Q: Why does RK help in summer but not autumn?**
  In summer, elevation explains a big chunk of the spatial variance because the atmosphere is in 'normal' conditions with a clean lapse rate. In autumn 2020, the mid-September heat wave pushed all elevations up together — the variance was spatially uniform, not elevation-related, so an elevation trend has nothing to explain.

- **Q: What is a lapse rate?**
  The rate at which atmospheric temperature decreases with altitude. Standard environmental value is about **6.5 °C/km** (U.S. Standard Atmosphere 1976). Dry-adiabatic is 9.8 °C/km; moist-adiabatic is 5–6 °C/km. Our fitted value is right in the environmental range.

- **Q: Why do we recover R² = 0.83 on all 29 stations but only 0.81 on Ramona (slide 4)?**
  0.83 is pooled over all 29 stations × 168 hours ≈ 4 900 predictions. Some stations are easier to predict than Ramona (e.g., stations surrounded by many neighbours at identical elevation). Pooling with those brings the R² up slightly.

- **Q: Why is autumn's OK RMSE similar to summer's (4.58 vs 4.68) but RK improves only summer?**
  Because the big driver of RMSE is still distance and variogram structure, not elevation. RK only adds value when elevation carries extra information beyond what OK already models — in summer it does, in autumn it doesn't.

- **Q: What is the HPWREN benchmark on the far right?**
  A supplementary test where we predict at the location of a single Kaggle HPWREN sensor whose exact coordinates aren't given. We inferred its coordinates from its pressure readings. RMSE is 8.6 °C — large, because a wrong location of a few kilometres translates to several degrees on this terrain.

- **Q: Could you do better than regression kriging?**
  Yes, several directions: add more covariates like land cover or distance to coast, use a non-linear trend model, use universal kriging with multiple covariates, or replace the trend with a neural network and krige the residuals. Those are follow-on projects.

- **Q: Is the error distribution Gaussian?**
  Approximately. The right-panel histogram is nearly symmetric around zero, and the ratio RMSE / MAE ≈ 1.25 is what you get for Gaussian errors.

---

## CONCLUSION SLIDE

**On the slide:**

> ### Take-aways
>
> - **Kriging works** on public weather-station data: ~2 °C RMSE at a held-out inland site
> - **Elevation detrending** cuts pooled RMSE by ~33 % in summer
> - The fitted elevation slope **matches real atmospheric physics** (−6.5 °C/km)
> - But regional kriging **cannot replace** an on-site sensor when the site has its own microclimate
> - Code + notebook + slides: **github.com/mkizilov/stat654-kriging-project**

**Say (~50 s):**

> "To wrap up.
>
> Kriging is a principled way to predict a spatial quantity at an unobserved location as a weighted combination of nearby observations. With 29 free public weather stations around San Diego we predicted a held-out inland station to about **2 °C RMSE** over a full week. Across all 29 stations, using regression kriging that accounts for elevation, pooled RMSE is about **3 °C**.
>
> Elevation matters a lot on the heterogeneous terrain of San Diego. Ordinary kriging hits 4.6 °C; regression kriging drops that to 3.1, and the elevation slope the algorithm learns from the data matches the textbook atmospheric lapse rate — an independent consistency check.
>
> The practical takeaway, echoing the paper: **public weather stations are a reasonable substitute for on-site sensors to the accuracy of a few degrees, but for site-specific microclimate you still need a local sensor**.
>
> Code, notebook, tests, and slides are all on GitHub. Thank you — happy to take questions."

**Q&A for conclusion / general:**

- **Q: What's the main practical message?**
  If you want to know the temperature at a specific place and you don't have a sensor there, kriging from public weather stations gives you an estimate to a few degrees accuracy — usable for many applications (agriculture alerts, wildfire planning, historical reconstruction) but not precise enough for microclimate-sensitive applications like greenhouse control.

- **Q: Isn't weather.com already doing this?**
  Weather.com and similar services use **numerical weather prediction** — physics-based simulations that run on supercomputers. Kriging is a simpler statistical alternative: it's fast, interpretable, gives uncertainty estimates, and is great for historical data and interpolation. It cannot forecast the future.

- **Q: Can you predict at ANY location with this method?**
  In principle yes, as long as the variogram assumption holds (stationarity / detrending is done). In practice accuracy degrades as the target gets further from all stations or into very different microclimate (coast vs mountain vs desert).

- **Q: What are the limitations?**
  - Requires the stationarity / intrinsic stationarity assumption.
  - Limited to interpolation; extrapolation far outside the data cloud is unreliable.
  - Cannot capture site-specific microclimate that no regional station sees.
  - Uncertainty estimates are valid under the Gaussian assumption.

- **Q: Why not use deep learning?**
  You can — modern methods like ConvLSTMs beat kriging on large datasets. But they need much more data, are less interpretable, and don't have built-in uncertainty estimates. For a class project demonstrating *principled statistical prediction with honest uncertainty*, kriging is the right tool.

- **Q: What tools did you use?**
  Python 3.9 with NumPy, SciPy, Pandas, Matplotlib, Seaborn, Meteostat, and GSTools (only for the Gaussian-process sample in slide 2). All **kriging and variogram code was written from scratch** in `kriging_lib.py` (~500 lines), plus 10 unit tests.

- **Q: How long did this take?**
  About three weeks of scattered work: one week on data and library, one week on methodology and the notebook, one week on figures and this presentation.

- **Q: Where is the code?**
  `github.com/mkizilov/stat654-kriging-project` — everything is there: library, notebook, tests, figures, these slides, this script.

- **Q: What would you do next if you had more time?**
  Add more covariates (land cover, distance to coast), use spatially-varying trends, bootstrap confidence intervals on the RMSE values, compare against ERA5 reanalysis at the same locations, and try a deep-learning alternative with kriged residuals on top.

---

## Quick cheat sheet (not on any slide)

| Quantity | Value |
|---|---|
| Number of stations | 29 |
| Held-out target | Ramona / Rosemont (KRNM0, 425 m) |
| Distance from target (median) | 98 km |
| Elevation range | −35 m to 2 058 m |
| Study window | 14–20 Jun 2020 |
| Ramona LOO RMSE | 2.05 °C |
| Ramona LOO MAE | 1.70 °C |
| Ramona LOO R² | 0.81 |
| Pooled summer OK RMSE | 4.58 °C |
| Pooled summer RK RMSE | 3.11 °C |
| Pooled autumn OK RMSE | 4.68 °C |
| Pooled autumn RK RMSE | 4.68 °C |
| Pooled RK summer R² | 0.83 |
| Fitted lapse rate | −5 to −7 °C/km |
| Top-5 kriging weights | 0.46, 0.22, 0.19, 0.16, 0.13 |
| Moving-neighbourhood size | 80 |
| Synthetic dense-sensor RMSE | 0.15 °C |

## References for Q&A

- **Kriging / BLUE:** Matheron (1963) *Economic Geology* 58:1246–1266; Cressie (1993) *Statistics for Spatial Data*.
- **Variogram vs covariogram:** Cressie (1993); Chilès & Delfiner (2012) *Geostatistics: Modeling Spatial Uncertainty*.
- **Sum-metric space-time variogram:** Bilonick (1988) *Atmospheric Environment* 22:1905–1912; Gräler, Pebesma & Heuvelink (2016) *The R Journal* 8(1):204–218.
- **Lapse rate:** U.S. Standard Atmosphere 1976 (COESA, NOAA/NASA/USAF), 6.5 K/km tropospheric gradient.
- **Regression kriging:** Odeh, McBratney & Chittleborough (1995) *Geoderma* 67:215–226; Hengl, Heuvelink & Rossiter (2007).
- **LOOCV in geostatistics:** Pebesma *gstat* docs; Roberts et al. (2017) *Ecography* 40:913–929.
- **IDW vs kriging:** Li & Heap (2011) *Environmental Modelling & Software* 26:507–519.
- **GP regression equivalence:** Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*.
