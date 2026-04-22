# Spatio-Temporal Kriging for Local Temperature Prediction

STAT 654 class project reimplementing and extending the spatio-temporal kriging analysis of

> **Kuo, P.-F.; Huang, T.-E.; Putra, I.G.B.** (2021). *Comparing Kriging Estimators Using Weather Station Data and Local Greenhouse Sensors.* **Sensors**, 21(5), 1853. [DOI:10.3390/s21051853](https://doi.org/10.3390/s21051853)

The project reproduces the paper's **sum-metric spatio-temporal variogram** (eq. 5)

$$
\gamma(h, u) = n \cdot \mathbf{1}\{h > 0 \lor u > 0\} + \gamma_s(h) + \gamma_t(u) + \gamma_j\!\left(\sqrt{h^2 + (k u)^2}\right)
$$

from scratch in Python (paper uses R / `gstat` / `sp` / `spacetime`), and applies it to:

1. A fully controlled **synthetic experiment** (known ground truth via GSTools Matérn ST-field).
2. **29 real NOAA/Meteostat hourly stations** around San Diego for the paper's two study windows (14–20 June 2020 and 14–20 September 2020).
3. The **Kaggle HPWREN Minute Weather** sensor (2013 subset) aggregated to hourly, as a single real on-site sensor.

## Deliverables

| File | Role |
|---|---|
| `kriging_project.ipynb` | **Main notebook**, 54 cells, 15 figures. Run end-to-end, every figure saved under `kriging_figures/`. |
| `kriging_lib.py` | From-scratch kriging library: Haversine, empirical + theoretical variograms, sum-metric model, ordinary / spatio-temporal / local-neighbourhood kriging, metrics. |
| `test_kriging_lib.py` | 10 unit tests (including regression tests for the zero-lag-binning bug and moving-neighbourhood correctness). |
| `build_notebook.py` | Programmatic builder — regenerates `kriging_project.ipynb` from source. |
| `kriging_data/fetch_noaa.py` | One-time NOAA data fetch via Meteostat. Outputs the CSVs used below. |
| `kriging_data/stations_*.csv` | Cached per-period station metadata and hourly observations (3 periods × 2 CSVs). |
| `kriging_figures/*.png` | All 15 figures produced by the notebook. |
| `kriging_figures/final_results.csv` | Machine-readable table of every RMSE / MAE / R² in the report. |
| `overview.tex` | Separate LaTeX section discussing the raw Kaggle dataset (precursor work, not part of the kriging analysis). |
| `analysis.py` | The EDA script that produced `figures/fig1_*…fig6_*` for `overview.tex`. |

## Requirements

Python 3.9+ with:

```
numpy scipy pandas matplotlib seaborn scikit-learn
scikit-gstat gstools pykrige meteostat
jupyter nbformat
```

Install into a venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib seaborn scikit-learn \
            scikit-gstat gstools pykrige meteostat \
            jupyter nbformat
```

## How to reproduce

```bash
python test_kriging_lib.py

python kriging_data/fetch_noaa.py

python build_notebook.py

jupyter nbconvert --to notebook --execute kriging_project.ipynb \
                  --output kriging_project.ipynb \
                  --ExecutePreprocessor.timeout=1200
```

Expected:
- All 10 tests pass
- `kriging_project.ipynb` regenerates with 0 execution errors
- 15 PNGs written under `kriging_figures/`
- `kriging_figures/final_results.csv` is overwritten with the new run's numbers

## About the big dataset (not checked in)

The HPWREN *Minute Weather* CSV is 122 MB and exceeds GitHub's 100 MB per-file limit, so it is excluded from this repo. Section 10 of the notebook needs it. Download it from Kaggle and place it at:

```
dataset/minute_weather.csv
```

Source: <https://www.kaggle.com/datasets/julianjose/minute-weather>

The notebook only uses the 2013-09-14 → 2013-09-20 slice, so if disk is tight you can pre-filter:

```python
import pandas as pd
df = pd.read_csv("minute_weather_full.csv", parse_dates=["hpwren_timestamp"])
mask = (df["hpwren_timestamp"] >= "2013-09-14") & (df["hpwren_timestamp"] <= "2013-09-20 23:59")
df[mask].to_csv("dataset/minute_weather.csv", index=False)
```

## Reproducibility notes

- All RNG seeds are fixed (`SEED = 20260421`).
- NOAA / Meteostat data are cached locally in `kriging_data/` so the notebook runs offline once the initial fetch is done.
- `final_results.csv` is the single source of truth for the numbers quoted in the notebook's conclusions. Those numbers are also embedded in `fig14_final_summary.png`.

## Main findings

- **Synthetic experiment (known truth).** Local-sensor kriging (44 points in a 1 km box around target) achieves RMSE ≈ 0.15 °C; sparse weather-station kriging (24 points over 30 × 30 km) gives RMSE ≈ 0.95 °C — roughly a 6× gap, consistent with the paper's qualitative ordering.
- **Real NOAA San Diego, summer 2020.** Ordinary ST-kriging LOO RMSE ≈ 4.6 °C; regression kriging with elevation detrending (fold-local trend + variogram, no leakage) improves this to ≈ 3.1 °C.
- **Real NOAA San Diego, autumn 2020.** Ordinary and regression kriging are essentially tied at ≈ 4.7 °C — the Sept 2020 heat wave dominates variance uniformly across elevation, so elevation detrending does not help.
- **Near-10 vs Far-10 real-station subsets.** Nearer cluster always beats the farther cluster (≈ 1.5–2.5 °C RMSE gap) — qualitatively the paper's finding, with a larger absolute gap due to San Diego's terrain heterogeneity (sea level to ~2000 m).
- **HPWREN single-sensor comparison.** Regression kriging from 23 NOAA stations achieves only negative R² against the HPWREN minute sensor, suggesting (but not proving, because the sensor's coordinates are only inferred) that regional public networks cannot fully substitute for on-site sensing when the site has a distinct microclimate.

See section 11 of the notebook for a full honest discussion of these numbers and their limitations.

## License

This repository is a class project; no license is asserted for the notebook/code. Underlying data:
- NOAA/Meteostat: public-access US government / provider data, see Meteostat's terms.
- HPWREN Minute Weather: per Kaggle dataset page terms.
- Kuo, Huang & Putra (2021): CC BY 4.0 (MDPI open-access).
