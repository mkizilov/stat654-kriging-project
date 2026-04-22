import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV = "dataset/minute_weather.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

df = pd.read_csv(CSV)
df["ts"] = pd.to_datetime(df["hpwren_timestamp"])
df = df.drop(columns=["hpwren_timestamp"])
df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
df["date"] = df["ts"].dt.date

print(f"Loaded {len(df):,} rows, {df['ts'].min()} → {df['ts'].max()}")

NUMERIC_MAIN = ["air_temp", "relative_humidity", "air_pressure",
                "avg_wind_speed", "max_wind_speed", "rain_accumulation"]

print("\n--- summary stats (for report) ---")
summary = df[NUMERIC_MAIN].describe().T[["count", "mean", "std", "min", "50%", "max"]]
print(summary.round(2))

print("\n--- missingness ---")
miss = df.isna().sum()
for c, n in miss.items():
    if n > 0:
        print(f"  {c:25s} {n:>8d}  ({100*n/len(df):.3f}%)")

daily = df.set_index("ts")[["air_temp", "relative_humidity",
                            "air_pressure", "avg_wind_speed",
                            "rain_accumulation"]].resample("1D").mean()
monthly_rain = df.set_index("ts")["rain_accumulation"].resample("1ME").sum()


fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)

ax = axes[0, 0]
ax.plot(daily.index, daily["air_temp"], color="#c0392b", lw=0.7)
ax.set_ylabel("Air temp (°F)")
ax.set_title("(a) Daily mean air temperature")

ax = axes[0, 1]
ax.plot(daily.index, daily["relative_humidity"], color="#2980b9", lw=0.7)
ax.set_ylabel("Relative humidity (%)")
ax.set_title("(b) Daily mean relative humidity")

ax = axes[1, 0]
ax.plot(daily.index, daily["air_pressure"], color="#16a085", lw=0.7)
ax.set_ylabel("Air pressure (mbar)")
ax.set_title("(c) Daily mean air pressure")

ax = axes[1, 1]
ax.bar(monthly_rain.index, monthly_rain.values, width=20,
       color="#34495e", alpha=0.85)
ax.set_ylabel("Rain accumulation (sum)")
ax.set_title("(d) Monthly rain accumulation (sum of minute values)")

for ax in axes.flat:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_timeseries.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig1_timeseries.png")


fig, axes = plt.subplots(2, 3, figsize=(11, 6))

axes[0, 0].hist(df["air_temp"].dropna(), bins=60, color="#c0392b",
                edgecolor="white", linewidth=0.2)
axes[0, 0].set_xlabel("Air temp (°F)")
axes[0, 0].set_ylabel("count")
axes[0, 0].set_title("(a) Air temperature")

axes[0, 1].hist(df["relative_humidity"].dropna(), bins=60, color="#2980b9",
                edgecolor="white", linewidth=0.2)
axes[0, 1].set_xlabel("Relative humidity (%)")
axes[0, 1].set_title("(b) Relative humidity")

axes[0, 2].hist(df["air_pressure"].dropna(), bins=60, color="#16a085",
                edgecolor="white", linewidth=0.2)
axes[0, 2].set_xlabel("Air pressure (mbar)")
axes[0, 2].set_title("(c) Air pressure")

axes[1, 0].hist(df["avg_wind_speed"].dropna(), bins=60, color="#8e44ad",
                edgecolor="white", linewidth=0.2)
axes[1, 0].set_xlabel("Avg wind speed")
axes[1, 0].set_ylabel("count")
axes[1, 0].set_title("(d) Average wind speed")

axes[1, 1].hist(df["max_wind_speed"].dropna(), bins=60, color="#d35400",
                edgecolor="white", linewidth=0.2)
axes[1, 1].set_xlabel("Max wind speed")
axes[1, 1].set_title("(e) Max wind speed")

rain_positive = df.loc[df["rain_accumulation"] > 0, "rain_accumulation"]
axes[1, 2].hist(rain_positive, bins=50, color="#34495e",
                edgecolor="white", linewidth=0.2)
axes[1, 2].set_yscale("log")
axes[1, 2].set_xlabel("Rain accumulation (> 0 only)")
axes[1, 2].set_title(f"(f) Rain, non-zero ({len(rain_positive):,} obs)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_distributions.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig2_distributions.png")


hourly = df.groupby("hour")[["air_temp", "relative_humidity",
                             "air_pressure", "avg_wind_speed"]].mean()

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

ax = axes[0, 0]
ax.plot(hourly.index, hourly["air_temp"], color="#c0392b", marker="o", ms=3)
ax.set_ylabel("Air temp (°F)")
ax.set_title("(a) Temperature vs hour of day")

ax = axes[0, 1]
ax.plot(hourly.index, hourly["relative_humidity"], color="#2980b9", marker="o", ms=3)
ax.set_ylabel("Relative humidity (%)")
ax.set_title("(b) Humidity vs hour of day")

ax = axes[1, 0]
ax.plot(hourly.index, hourly["air_pressure"], color="#16a085", marker="o", ms=3)
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Air pressure (mbar)")
ax.set_title("(c) Pressure vs hour of day")

ax = axes[1, 1]
ax.plot(hourly.index, hourly["avg_wind_speed"], color="#8e44ad", marker="o", ms=3)
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Avg wind speed")
ax.set_title("(d) Wind speed vs hour of day")

for ax in axes.flat:
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_diurnal.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig3_diurnal.png")


corr_cols = ["air_temp", "relative_humidity", "air_pressure",
             "avg_wind_speed", "max_wind_speed", "min_wind_speed",
             "rain_accumulation", "rain_duration"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.grid(False)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=40, ha="right")
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.iloc[i, j]
        color = "white" if abs(v) > 0.55 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=color, fontsize=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r")
ax.set_title("Correlation between weather variables")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4a_correlation.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig4a_correlation.png")

print("\n--- correlation matrix (for report) ---")
print(corr.round(2))


wind = df[["avg_wind_direction", "avg_wind_speed"]].dropna()
wind = wind.sample(n=200_000, random_state=42)

dir_bins = np.arange(0, 361, 22.5)
speed_bins = [0, 1, 2, 4, 6, 10, 40]
speed_labels = ["0-1", "1-2", "2-4", "4-6", "6-10", "10+"]

wind["dir_bin"] = pd.cut(wind["avg_wind_direction"], bins=dir_bins,
                         include_lowest=True)
wind["speed_bin"] = pd.cut(wind["avg_wind_speed"], bins=speed_bins,
                           labels=speed_labels, include_lowest=True,
                           right=False)

pivot = wind.groupby(["dir_bin", "speed_bin"], observed=True).size().unstack(fill_value=0)
pivot = 100 * pivot / pivot.values.sum()

theta = np.deg2rad((dir_bins[:-1] + dir_bins[1:]) / 2)
width = np.deg2rad(22.5) * 0.95

fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_subplot(111, projection="polar")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.grid(alpha=0.3)

colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(speed_labels)))
bottom = np.zeros(len(theta))
for i, label in enumerate(speed_labels):
    vals = pivot[label].values if label in pivot.columns else np.zeros(len(theta))
    ax.bar(theta, vals, width=width, bottom=bottom,
           color=colors[i], edgecolor="white", linewidth=0.4,
           label=label)
    bottom = bottom + vals

ax.set_title("Wind rose (direction + avg speed, % of observations)", pad=18)
ax.legend(title="speed", loc="upper right", bbox_to_anchor=(1.28, 1.08),
          fontsize=8, title_fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4b_windrose.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig4b_windrose.png")


print("\n--- extras for writeup ---")
print(f"total rows: {len(df):,}")
print(f"days present: {df['date'].nunique()} / {(df['ts'].max() - df['ts'].min()).days + 1} possible")
print(f"temp range : {df['air_temp'].min():.1f} - {df['air_temp'].max():.1f} F")
print(f"temp mean  : {df['air_temp'].mean():.1f} F")
print(f"humidity mean: {df['relative_humidity'].mean():.1f} %")
print(f"pressure mean: {df['air_pressure'].mean():.1f} mbar")
print(f"wind avg mean: {df['avg_wind_speed'].mean():.2f}, max observed max: {df['max_wind_speed'].max():.1f}")
rain_any = (df['rain_accumulation'] > 0).sum()
print(f"minutes with rain > 0: {rain_any:,} ({100*rain_any/len(df):.2f}%)")

dry_minutes_per_day = df.groupby("date")["rain_accumulation"].apply(lambda s: (s == 0).mean())
print(f"median fraction of dry minutes per day: {dry_minutes_per_day.median():.4f}")
dry_days = (df.groupby("date")["rain_accumulation"].sum() == 0).sum()
print(f"days with zero rain recorded: {dry_days} / {df['date'].nunique()}")


print("\n=== SUMMARY STATS FOR LATEX TABLE ===")
table_cols = ["air_temp", "relative_humidity", "air_pressure",
              "avg_wind_speed", "max_wind_speed", "min_wind_speed",
              "rain_accumulation", "rain_duration"]
stats = df[table_cols].describe().T
stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
print(stats.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n--- LaTeX rows ---")
pretty = {
    "air_temp": "Air temperature (°F)",
    "relative_humidity": "Relative humidity (\\%)",
    "air_pressure": "Air pressure (mbar)",
    "avg_wind_speed": "Avg wind speed",
    "max_wind_speed": "Max wind speed",
    "min_wind_speed": "Min wind speed",
    "rain_accumulation": "Rain accumulation",
    "rain_duration": "Rain duration (s)",
}
for c in table_cols:
    r = stats.loc[c]
    print(f"{pretty[c]} & {int(r['count']):,} & {r['mean']:.2f} & "
          f"{r['std']:.2f} & {r['min']:.2f} & {r['25%']:.2f} & "
          f"{r['50%']:.2f} & {r['75%']:.2f} & {r['max']:.2f} \\\\")


print("\n=== RAIN EVENTS ===")
is_rain = (df["rain_accumulation"].fillna(0) > 0).values
run_change = np.concatenate([[True], is_rain[1:] != is_rain[:-1]])
run_id = np.cumsum(run_change)
tmp = pd.DataFrame({
    "ts": df["ts"].values,
    "acc": df["rain_accumulation"].fillna(0).values,
    "dur": df["rain_duration"].fillna(0).values,
    "raining": is_rain,
    "run_id": run_id,
})
events = tmp[tmp["raining"]].groupby("run_id").agg(
    start=("ts", "first"),
    duration_min=("ts", "count"),
    total_accum=("acc", "sum"),
    total_duration_sec=("dur", "sum"),
).reset_index(drop=True)
events["start_month"] = events["start"].dt.month

print(f"Number of rain events (contiguous non-zero rain): {len(events):,}")
print(f"Median event duration (min): {events['duration_min'].median():.1f}")
print(f"Mean event duration   (min): {events['duration_min'].mean():.1f}")
print(f"Max event duration    (min): {events['duration_min'].max()}")
print(f"Median event accumulation:   {events['total_accum'].median():.2f}")
print(f"Mean event accumulation:     {events['total_accum'].mean():.2f}")
print(f"Max event accumulation:      {events['total_accum'].max():.2f}")
print(f"Events that are 1 minute only: "
      f"{(events['duration_min'] == 1).sum()} "
      f"({100*(events['duration_min'] == 1).mean():.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

month_counts = events["start_month"].value_counts().sort_index()
month_counts = month_counts.reindex(range(1, 13), fill_value=0)
axes[0].bar(month_counts.index, month_counts.values, color="#34495e",
            edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Month of year")
axes[0].set_ylabel("Number of rain events")
axes[0].set_title("(a) Events by month (3 yr total)")
axes[0].set_xticks(range(1, 13))

bins_dur = np.logspace(0, np.log10(events["duration_min"].max() + 1), 40)
axes[1].hist(events["duration_min"], bins=bins_dur, color="#2980b9",
             edgecolor="white", linewidth=0.3)
axes[1].set_xscale("log")
axes[1].set_xlabel("Event duration (minutes, log scale)")
axes[1].set_ylabel("Count of events")
axes[1].set_title("(b) Event duration")

bins_acc = np.logspace(np.log10(events["total_accum"][events["total_accum"] > 0].min()),
                       np.log10(events["total_accum"].max() + 1), 40)
axes[2].hist(events["total_accum"], bins=bins_acc, color="#16a085",
             edgecolor="white", linewidth=0.3)
axes[2].set_xscale("log")
axes[2].set_xlabel("Total accumulation per event (log scale)")
axes[2].set_ylabel("Count of events")
axes[2].set_title("(c) Event total rain")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_rain_events.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig5_rain_events.png")


pivot_temp = df.groupby(["month", "hour"])["air_temp"].mean().unstack()
pivot_hum = df.groupby(["month", "hour"])["relative_humidity"].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, data, title, cmap, cb_label in [
    (axes[0], pivot_temp, "(a) Mean air temperature (°F)", "RdYlBu_r", "°F"),
    (axes[1], pivot_hum,  "(b) Mean relative humidity (%)", "YlGnBu", "%"),
]:
    ax.grid(False)
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(range(0, 24, 3))
    ax.set_yticks(range(12))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Hour of day (local)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cb_label)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_month_hour_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig6_month_hour_heatmap.png")
