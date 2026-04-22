"""Download hourly temperature data from NOAA/Meteostat stations around San Diego.

Periods:
  - June 14-20 2020 (summer, paper replication)
  - September 14-20 2020 (autumn, paper replication)
  - September 14-20 2013 (inside minute_weather.csv date range for sensor co-analysis)

Target location (HPWREN-like site in San Diego mountains, ~860 m elevation inferred
from minute_weather air_pressure ~917 mbar):
  lat 33.1 N, lon -116.6 W

Run once; notebook loads the CSVs it emits.
"""
import os
import pandas as pd
from datetime import datetime
from meteostat import Stations, Hourly

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_LAT = 33.1
TARGET_LON = -116.6

PERIODS = [
    ("summer_2020", datetime(2020, 6, 14), datetime(2020, 6, 20, 23, 59)),
    ("autumn_2020", datetime(2020, 9, 14), datetime(2020, 9, 20, 23, 59)),
    ("autumn_2013", datetime(2013, 9, 14), datetime(2013, 9, 20, 23, 59)),
]

NEAR_N = 40
MIN_OBS = 120


def fetch_period(tag, start, end):
    print(f"=== {tag}: {start} → {end} ===")
    stations = Stations().nearby(TARGET_LAT, TARGET_LON).fetch(NEAR_N)
    rows = []
    meta_rows = []
    for sid in stations.index:
        data = Hourly(sid, start, end).fetch()
        if "temp" not in data.columns:
            continue
        n = data["temp"].notna().sum()
        if n < MIN_OBS:
            continue
        temp = data["temp"].rename("temp").reset_index()
        temp["station_id"] = sid
        rows.append(temp[["station_id", "time", "temp"]])
        meta_rows.append({
            "station_id": sid,
            "name": stations.loc[sid, "name"],
            "latitude": float(stations.loc[sid, "latitude"]),
            "longitude": float(stations.loc[sid, "longitude"]),
            "elevation": float(stations.loc[sid, "elevation"]) if not pd.isna(
                stations.loc[sid, "elevation"]
            ) else float("nan"),
            "country": stations.loc[sid, "country"],
            "n_obs": int(n),
        })
    if not rows:
        print(f"  WARNING: no stations qualified for {tag}")
        return None
    obs = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    print(f"  kept {len(meta)} stations, {len(obs):,} total hourly obs")
    obs_path = os.path.join(HERE, f"stations_obs_{tag}.csv")
    meta_path = os.path.join(HERE, f"stations_meta_{tag}.csv")
    obs.to_csv(obs_path, index=False)
    meta.to_csv(meta_path, index=False)
    print(f"  wrote {obs_path}")
    print(f"  wrote {meta_path}")
    return meta, obs


if __name__ == "__main__":
    for tag, start, end in PERIODS:
        fetch_period(tag, start, end)
    print("\nDone.")
