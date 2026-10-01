"""
data_loader.py — load raw weather data and normalize it into a per-city
shape that label_rules.py, features.py, baselines.py, and models.py can
all consume the same way, regardless of which tier of dataset it came
from.

Currently implements the Tier 1 loader (florian-huber/weather_prediction_dataset
via Zenodo, daily data for 18 European cities, 2000-2010). Tier 2/3
loaders (Korea ASOS / Murcia / NASA POWER / AgERA5) are intentionally not
built yet — the brief time-boxes Tier 2 and defers Tier 3 to weeks 9-10,
so building those loaders now would be premature. Add them here, next to
this one, when that phase starts, keeping the same per-city DataFrame
*shape* returned by get_city_frame() so nothing downstream has to change
just because the data source changed.

UNIT NOTE: in this Tier-1 dataset, `<CITY>_humidity` is a FRACTION in
[0, 1], not a percentage, and `<CITY>_cloud_cover` is in oktas (0-8, WMO
convention), not percent. `get_city_frame()` adds a `humidity_pct` column
(0-100) so callers don't have to remember the fraction->percent
conversion at every call site — see features.py's unit warning for why
getting this wrong is a silent, not a loud, failure.
"""

from typing import List

import pandas as pd

TIER1_DEFAULT_PATH = "data/raw/weather_prediction_dataset.csv"

# Suffixes actually present across the 18 Tier-1 stations. Not every city
# has every suffix (e.g. Basel has no wind columns) — that's expected and
# handled by get_city_frame() returning whatever columns exist.
_KNOWN_SUFFIXES = (
    "_cloud_cover",
    "_humidity",
    "_pressure",
    "_global_radiation",
    "_precipitation",
    "_sunshine",
    "_temp_mean",
    "_temp_min",
    "_temp_max",
    "_wind_speed",
    "_wind_gust",
)


def load_tier1_raw(path: str = TIER1_DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw Tier-1 (multi-city, wide-format) CSV, parsing DATE."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    return df


def list_cities(df: pd.DataFrame) -> List[str]:
    """Return the distinct city prefixes present as columns in a Tier-1 frame."""
    cities = set()
    for col in df.columns:
        for suffix in _KNOWN_SUFFIXES:
            if col.endswith(suffix):
                cities.add(col[: -len(suffix)])
                break
    return sorted(cities)


def get_city_frame(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Extract one city's columns into a plain, city-prefix-free DataFrame
    with columns like DATE, temp_min, temp_max, temp_mean, humidity
    (fraction, as-is from source), humidity_pct (0-100, derived),
    cloud_cover (oktas), pressure, precipitation, sunshine, and
    wind_speed/wind_gust *where that station reports them*.

    Missing variables for a given city are simply absent from the
    returned frame (not filled with NaN) — check `df.columns` before
    relying on wind/pressure/etc. for a given city.
    """
    prefix = f"{city}_"
    rename_map = {c: c[len(prefix):] for c in df.columns if c.startswith(prefix)}
    if not rename_map:
        raise ValueError(
            f"No columns found for city '{city}'. Available cities: {list_cities(df)}"
        )
    out = df[["DATE"] + list(rename_map.keys())].rename(columns=rename_map).copy()
    if "humidity" in out.columns:
        out["humidity_pct"] = out["humidity"] * 100.0
    out.insert(0, "city", city)
    return out


def missingness_report(df: pd.DataFrame) -> pd.Series:
    """
    Per-column count of missing values, most-missing first. Call this
    during EDA before any modeling — the brief explicitly calls out
    missingness as something to check, not assume away.
    """
    return df.isna().sum().sort_values(ascending=False)
