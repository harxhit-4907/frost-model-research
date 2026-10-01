"""
label_rules.py — physically-defined frost/cold-injury label rules.

Frost has a known physical definition, so labels are *computed* from raw
temperature (and, for the stricter variant, humidity) — not learned,
clustered, or taken from a dataset-provided column. See Omazić et al.
(2024, Agricultural and Forest Meteorology), which validated 11
frost-detection methods against real recorded frost occurrence across 74
Croatian stations over 40 years.

Two label variants are implemented here and BOTH should be produced and
reported side by side (see reports/comparison_table.md), because they
diverge for a real, citable physical reason — not because we couldn't
decide:

  - `label_tmin_only`   (broad):    Tmin < 3.0 C
        Not gated on humidity at all. Catches "black frost" — real,
        crop-damaging cold with no visible ice, because the air was too
        dry for frost to physically form. Higher recall, more false
        positives *relative to visually observed frost specifically*.

  - `label_tmin_and_td` (strict, Omazić et al.'s validated "Method 7",
    Tmin3_Td0): Tmin < 3.0 C AND dew point (computed from Tmin and the
    *daily mean* RH) < 0.0 C.
        This is the paper's best-scoring rule against their real station
        records (POD 0.98, POFD 0.18 all-year). But their ground truth is
        an observer visually confirming ice on grass — a white-frost
        record by construction. The paper separately notes black frost at
        their continental stations on very cold, very dry nights (roughly
        below -5C with dew point below roughly -10C), which no observer
        could log as "frost occurred." So some of what Method 7 scores as
        a false alarm may actually be a real, unrecorded black-frost
        event — i.e. its reported false-alarm rate is plausibly inflated
        by exactly the failure mode this project is trying to avoid.

Neither variant is "the" label. Report both and let the divergence between
them be part of the write-up, per the project's label-definition note.
"""

import pandas as pd

from features import dew_point_from_rh

DEFAULT_TEMP_THRESHOLD_C = 3.0
DEFAULT_TD_THRESHOLD_C = 0.0


def label_tmin_only(
    future_temp_window: pd.Series, threshold: float = DEFAULT_TEMP_THRESHOLD_C
) -> int:
    """
    Broad frost/cold-injury label: does the minimum temperature over the
    prediction window drop to or below `threshold`?

    Deliberately NOT gated on humidity/dew point, so it also catches
    black-frost-style damage that never becomes visible ice.
    """
    return int(future_temp_window.min() <= threshold)


def label_tmin_and_td(
    future_temp_window: pd.Series,
    future_rh_mean_percent_window: pd.Series,
    temp_threshold: float = DEFAULT_TEMP_THRESHOLD_C,
    td_threshold: float = DEFAULT_TD_THRESHOLD_C,
) -> int:
    """
    Omazić et al. (2024) "Method 7" (Tmin3_Td0): Tmin < temp_threshold AND
    dew point (from Tmin + daily-mean RH) < td_threshold.

    Tuned to *visible* (white) frost — will systematically under-catch
    black frost by design (see module docstring). Report this alongside
    `label_tmin_only`, not instead of it.

    Args:
        future_temp_window: temperatures over the prediction horizon.
        future_rh_mean_percent_window: the *daily mean* RH, in PERCENT
            (0-100, not a 0-1 fraction — convert first if your source
            uses fractions, see features.humidity_fraction_to_percent),
            aligned index-for-index with future_temp_window.
        temp_threshold: deg C.
        td_threshold: deg C.
    """
    idx_of_min = future_temp_window.idxmin()
    tmin = future_temp_window.loc[idx_of_min]
    rh_mean_at_tmin = future_rh_mean_percent_window.loc[idx_of_min]
    td = dew_point_from_rh(tmin, rh_mean_at_tmin)
    return int((tmin < temp_threshold) and (td < td_threshold))


def build_labels_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    temp_col: str = "temp_min",
    rh_pct_col: str = "humidity_pct",
    temp_threshold: float = DEFAULT_TEMP_THRESHOLD_C,
    td_threshold: float = DEFAULT_TD_THRESHOLD_C,
) -> pd.DataFrame:
    """
    Slide a forward-looking window of `horizon` rows across `df` and attach
    three columns to each remaining row: the continuous target
    (`target_min_temp`, the minimum temp over the next `horizon` rows) and
    both label variants (`frost_label_broad`, `frost_label_strict`).

    This is the "(X, y) via windowing" step from the project's label note:
    X is whatever past-N-rows feature columns the caller has already built
    (untouched here), y is the frost outcome over the *next* `horizon`
    rows. The last `horizon` rows of `df` have no future window and are
    dropped.

    Args:
        df: must already contain `temp_col` and `rh_pct_col` (percent,
            0-100 — convert first with features.humidity_fraction_to_percent
            if your source uses a 0-1 fraction, as the Tier-1 dataset does).
        horizon: number of rows ahead to look for the frost outcome (rows,
            not necessarily hours — see features.py's naming-convention
            note on this).
    """
    n = len(df)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n <= horizon:
        raise ValueError("Not enough rows for the requested horizon.")

    temps = df[temp_col].to_numpy()
    rhs = df[rh_pct_col].to_numpy()

    targets, broads, stricts = [], [], []
    for i in range(n - horizon):
        window_temp = pd.Series(temps[i + 1 : i + 1 + horizon])
        window_rh = pd.Series(rhs[i + 1 : i + 1 + horizon])
        targets.append(window_temp.min())
        broads.append(label_tmin_only(window_temp, temp_threshold))
        stricts.append(
            label_tmin_and_td(window_temp, window_rh, temp_threshold, td_threshold)
        )

    out = df.iloc[: n - horizon].reset_index(drop=True).copy()
    out["target_min_temp"] = targets
    out["frost_label_broad"] = broads
    out["frost_label_strict"] = stricts
    return out
