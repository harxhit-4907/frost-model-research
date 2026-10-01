import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from label_rules import (  # noqa: E402
    build_labels_for_horizon,
    label_tmin_and_td,
    label_tmin_only,
)


def test_label_tmin_only_flags_a_cold_window():
    window = pd.Series([5.0, 2.5, -1.0, 1.0])
    assert label_tmin_only(window, threshold=3.0) == 1


def test_label_tmin_only_ignores_a_warm_window():
    window = pd.Series([6.0, 5.0, 4.0])
    assert label_tmin_only(window, threshold=3.0) == 0


def test_label_tmin_only_is_inclusive_at_the_threshold():
    window = pd.Series([3.0, 4.0])
    assert label_tmin_only(window, threshold=3.0) == 1


def test_label_tmin_and_td_flags_a_cold_dry_night():
    # Tmin well below threshold AND dry enough that dew point is well below 0.
    temp_window = pd.Series([1.0])
    rh_window = pd.Series([60.0])  # percent
    assert label_tmin_and_td(temp_window, rh_window) == 1


def test_label_tmin_and_td_does_not_fire_on_warm_dry_nights():
    temp_window = pd.Series([8.0])
    rh_window = pd.Series([40.0])
    assert label_tmin_and_td(temp_window, rh_window) == 0


def test_variants_diverge_on_a_black_frost_style_case():
    """
    This is the divergence the project report is supposed to discuss:
    cold enough to injure crops (Tmin < 3C) but humid enough that the
    computed dew point stays >= 0C, so the strict/visible-frost variant
    stays silent while the broad variant still flags it.

    Concretely: 2.8C air temp at 90% RH -> dew point ~= 1.3C (>= 0C), via
    the same Magnus formula label_rules uses internally.
    """
    temp_window = pd.Series([2.8])
    rh_window = pd.Series([90.0])

    broad = label_tmin_only(temp_window, threshold=3.0)
    strict = label_tmin_and_td(temp_window, rh_window)

    assert broad == 1
    assert strict == 0


def test_variants_agree_on_a_clearly_cold_and_dry_case():
    temp_window = pd.Series([-2.0])
    rh_window = pd.Series([50.0])

    broad = label_tmin_only(temp_window, threshold=3.0)
    strict = label_tmin_and_td(temp_window, rh_window)

    assert broad == 1
    assert strict == 1


def test_label_tmin_and_td_uses_rh_aligned_with_the_minimum_temperature():
    # The coldest hour in the window is index 1 (0.5C), which is dry
    # (RH 55%) -> should fire. The other hour is milder but very humid;
    # that humidity must NOT be the one used for the dew-point check.
    temp_window = pd.Series({"t0": 2.9, "t1": 0.5})
    rh_window = pd.Series({"t0": 99.0, "t1": 55.0})
    assert label_tmin_and_td(temp_window, rh_window) == 1


def test_build_labels_for_horizon_shifts_forward_and_drops_tail():
    df = pd.DataFrame(
        {
            "temp_min": [10.0, 2.0, -1.0, 8.0, 6.0],
            "humidity_pct": [70.0, 70.0, 50.0, 70.0, 70.0],
        }
    )
    out = build_labels_for_horizon(df, horizon=1)

    # Last row dropped (no "tomorrow" to look at).
    assert len(out) == 4
    # Row 0's target is tomorrow's (row 1's) temp_min: 2.0.
    assert out["target_min_temp"].iloc[0] == 2.0
    assert out["frost_label_broad"].iloc[0] == 1  # 2.0 < 3.0
    # Row 1's target is row 2's temp_min (-1.0, dry at 50% RH) -> both variants fire.
    assert out["target_min_temp"].iloc[1] == -1.0
    assert out["frost_label_broad"].iloc[1] == 1
    assert out["frost_label_strict"].iloc[1] == 1


def test_build_labels_for_horizon_rejects_too_short_input():
    df = pd.DataFrame({"temp_min": [1.0], "humidity_pct": [50.0]})
    try:
        build_labels_for_horizon(df, horizon=1)
        assert False, "expected a ValueError for insufficient rows"
    except ValueError:
        pass
