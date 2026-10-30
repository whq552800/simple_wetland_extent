"""Checks for the two ambiguous steps in ELM/CWatM area comparison."""

from datetime import date, timedelta

import numpy as np
import pytest

from tools.compare_wetland_with_elm import month_indices, wetland_union


def test_union_removes_surface_overlap_on_vegetated_wetland():
    union = wetland_union(
        saturated_veg=np.array([0.20, 0.0]),
        surface=np.array([0.10, 0.10]),
        veg_fraction=np.array([0.30, 0.0]),
        flood_fraction=np.array([0.10, 0.20]),
    )
    np.testing.assert_allclose(union, [0.225, 0.10])


def test_month_indices_exclude_gregorian_leap_day_for_365_day_reference():
    dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(366)]
    january = month_indices(dates, 2000, 1)
    february = month_indices(dates, 2000, 2)
    december = month_indices(dates, 2000, 12)
    assert len(january) == 31
    assert len(february) == 28
    assert len(december) == 31
    assert 59 not in february  # February 29 in Gregorian 2000
    with pytest.raises(ValueError, match="complete"):
        month_indices(dates[:20], 2000, 1)
