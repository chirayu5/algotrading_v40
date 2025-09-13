import datetime as dt

import pandas as pd
import pytest

import algotrading_v40.bar_groupers.time_based as bg_tb
import algotrading_v40.constants as ctnts


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _market_ts(minutes_after_open: int) -> pd.Timestamp:
  """
  Convenience: return a Timestamp <minutes_after_open> minutes after the
  first (minute-bar-close) timestamp of the trading session, in UTC.
  """
  base = pd.Timestamp.combine(
    dt.date.today(), ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC")
  return base + pd.Timedelta(minutes=minutes_after_open)


# --------------------------------------------------------------------------- #
# parameterised “good-path’’ tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
  "group_size, minutes_after_open, expected_first_ideal, expected_duration",
  [
    # 1) group_size divides 375 exactly - straight-forward case
    (
      15,  # group_size
      5,  # ts is 5 minutes after open => inside 1st group
      _market_ts(15 * 0),  # first ideal ts of that group
      15,  # duration = group_size
    ),
    (
      15,  # group_size
      372,  # place in the last group
      _market_ts(15 * 24),
      15,
    ),
    # 2) remainder < group_size / 2 => last group will be bigger
    (
      60,  # 375 % 60 == 15 (< 30) => last_group_size = 75
      123,  # 2 hours 3 min after open => inside a regular group
      _market_ts(60 * 2),  # 0-based groups of 60 min => ideal first = 120
      60,
    ),
    (
      60,  # 375 % 60 == 15 (< 30) => last_group_size = 75
      363,  # 6 hours 3 min after open => place in the last group
      _market_ts(60 * 5),
      75,
    ),
    (
      60,  # 375 % 60 == 15 (< 30) => last_group_size = 75
      304,  # 5 hours 4 min after open => place in the last group
      _market_ts(60 * 5),
      75,
    ),
    # 3) remainder >= group_size/2 => an extra (smaller) last group appears
    (
      100,  # 375 % 100 == 75 (>= 50) => extra last group, size 75
      250,  # inside the 3rd "100-minute" group
      _market_ts(200),
      100,
    ),
    (
      100,
      300,  # inside the 4th "75-minute" group (the FIRST member of the 4th "75-minute" group in fact)
      _market_ts(300),
      75,
    ),
    (
      100,
      372,  # inside the 4th "75-minute" group
      _market_ts(300),
      75,
    ),
  ],
)
def test_group_info_for_indian_market(
  group_size, minutes_after_open, expected_first_ideal, expected_duration
):
  ts = _market_ts(minutes_after_open)
  res = bg_tb.group_info_for_indian_market(ts=ts, group_size=group_size, offset=0)
  assert res.duration == expected_duration
  assert res.first_ideal_ts == expected_first_ideal
  assert (
    res.first_ideal_ts <= ts < res.first_ideal_ts + pd.Timedelta(minutes=res.duration)
  )


# --------------------------------------------------------------------------- #
# validation / error-handling tests
# --------------------------------------------------------------------------- #
def test_group_size_must_be_within_bounds():
  ts = _market_ts(0)
  with pytest.raises(ValueError):
    bg_tb.group_info_for_indian_market(ts, group_size=0, offset=0)  # <= 0
  with pytest.raises(ValueError):
    bg_tb.group_info_for_indian_market(ts, group_size=376, offset=0)  # > 375


# --------------------------------------------------------------------------- #
# tests for get_time_based_bar_group_for_indian_market
# --------------------------------------------------------------------------- #


def test_get_time_based_bar_group_basic_functionality():
  timestamps = [
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # 0 min
    pd.Timestamp("2023-01-02 03:46:59.999000+00:00"),  # 1 min
    pd.Timestamp("2023-01-02 03:55:59.999000+00:00"),  # 10 min
    pd.Timestamp("2023-01-02 03:59:59.999000+00:00"),  # 14 min
    pd.Timestamp("2023-01-02 04:00:59.999000+00:00"),  # 15 min
    pd.Timestamp("2023-01-02 04:05:59.999000+00:00"),  # 20 min
    pd.Timestamp("2023-01-02 04:14:59.999000+00:00"),  # 29 min
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104, 105, 106]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=13)

  expected_groups = [
    timestamps[0],
    timestamps[0],
    timestamps[0],
    timestamps[3],
    timestamps[3],
    timestamps[3],
    timestamps[6],
  ]

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_with_missing_bars():
  """Test grouping when some minute bars are missing."""
  # 0,14 THERE
  # 15,29 MISSING
  # 30,44 THERE
  # 45,59 THERE
  timestamps = [
    # Ideal group 1
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # 0 min
    pd.Timestamp("2023-01-02 03:50:59.999000+00:00"),  # 5 min (gap from 1-4)
    pd.Timestamp("2023-01-02 03:59:59.999000+00:00"),  # 14 min
    # GAP (Ideal group 2 MISSING)
    # Ideal group 3
    pd.Timestamp("2023-01-02 04:28:59.999000+00:00"),  # 43 min
    # Ideal group 4
    pd.Timestamp("2023-01-02 04:34:59.999000+00:00"),  # 49 min
    pd.Timestamp("2023-01-02 04:35:59.999000+00:00"),  # 50 min
    pd.Timestamp("2023-01-02 04:40:59.999000+00:00"),  # 55 min
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104, 105, 106]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=15)

  expected_groups = [
    timestamps[0],
    timestamps[0],
    timestamps[0],
    timestamps[3],
    timestamps[4],
    timestamps[4],
    timestamps[4],
  ]

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_remainder_smaller_case():
  """Test with group_size=60 where remainder < group_size/2 (last group gets bigger)."""
  # 375 % 60 = 15 (< 30), so last group size = 75
  # Groups: [0-59], [60-119], [120-179], [180-239], [240-299], [300-374] (last group: 75 min)
  timestamps = [
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # 0 min (group 1)
    pd.Timestamp("2023-01-02 04:15:59.999000+00:00"),  # 30 min (group 1)
    pd.Timestamp("2023-01-02 04:44:59.999000+00:00"),  # 59 min (last of group 1)
    pd.Timestamp("2023-01-02 04:45:59.999000+00:00"),  # 60 min (first of group 2)
    pd.Timestamp("2023-01-02 06:45:59.999000+00:00"),  # 180 min (first of group 4)
    pd.Timestamp("2023-01-02 08:45:59.999000+00:00"),  # 300 min (first of last group)
    pd.Timestamp("2023-01-02 09:30:59.999000+00:00"),  # 345 min (in last group)
    pd.Timestamp("2023-01-02 09:59:59.999000+00:00"),  # 374 min (last bar)
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104, 105, 106, 107]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=60)

  expected_groups = [
    timestamps[0],
    timestamps[0],
    timestamps[0],
    ####
    timestamps[3],
    ####
    timestamps[4],
    ####
    timestamps[5],
    timestamps[5],
    timestamps[5],
  ]

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_remainder_larger_case():
  """Test with group_size=100 where remainder >= group_size/2 (extra last group appears)."""
  # 375 % 100 = 75 (>= 50), so we get: [0-99], [100-199], [200-299], [300-374]
  # 4 groups total: 3 of size 100, 1 of size 75
  timestamps = [
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # 0 min (group 1)
    pd.Timestamp("2023-01-02 05:24:59.999000+00:00"),  # 99 min (last of group 1)
    pd.Timestamp("2023-01-02 05:25:59.999000+00:00"),  # 100 min (first of group 2)
    pd.Timestamp("2023-01-02 09:30:59.999000+00:00"),  # 345 min (in group 4)
    pd.Timestamp("2023-01-02 09:59:59.999000+00:00"),  # 374 min (last bar)
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=100)

  expected_groups = [
    timestamps[0],
    timestamps[0],
    ###
    timestamps[2],
    ###
    timestamps[3],
    timestamps[3],
  ]

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_single_large_group():
  """Test with group_size=375 (single group for entire session)."""
  timestamps = [
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # 0 min
    pd.Timestamp("2023-01-02 06:00:59.999000+00:00"),  # 135 min
    pd.Timestamp("2023-01-02 08:30:59.999000+00:00"),  # 285 min
    pd.Timestamp("2023-01-02 09:59:59.999000+00:00"),  # 374 min (last)
    pd.Timestamp("2023-01-04 08:30:59.999000+00:00"),  # Different day starts
    pd.Timestamp("2023-01-04 09:59:59.999000+00:00"),  #
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104, 101]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=375)

  # All bars should belong to the same group (first timestamp)
  expected_groups = [timestamps[0]] * 4 + [timestamps[4]] * 2

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_cross_day_boundary():
  """Test that function works with timestamps from different days."""
  # Test with 3 bars from different trading days to ensure day-based logic works
  timestamps = [
    pd.Timestamp("2023-01-02 03:45:59.999000+00:00"),  # Day 1, 0 min
    pd.Timestamp("2023-01-02 04:00:59.999000+00:00"),  # Day 1, 15 min
    pd.Timestamp("2023-01-03 03:45:59.999000+00:00"),  # Day 2, 0 min (new day)
    pd.Timestamp("2023-01-03 04:14:59.999000+00:00"),  # Day 2, 29 min
    pd.Timestamp("2023-01-03 04:15:59.999000+00:00"),  # Day 2, 30 min
  ]

  df = pd.DataFrame(
    {"price": [100, 101, 102, 103, 104]},
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=30)

  # Each day should start its own groups
  expected_groups = [
    timestamps[0],  # Day 1, group 1
    timestamps[0],  # Day 1, group 1
    timestamps[2],  # Day 2, group 1
    timestamps[2],  # Day 2, group 1
    timestamps[4],  # Day 2, group 2
  ]

  assert result.tolist() == expected_groups


def test_get_time_based_bar_group_single_bar():
  """Test with single bar."""
  timestamps = [pd.Timestamp("2023-01-02 04:14:59.999000+00:00")]

  df = pd.DataFrame(
    {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
  )

  result = bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=15)

  assert result.tolist() == [timestamps[0]]


# --------------------------------------------------------------------------- #
# error-handling tests for get_time_based_bar_group_for_indian_market
# --------------------------------------------------------------------------- #


def test_get_time_based_bar_group_invalid_group_size():
  """Test error handling for invalid group_size."""
  timestamps = [pd.Timestamp("2023-01-02 03:45:59.999000+00:00")]
  df = pd.DataFrame(
    {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
  )

  # Test group_size <= 0
  with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
    bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=0)

  with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
    bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=-1)

  # Test group_size > 375
  with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
    bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=376)
