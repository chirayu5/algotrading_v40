import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.bar_groupers.time_based as bg_tb
import algotrading_v40.constants as ctnts
import algotrading_v40.utils.streaming as u_s
import algotrading_v40.utils.testing as u_t

CURR_DAY = dt.date(2025, 9, 15)
PREV_DAY = dt.date(2025, 9, 12)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _market_ts(minutes_after_open: int, day: dt.date = CURR_DAY) -> pd.Timestamp:
  """
  Convenience: return a Timestamp <minutes_after_open> minutes after the
  first (minute-bar-close) timestamp of the trading session, in UTC.
  """
  base = pd.Timestamp.combine(
    day, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC")
  return base + pd.Timedelta(minutes=minutes_after_open)


class TestGroupInfoForIndianMarket:
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
  def test_without_offset(
    self,
    group_size,
    minutes_after_open,
    expected_first_ideal,
    expected_duration,
  ):
    ts = _market_ts(minutes_after_open)
    res = bg_tb.group_info_for_indian_market(
      ts=ts,
      group_size=group_size,
      offset=0,
      prev_date=None,
    )
    assert res.duration == expected_duration
    assert res.first_ideal_ts == expected_first_ideal
    assert (
      res.first_ideal_ts <= ts < res.first_ideal_ts + pd.Timedelta(minutes=res.duration)
    )

  @pytest.mark.parametrize(
    "offset, group_size, minutes_after_open, expected_first_ideal_ts, expected_duration, prev_date",
    [
      (
        3,  # offset
        15,  # group_size
        17,  # timestamp is 17 min after open => 14 min into offset session => first group
        _market_ts(3),  # first ideal should be at offset (3 min after open)
        15,  # duration = group_size
        None,
      ),
      (
        3,  # offset
        15,  # group_size
        18,  # timestamp is 18 min after open => 15 min into offset session (so after the first group's 0-14) => second group
        _market_ts(18),  # second group after offset
        15,  # duration = group_size
        None,
      ),
      (
        3,  # offset
        15,  # group_size
        2,  # belongs to the previous day's last group
        pd.Timestamp.combine(
          PREV_DAY,
          ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC,
        ).tz_localize("UTC")
        + pd.Timedelta(minutes=360 + 3),
        15,
        PREV_DAY,
      ),
      # 0-59 60-119 120-179 180-239 240-299 300-374
      (
        7,  # offset
        60,  # group_size
        6,  # belongs to the previous day's last group
        pd.Timestamp.combine(
          PREV_DAY,
          ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC,
        ).tz_localize("UTC")
        + pd.Timedelta(minutes=300 + 7),
        75,
        PREV_DAY,
      ),
      # 0-99 100-199 200-299 300-374
      (
        11,  # offset
        100,  # group_size
        7,  # belongs to the previous day's last group
        pd.Timestamp.combine(
          PREV_DAY,
          ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC,
        ).tz_localize("UTC")
        + pd.Timedelta(minutes=300 + 11),
        75,
        PREV_DAY,
      ),
      (
        11,  # offset
        100,  # group_size
        7,  # belongs to the previous day's last group
        None,
        None,
        None,  # prev_date is None, group's first ts lies on the previous day => None will be returned
      ),
    ],
  )
  def test_with_offset(
    self,
    offset,
    group_size,
    minutes_after_open,
    expected_first_ideal_ts,
    expected_duration,
    prev_date,
  ):
    ts = _market_ts(minutes_after_open)
    res = bg_tb.group_info_for_indian_market(
      ts=ts,
      group_size=group_size,
      offset=offset,
      prev_date=prev_date,
    )
    assert res.duration == expected_duration
    assert res.first_ideal_ts == expected_first_ideal_ts

  def test_offset_validation(self):
    """Test that offset validation works correctly."""
    ts = _market_ts(10)

    # Test with offset that's too large for the last group
    # For group_size=60: 375 % 60 = 15 < 30, so last_group_size = 75
    # Valid offset range should be 0 to 74
    with pytest.raises(ValueError, match="offset must be between 0 and 74"):
      bg_tb.group_info_for_indian_market(ts, group_size=60, offset=75, prev_date=None)

    # Test with negative offset
    with pytest.raises(ValueError, match="offset must be between 0 and"):
      bg_tb.group_info_for_indian_market(ts, group_size=60, offset=-1, prev_date=None)

  def test_group_size_must_be_within_bounds(self):
    ts = _market_ts(0)
    with pytest.raises(ValueError):
      bg_tb.group_info_for_indian_market(
        ts, group_size=0, offset=0, prev_date=None
      )  # <= 0
    with pytest.raises(ValueError):
      bg_tb.group_info_for_indian_market(
        ts, group_size=376, offset=0, prev_date=None
      )  # > 375


class TestGetTimeBasedBarGroupForIndianMarket:
  def test_basic_functionality(self):
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(1, CURR_DAY),
      _market_ts(12, CURR_DAY),
      _market_ts(13, CURR_DAY),
      _market_ts(15, CURR_DAY),
      _market_ts(25, CURR_DAY),
      _market_ts(26, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0-12, 13-25, 26-38,...
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=13, offset=0
    )

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

  def test_with_missing_bars(self):
    """Test grouping when some minute bars are missing."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(5, CURR_DAY),
      _market_ts(14, CURR_DAY),
      # GAP (Ideal group 2 MISSING)
      # Ideal group 3
      _market_ts(30, CURR_DAY),
      _market_ts(44, CURR_DAY),
      # Ideal group 4
      _market_ts(45, CURR_DAY),
      _market_ts(50, CURR_DAY),
      _market_ts(59, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106, 76]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0-14, 15-29, 30-44, 45-59
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=15, offset=0
    )

    expected_groups = [
      timestamps[0],
      timestamps[0],
      timestamps[0],
      timestamps[3],
      timestamps[3],
      timestamps[5],
      timestamps[5],
      timestamps[5],
    ]

    assert result.tolist() == expected_groups

  def test_remainder_smaller_case(self):
    """Test with group_size=60 where remainder < group_size/2 (last group gets bigger)."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(30, CURR_DAY),
      _market_ts(59, CURR_DAY),
      ####
      _market_ts(60, CURR_DAY),
      ####
      _market_ts(299, CURR_DAY),
      ####
      _market_ts(300, CURR_DAY),
      _market_ts(345, CURR_DAY),
      _market_ts(374, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106, 107]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0-59, 60-119, 120-179, 180-239, 240-299, 300-374 (300-359 & 360-374 get merged)
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=60, offset=0
    )

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

  def test_remainder_larger_case(self):
    """Test with group_size=100 where remainder >= group_size/2 (extra last group appears)."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(99, CURR_DAY),
      ####
      _market_ts(299, CURR_DAY),
      ####
      _market_ts(300, CURR_DAY),
      _market_ts(374, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0-99, 100-199, 200-299, 300-374
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=100, offset=0
    )

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

  def test_single_large_group(self):
    """Test with group_size=375 (single group for entire session)."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(135, CURR_DAY),
      _market_ts(285, CURR_DAY),
      _market_ts(374, CURR_DAY),
      _market_ts(0, CURR_DAY + dt.timedelta(days=3)),  # Different day starts
      _market_ts(374, CURR_DAY + dt.timedelta(days=3)),  #
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 101]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=375, offset=0
    )

    # All bars should belong to the same group (first timestamp)
    expected_groups = [timestamps[0]] * 4 + [timestamps[4]] * 2

    assert result.tolist() == expected_groups

  def test_cross_day_boundary(self):
    """Test that function works with timestamps from different days."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(29, CURR_DAY),
      ####
      _market_ts(60, CURR_DAY + dt.timedelta(days=2)),
      _market_ts(89, CURR_DAY + dt.timedelta(days=2)),
      ####
      _market_ts(179, CURR_DAY + dt.timedelta(days=2)),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0-29, 30-59, 60-89, 90-119, 120-149, 150-179,...
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=30, offset=0
    )

    expected_groups = [
      timestamps[0],  # Day 1, group 1
      timestamps[0],  # Day 1, group 1
      timestamps[2],  # Day 2, group 1
      timestamps[2],  # Day 2, group 1
      timestamps[4],  # Day 2, group 2
    ]

    assert result.tolist() == expected_groups

  def test_single_bar(self):
    """Test with single bar."""
    timestamps = [_market_ts(29, CURR_DAY)]

    df = pd.DataFrame(
      {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
    )

    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=15, offset=0
    )

    assert result.tolist() == [timestamps[0]]

  def test_non_zero_offset_basic(self):
    timestamps = [
      _market_ts(2, CURR_DAY),  # belongs to the previous day's last group
      ####
      _market_ts(3, CURR_DAY),
      _market_ts(13, CURR_DAY),
      _market_ts(17, CURR_DAY),
      ####
      _market_ts(18, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0,1,2 3-17, 18-32, ...
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=15, offset=3
    )

    expected_groups = [
      timestamps[0],
      timestamps[1],
      timestamps[1],
      timestamps[1],
      timestamps[4],
    ]
    assert result.tolist() == expected_groups

  def test_offset_cross_day(self):
    day1 = dt.date(2023, 1, 3)
    day2 = dt.date(2023, 1, 13)

    timestamps = [
      _market_ts(304, day1),
      _market_ts(305, day1),
      _market_ts(4, day2),
      _market_ts(5, day2),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    # 0,1,2,3,4 5-104, 105-204, 205-304, 305-374
    result = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size=100, offset=5
    )

    expected_groups = [
      timestamps[0],
      timestamps[1],
      timestamps[1],  # belongs to previous day's last group
      timestamps[3],
    ]

    assert result.tolist() == expected_groups

  def test_streaming_matches_batch(self):
    np.random.seed(42)
    df = u_t.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )
    df = df.sample(frac=0.3).sort_index()
    # check that test is meaningful
    # it should have more than 100 bars and more than 1 day
    assert len(df) > 100
    assert len(set(df.index.date)) > 1
    result = u_s.compare_batch_and_stream(
      df,
      lambda df_: bg_tb.get_time_based_bar_group_for_indian_market(
        df_,
        group_size=47,
        offset=11,
      ),
    )
    assert result.dfs_match

  def test_invalid_inputs(self):
    """Test error handling for invalid group_size."""
    timestamps = [pd.Timestamp("2023-01-02 03:45:59.999000+00:00")]
    df = pd.DataFrame(
      {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
    )

    # Test group_size <= 0
    with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
      bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=0, offset=0)

    with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
      bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=-1, offset=0)

    # Test group_size > 375
    with pytest.raises(ValueError, match="group_size must be between 1 and 375"):
      bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=376, offset=0)

    with pytest.raises(ValueError, match="offset must be between 0 and 74"):
      bg_tb.get_time_based_bar_group_for_indian_market(df, group_size=60, offset=75)
