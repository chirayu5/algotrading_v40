import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.bar_groupers.time_based_uniform as bg_tbu
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
  border = pd.Timestamp.combine(
    day, ctnts.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC")
  result = base + pd.Timedelta(minutes=minutes_after_open)
  if result > border:
    raise ValueError(f"minutes_after_open {minutes_after_open} is too large")
  return result


class TestGetTimeBasedUniformBarGroupForIndianMarket:
  def test_basic_functionality(self):
    """Test basic uniform grouping with simple timestamps starting from the middle of the day (mod)."""
    shift = 13
    timestamps = [
      # first timestamp can be any valid indian market timestamp
      # it will become the origin for the grouping
      _market_ts(0 + shift, CURR_DAY),
      _market_ts(1 + shift, CURR_DAY),
      _market_ts(12 + shift, CURR_DAY),
      _market_ts(13 + shift, CURR_DAY),
      _market_ts(30 + shift, CURR_DAY),
      _market_ts(32 + shift, CURR_DAY),
      _market_ts(44 + shift, CURR_DAY),
      _market_ts(45 + shift, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106, 107]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    # With group_size_minutes=15: groups should be 0-14, 15-29, 30-44, 45-59, etc.
    result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df, group_size_minutes=15, offset_minutes=0
    )

    expected_groups = [
      timestamps[0],
      timestamps[0],
      timestamps[0],
      timestamps[0],
      timestamps[4],
      timestamps[4],
      timestamps[4],
      timestamps[7],
    ]

    assert result.bar_groups.tolist() == expected_groups
    assert result.offsets.tolist() == [0, 1, 12, 13, 0, 2, 14, 0]

  def test_cross_day_boundary(self):
    """Test that function correctly handles timestamps across multiple days."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(5, CURR_DAY),
      ##################
      _market_ts(6, CURR_DAY),
      _market_ts(11, CURR_DAY),
      ##################
      _market_ts(12, CURR_DAY),
      _market_ts(14, CURR_DAY),
      ##################
      _market_ts(372, CURR_DAY),
      # Next day
      # prev372 prev373 prev374 0 1 2| 3 - 8 | 9 - 14 | 15 - 20 | 21 - 26 | 27 - 32 | 33 - 38
      _market_ts(2, CURR_DAY + dt.timedelta(days=12)),
      ##################
      _market_ts(14, CURR_DAY + dt.timedelta(days=12)),
      ##################
      _market_ts(27, CURR_DAY + dt.timedelta(days=12)),  # A new group starts
      _market_ts(32, CURR_DAY + dt.timedelta(days=12)),
    ]

    df = pd.DataFrame(
      {"price": np.arange(len(timestamps))},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df, group_size_minutes=6, offset_minutes=0
    )

    expected_groups = [
      timestamps[0],
      timestamps[0],
      ##################
      timestamps[2],
      timestamps[2],
      ##################
      timestamps[4],
      timestamps[4],
      ##################
      timestamps[6],
      # Next day
      timestamps[6],
      ##################
      timestamps[8],
      ##################
      timestamps[9],
      timestamps[9],
    ]

    assert result.bar_groups.tolist() == expected_groups
    assert result.offsets.tolist() == [0, 5, 0, 5, 0, 2, 0, 5, 5, 0, 5]

  def test_cross_day_boundary_with_offset(self):
    """Test that function correctly handles timestamps across multiple days."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(1, CURR_DAY),
      ##################
      _market_ts(2, CURR_DAY),
      _market_ts(7, CURR_DAY),
      ##################
      _market_ts(8, CURR_DAY),
      _market_ts(13, CURR_DAY),
      ##################
      _market_ts(14, CURR_DAY),
      ##################
      _market_ts(372, CURR_DAY),
      ##################
      _market_ts(374, CURR_DAY),
      # Next day
      # prev374 0 1 2 3 4 | 5-10 | 11-16 | 17-22 | 23-28
      _market_ts(4, CURR_DAY + dt.timedelta(days=12)),
      ##################
      _market_ts(17, CURR_DAY + dt.timedelta(days=12)),
      _market_ts(22, CURR_DAY + dt.timedelta(days=12)),
      ##################
    ]

    df = pd.DataFrame(
      {"price": np.arange(len(timestamps))},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df, group_size_minutes=6, offset_minutes=2
    )

    expected_groups = [
      timestamps[0],
      timestamps[0],
      ##################
      timestamps[2],
      timestamps[2],
      ##################
      timestamps[4],
      timestamps[4],
      ##################
      timestamps[6],
      ##################
      timestamps[7],
      ##################
      timestamps[8],
      timestamps[8],
      #################
      timestamps[10],
      timestamps[10],
    ]

    assert result.bar_groups.tolist() == expected_groups
    assert result.offsets.tolist() == [4, 5, 0, 5, 0, 5, 0, 4, 0, 5, 0, 5]

  def test_consistency(self):
    """Test that changing offset parameter shifts which timestamps get offset 0 in the resulting series."""
    np.random.seed(42)
    df = u_t.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 2, 3),
    )
    df = df.sample(frac=0.3).sort_index()
    # Check that test is meaningful
    assert len(df) > 2000
    assert len(set(df.index.date)) > 20

    with u_t.expect_no_mutation(df):
      base_result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
        df, group_size_minutes=17, offset_minutes=0
      )

    for test_offset in range(17):
      timestamps_with_target_offset = df.index[base_result.offsets == test_offset]
      assert len(timestamps_with_target_offset) > 0
      with u_t.expect_no_mutation(df):
        shifted_result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
          df, group_size_minutes=17, offset_minutes=test_offset
        )
      timestamps_with_zero_offset = df.index[shifted_result.offsets == 0]
      assert timestamps_with_zero_offset.equals(timestamps_with_target_offset)

  def test_streaming_matches_batch(self):
    """Test that streaming and batch processing produce the same results."""
    np.random.seed(42)
    df = u_t.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )
    df = df.sample(frac=0.3).sort_index()

    # Check that test is meaningful
    assert len(df) > 100
    assert len(set(df.index.date)) > 1

    result = u_s.compare_batch_and_stream(
      df,
      lambda df_: bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
        df_,
        group_size_minutes=17,
        offset_minutes=4,
      ).bar_groups,
    )
    assert result.dfs_match

  def test_empty_and_single_bar_dataframe(self):
    """Test behavior with empty DataFrame."""
    df = pd.DataFrame(
      {"price": []},
      index=pd.DatetimeIndex([], name="bar_close_timestamp", tz="UTC"),
    )

    result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df, group_size_minutes=15, offset_minutes=0
    )

    assert len(result.bar_groups) == 0
    assert isinstance(result.bar_groups, pd.Series)

    df = pd.DataFrame(
      {"price": [100]},
      index=pd.DatetimeIndex(
        [pd.Timestamp("2023-01-02 09:15:00").tz_localize("UTC")],
        name="bar_close_timestamp",
      ),
    )
    result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df, group_size_minutes=15, offset_minutes=0
    )
    assert len(result.bar_groups) == 1
    assert isinstance(result.bar_groups, pd.Series)
    assert result.bar_groups.iloc[0] == pd.Timestamp("2023-01-02 09:15:00").tz_localize(
      "UTC"
    )

  def test_non_utc_timezone_raises_error(self):
    """Test that non-UTC timezone raises appropriate error."""
    timestamps = [pd.Timestamp("2023-01-02 09:15:00").tz_localize("US/Eastern")]
    df = pd.DataFrame(
      {"price": [100]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )
    with pytest.raises(ValueError):
      bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
        df, group_size_minutes=15, offset_minutes=0
      )
