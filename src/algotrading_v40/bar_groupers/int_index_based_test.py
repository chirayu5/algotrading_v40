import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.bar_groupers.int_index_based as bg_iib
import algotrading_v40.constants as ctnts
import algotrading_v40.utils.streaming as u_s
import algotrading_v40.utils.testing as u_t

CURR_DAY = dt.date(2025, 9, 15)
PREV_DAY = dt.date(2025, 9, 12)


def _market_ts(minutes_after_open: int, day: dt.date = CURR_DAY) -> pd.Timestamp:
  """
  Convenience: return a Timestamp <minutes_after_open> minutes after the
  first (minute-bar-close) timestamp of the trading session, in UTC.
  """
  base = pd.Timestamp.combine(
    day, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC")
  return base + pd.Timedelta(minutes=minutes_after_open)


class TestGetIntIndexBasedBarGroup:
  def test_basic_functionality_group_size_3_no_offset(self):
    """Test basic grouping with group_size=3 and offset=0."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(11, CURR_DAY),
      _market_ts(22, CURR_DAY),
      _market_ts(27, CURR_DAY),
      _market_ts(41, CURR_DAY),
      _market_ts(59, CURR_DAY),
      _market_ts(300, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    with u_t.expect_no_mutation(df):
      result = bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=0)

    # Groups: [0,1,2], [3,4,5], [6]
    expected_groups = [
      timestamps[0],  # index 0 -> group 0
      timestamps[0],  # index 1 -> group 0
      timestamps[0],  # index 2 -> group 0
      timestamps[3],  # index 3 -> group 1
      timestamps[3],  # index 4 -> group 1
      timestamps[3],  # index 5 -> group 1
      timestamps[6],  # index 6 -> group 2
    ]

    assert result.tolist() == expected_groups

  def test_with_offset_group_size_3_offset_1(self):
    """Test grouping with group_size=3 and offset=1."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(11, CURR_DAY),
      _market_ts(22, CURR_DAY),
      _market_ts(27, CURR_DAY),
      _market_ts(41, CURR_DAY),
      _market_ts(59, CURR_DAY),
      _market_ts(300, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104, 105, 106]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=1)

    # With offset=1, first bar gets group 0, then shift by 1
    # Groups: [0], [1,2,3], [4,5,6]
    expected_groups = [
      timestamps[0],  # index 0 -> group 0 (due to fillna(0))
      timestamps[1],  # index 1 -> group 1
      timestamps[1],  # index 2 -> group 1
      timestamps[1],  # index 3 -> group 1
      timestamps[4],  # index 4 -> group 2
      timestamps[4],  # index 5 -> group 2
      timestamps[4],  # index 6 -> group 2
    ]

    assert result.tolist() == expected_groups

  def test_with_offset_group_size_2_offset_1(self):
    """Test grouping with group_size=2 and offset=1."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(5, CURR_DAY),
      _market_ts(10, CURR_DAY),
      _market_ts(15, CURR_DAY),
      _market_ts(20, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_iib.get_int_index_based_bar_group(df, group_size=2, offset=1)

    # With offset=1, groups shift: [0], [1,2], [3,4]
    expected_groups = [
      timestamps[0],  # index 0 -> group 0 (due to fillna(0))
      timestamps[1],  # index 1 -> group 1
      timestamps[1],  # index 2 -> group 1
      timestamps[3],  # index 3 -> group 2
      timestamps[3],  # index 4 -> group 2
    ]

    assert result.tolist() == expected_groups

  def test_single_bar(self):
    """Test with single bar."""
    timestamps = [_market_ts(29, CURR_DAY)]
    df = pd.DataFrame(
      {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
    )
    result = bg_iib.get_int_index_based_bar_group(df, group_size=5, offset=0)
    assert result.tolist() == [timestamps[0]]

    timestamps = [_market_ts(29, CURR_DAY)]
    df = pd.DataFrame(
      {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
    )
    result = bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=2)
    assert result.tolist() == [timestamps[0]]

  def test_group_size_larger_than_dataframe_length(self):
    """Test when group_size is larger than the number of bars."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(5, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    result = bg_iib.get_int_index_based_bar_group(df, group_size=5, offset=0)

    # All bars should be in the same group
    expected_groups = [timestamps[0], timestamps[0]]
    assert result.tolist() == expected_groups

  def test_max_offset_value(self):
    """Test with maximum allowed offset value."""
    timestamps = [
      _market_ts(0, CURR_DAY),
      _market_ts(13, CURR_DAY),
      _market_ts(26, CURR_DAY),
      _market_ts(29, CURR_DAY),
      _market_ts(200, CURR_DAY),
    ]

    df = pd.DataFrame(
      {"price": [100, 101, 102, 103, 104]},
      index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
    )

    # Test with offset = group_size - 1 (maximum allowed)
    result = bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=2)

    # With offset=2, first 2 bars get group 0, then shift by 2
    expected_groups = [
      timestamps[0],  # index 0 -> group 0 (due to fillna(0))
      timestamps[0],  # index 1 -> group 0 (due to fillna(0))
      timestamps[2],  # index 2 -> group 1
      timestamps[2],  # index 3 -> group 1
      timestamps[2],  # index 4 -> group 1
    ]

    assert result.tolist() == expected_groups

  def test_streaming_matches_batch(self):
    """Test that streaming results match batch results."""
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
      lambda df_: bg_iib.get_int_index_based_bar_group(
        df_,
        group_size=7,
        offset=3,
      ),
    )
    assert result.dfs_match

  def test_invalid_inputs(self):
    """Test error handling for invalid inputs."""
    timestamps = [pd.Timestamp("2023-01-02 03:45:59.999000+00:00")]
    df = pd.DataFrame(
      {"price": [100]}, index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp")
    )

    # Test negative offset
    with pytest.raises(ValueError, match="offset must be between 0 and"):
      bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=-1)

    # Test offset >= group_size
    with pytest.raises(ValueError, match="offset must be between 0 and 2"):
      bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=3)

    with pytest.raises(ValueError, match="offset must be between 0 and 4"):
      bg_iib.get_int_index_based_bar_group(df, group_size=5, offset=5)

  def test_empty_dataframe(self):
    """Test with empty DataFrame."""
    df = pd.DataFrame(
      {"price": []}, index=pd.DatetimeIndex([], name="bar_close_timestamp", tz="UTC")
    )

    result = bg_iib.get_int_index_based_bar_group(df, group_size=3, offset=0)

    assert len(result) == 0
    assert isinstance(result, pd.Series)
