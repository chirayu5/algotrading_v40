import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.constants as ctnts
import algotrading_v40.trading_time_elapsed_calculators.with_overnight_gaps_only as ttec_wogo
import algotrading_v40.utils.streaming as u_s
import algotrading_v40.utils.testing as u_t

CURR_DAY = dt.date(2025, 9, 15)
NEXT_DAY = CURR_DAY + dt.timedelta(days=2)


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


class TestWithOvernightGapsOnly:
  def test_intraday_only(self):
    """No overnight gap ⇒ cumulative minutes 0,1,2."""
    idx = pd.DatetimeIndex(
      [_market_ts(i) for i in (3, 11, 22)], name="bar_close_timestamp"
    )
    out = ttec_wogo.with_overnight_gaps_only(idx, overnight_gap_minutes=60)
    assert out.tolist() == [0, 8, 19]

  def test_single_overnight_gap(self):
    """Cross midnight once; subtract overnight_gap_minutes exactly once."""
    idx = pd.DatetimeIndex(
      [
        _market_ts(3, CURR_DAY),
        _market_ts(7, CURR_DAY),
        _market_ts(8, NEXT_DAY),
      ],
      name="bar_close_timestamp",
    )
    out = ttec_wogo.with_overnight_gaps_only(idx, overnight_gap_minutes=1065)
    assert out.tolist() == [0, 4, 375 - 3 + 8]

  def test_edge_cases(self):
    idx_empty = pd.DatetimeIndex([], name="bar_close_timestamp", tz="UTC")
    idx_single = pd.DatetimeIndex([_market_ts(0)], name="bar_close_timestamp")
    pd.testing.assert_series_equal(
      ttec_wogo.with_overnight_gaps_only(idx_empty, 30),
      pd.Series(index=idx_empty),
    )
    pd.testing.assert_series_equal(
      ttec_wogo.with_overnight_gaps_only(idx_single, 30),
      pd.Series([0], index=idx_single),
    )

  def test_validation_errors(self):
    idx_bad_tz = pd.date_range("2025-09-15", periods=1, freq="1min", tz="US/Eastern")
    with pytest.raises(ValueError):
      ttec_wogo.with_overnight_gaps_only(idx_bad_tz, 30)
    idx_good = pd.date_range("2025-09-15", periods=1, freq="1min", tz="UTC")
    with pytest.raises(ValueError):
      ttec_wogo.with_overnight_gaps_only(idx_good, -1)

    # zero overnight gap is valid
    _ = ttec_wogo.with_overnight_gaps_only(idx_good, 0)

  def test_streaming_matches_batch(self):
    """Ensure streaming and batch outputs are identical."""
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
      lambda df_: ttec_wogo.with_overnight_gaps_only(
        index=df_.index,
        overnight_gap_minutes=1065,
      ),
    )
    assert result.dfs_match
