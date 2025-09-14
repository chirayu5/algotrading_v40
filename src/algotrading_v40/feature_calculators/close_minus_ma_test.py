import datetime as dt
from functools import partial
from typing import Callable, List, Tuple

import pandas as pd
import pytest

import algotrading_v40.bar_groupers.time_based as bg_tb
import algotrading_v40.feature_calculators.close_minus_ma as fc_close_minus_ma
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut

# -----------------------------------------------------------------------------#
# Shared constants & parameterisation data
# -----------------------------------------------------------------------------#

LOOKBACK: int = 10
ATR_LENGTH: int = 25
# ATR_LENGTH in the book is 252 bars so ideally the test should use that.
# For 252 bars we would need to use a much larger test dataset which slows down the tests.
# So we use 25 bars.
TEST_START_DATE: dt.date = dt.date(2023, 1, 2)
TEST_END_DATE: dt.date = dt.date(2023, 1, 3)

INTERVAL_GROUPS: List[Tuple[int, Callable[[pd.DataFrame], pd.Series]]] = [
  (
    1,
    partial(
      bg_tb.get_time_based_bar_group_for_indian_market,
      group_size_minutes=1,
      offset_minutes=0,
    ),
  ),
  (
    11,
    partial(
      bg_tb.get_time_based_bar_group_for_indian_market,
      group_size_minutes=11,
      offset_minutes=3,
    ),
  ),
]

# -----------------------------------------------------------------------------#
# Test suite
# -----------------------------------------------------------------------------#


class TestCloseMma:
  """Verify that the streaming close minus MA implementation produces the same results as the
  batch version across different bar sizes (1-minute and 11-minute)."""

  @pytest.mark.parametrize("group_size_minutes, group_func", INTERVAL_GROUPS)
  def test_streaming_matches_batch_with_grouping(
    self,
    group_size_minutes: int,
    group_func: Callable[[pd.DataFrame], pd.Series],
  ) -> None:
    # -------- Arrange --------------------------------------------------------
    df = ut.get_test_df(start_date=TEST_START_DATE, end_date=TEST_END_DATE)
    df = df.sample(frac=1 - 0.2).sort_index()

    # Calculate close minus MA for a single group of bars
    def _calc_close_minus_ma(bar_df: pd.DataFrame) -> pd.DataFrame:
      return fc_close_minus_ma.close_minus_ma(
        bar_df, lookback=LOOKBACK, atr_length=ATR_LENGTH
      )

    # Add bar grouping *inside* the helper so the DataFrame passed to
    # compare_batch_and_stream remains OHLCV-only.
    def _group_and_calc(d: pd.DataFrame) -> pd.DataFrame:
      d["bar_group"] = group_func(d)
      return udf.calculate_grouped_values(d, _calc_close_minus_ma)

    # -------- Act -----------------------------------------------------------
    result = us.compare_batch_and_stream(df, _group_and_calc)

    # -------- Assert --------------------------------------------------------
    expected_col = f"close_minus_ma_{LOOKBACK}_{ATR_LENGTH}"
    assert list(result.df_batch.columns) == [expected_col]
    assert result.df_batch.index.equals(df.index)

    quality = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert quality.n_good_values >= min(10, 375 / group_size_minutes)

    if group_size_minutes == 1:
      # For 1-minute bars we expect no NaNs at the end, and all NaNs clustered at the start
      assert quality.n_bad_values_at_end == 0
      assert quality.n_bad_values_at_start == quality.n_bad_values

    assert result.dfs_match, (
      "Batch and streaming close minus MA results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )

  def test_1_minute_group_same_as_no_grouping(self) -> None:
    df = ut.get_test_df(start_date=TEST_START_DATE, end_date=TEST_END_DATE)

    def _calc_close_minus_ma(bar_df: pd.DataFrame) -> pd.DataFrame:
      return fc_close_minus_ma.close_minus_ma(
        bar_df, lookback=LOOKBACK, atr_length=ATR_LENGTH
      )

    with ut.expect_no_mutation(df):
      res1 = fc_close_minus_ma.close_minus_ma(
        df, lookback=LOOKBACK, atr_length=ATR_LENGTH
      )
    quality = udf.analyse_numeric_series_quality(
      res1[f"close_minus_ma_{LOOKBACK}_{ATR_LENGTH}"]
    )
    assert quality.n_good_values >= 375
    df["bar_group"] = bg_tb.get_time_based_bar_group_for_indian_market(
      df, group_size_minutes=1, offset_minutes=0
    )
    with ut.expect_no_mutation(df):
      res2 = udf.calculate_grouped_values(df, _calc_close_minus_ma)
    pd.testing.assert_series_equal(
      res1[f"close_minus_ma_{LOOKBACK}_{ATR_LENGTH}"],
      res2[f"close_minus_ma_{LOOKBACK}_{ATR_LENGTH}"],
    )
