import datetime as dt
from functools import partial
from typing import Callable, List, Tuple

import pandas as pd
import pytest

import algotrading_v40.bar_groupers.time_based_uniform as bg_tbu
import algotrading_v40.feature_calculators.adx as fc_adx
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut

# -----------------------------------------------------------------------------#
# Shared constants & parameterisation data
# -----------------------------------------------------------------------------#

LOOKBACK: int = 14
TEST_START_DATE: dt.date = dt.date(2023, 1, 2)
TEST_END_DATE: dt.date = dt.date(2023, 1, 3)

INTERVAL_GROUPS: List[Tuple[int, Callable[[pd.DataFrame], pd.Series]]] = [
  (
    1,
    partial(
      bg_tbu.get_time_based_uniform_bar_group_for_indian_market,
      group_size_minutes=1,
      offset_minutes=0,
    ),
  ),
  (
    11,
    partial(
      bg_tbu.get_time_based_uniform_bar_group_for_indian_market,
      group_size_minutes=11,
      offset_minutes=3,
    ),
  ),
]

# -----------------------------------------------------------------------------#
# Test suite
# -----------------------------------------------------------------------------#


class TestAdx:
  """Verify that the streaming ADX implementation produces the same results as the
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

    # Calculate ADX for a single group of bars
    def _calc_adx(bar_df: pd.DataFrame) -> pd.DataFrame:
      return fc_adx.adx(bar_df, lookback=LOOKBACK)

    # Add bar grouping *inside* the helper so the DataFrame passed to
    # compare_batch_and_stream remains OHLCV-only.
    def _group_and_calc(d: pd.DataFrame) -> pd.DataFrame:
      return udf.calculate_grouped_values(d, group_func(d).bar_groups, _calc_adx)

    # -------- Act -----------------------------------------------------------
    result = us.compare_batch_and_stream(df, _group_and_calc)

    # -------- Assert --------------------------------------------------------
    expected_col = f"adx_{LOOKBACK}"
    assert list(result.df_batch.columns) == [expected_col]
    assert result.df_batch.index.equals(df.index)

    quality = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert quality.n_good_values >= min(10, 375 / group_size_minutes)

    if group_size_minutes == 1:
      # For 1-minute bars we expect no NaNs at the end, and all NaNs clustered at the start
      assert quality.n_bad_values_at_end == 0
      assert quality.n_bad_values_at_start == quality.n_bad_values

    assert result.dfs_match, (
      "Batch and streaming ADX results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )

  def test_1_minute_group_same_as_no_grouping(self) -> None:
    df = ut.get_test_df(start_date=TEST_START_DATE, end_date=TEST_END_DATE)

    def _calc_adx(bar_df: pd.DataFrame) -> pd.DataFrame:
      return fc_adx.adx(bar_df, lookback=LOOKBACK)

    with ut.expect_no_mutation(df):
      res1 = fc_adx.adx(df, lookback=LOOKBACK)
    quality = udf.analyse_numeric_series_quality(res1[f"adx_{LOOKBACK}"])
    assert quality.n_good_values >= 375
    with ut.expect_no_mutation(df):
      res2 = udf.calculate_grouped_values(
        df,
        bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
          df, group_size_minutes=1, offset_minutes=0
        ).bar_groups,
        _calc_adx,
      )
    pd.testing.assert_series_equal(res1[f"adx_{LOOKBACK}"], res2[f"adx_{LOOKBACK}"])
