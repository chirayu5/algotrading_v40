import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.data_selectors.cusum as dsc
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


class TestValidateAndRunCusum:
  @staticmethod
  def _make_index(n: int) -> pd.DatetimeIndex:
    """Utility to build the required UTC index."""
    return pd.date_range(
      start="2023-01-02 03:45:59.999000+00:00",
      periods=n,
      freq="1min",
      tz="UTC",
      name="date",
    )

  def test_basic_positive_and_negative_events(self) -> None:
    index = self._make_index(4)
    s = pd.Series([0.0, 3.0, 0.0, 3.0], index=index, name="price")
    thresholds = pd.Series([2.0] * len(index), index=index)

    expected = pd.Series([0, 1, -1, 1], index=index, dtype="int32").rename("selected")
    with ut.expect_no_mutation(s):
      result = dsc.cusum(s=s, thresholds=thresholds)
    pd.testing.assert_series_equal(result["selected"], expected)

  def test_threshold_equal_does_not_trigger(self) -> None:
    index = self._make_index(3)
    s = pd.Series([0.0, 2.0, 3.0], index=index)
    thresholds = pd.Series([2.0] * len(index), index=index)

    expected = pd.Series([0, 0, 1], index=index, dtype="int32").rename("selected")
    with ut.expect_no_mutation(s):
      result = dsc.cusum(s=s, thresholds=thresholds)
    pd.testing.assert_series_equal(result["selected"], expected)

  def test_bad_values_in_s_raises(self) -> None:
    index = self._make_index(3)
    s = pd.Series([np.nan, 1.0, 2.0], index=index)
    thresholds = pd.Series([1.0] * len(index), index=index)

    with pytest.raises(ValueError, match="s must not have bad values"):
      with ut.expect_no_mutation(s):
        dsc.cusum(s=s, thresholds=thresholds)

  def test_non_positive_threshold_raises(self) -> None:
    index = self._make_index(3)
    s = pd.Series([0.0, 1.0, 2.0], index=index)
    thresholds = pd.Series([0.0] * len(index), index=index)

    with pytest.raises(ValueError, match="cusum threshold must be greater than 0"):
      with ut.expect_no_mutation(s):
        dsc.cusum(s=s, thresholds=thresholds)

  def test_stream_vs_batch(self) -> None:
    np.random.seed(42)
    df = ut.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 2),
    )

    def inner_(df_: pd.DataFrame) -> pd.DataFrame:
      return dsc.cusum(
        s=np.log(df_["open"]),
        thresholds=pd.Series([0.005] * len(df_["open"]), index=df_["open"].index),
      )

    result = us.compare_batch_and_stream(
      df,
      inner_,
    )
    assert result.dfs_match
    # check the test is testing a significant number of events
    assert result.df_batch["selected"].value_counts(dropna=False)[0] >= 20
    assert result.df_batch["selected"].value_counts(dropna=False)[1] >= 20
    assert result.df_batch["selected"].value_counts(dropna=False)[-1] >= 20
    assert result.df_batch["selected"].abs().sum() / len(result.df_batch) >= 0.01

  def test_variable_threshold(self) -> None:
    index = self._make_index(10)
    s = pd.Series([90, 95, 108, 99, 102, 98, 24, 106, 7, 72], index=index)
    thresholds = pd.Series([3, 4, 7, 9, 3, 6, 9, 3, 7, 7], index=index)

    with ut.expect_no_mutation(s):
      result = dsc.cusum(s=s, thresholds=thresholds)

    expected = pd.Series(
      [0, 1, 1, 0, -1, 0, -1, 1, -1, 1], index=index, dtype="int32"
    ).rename("selected")
    pd.testing.assert_series_equal(result["selected"], expected)

  def test_large_series_matches_On2_algorithm(self) -> None:
    index = self._make_index(300)
    s = pd.Series(np.random.randint(1, 120, len(index)), index=index)
    thresholds = pd.Series(np.random.randint(1, 40, len(index)), index=index)

    n = len(index)
    expected = [0] * n
    for j in range(1, n):
      # check if expected[j] should be 1
      for i in range(j - 1, -1, -1):
        if s.iloc[j] - s.iloc[i] > thresholds.iloc[j]:
          expected[j] = 1
          break
        if expected[i] == 1:
          break

      # check if expected[j] should be -1
      for i in range(j - 1, -1, -1):
        if s.iloc[j] - s.iloc[i] < -thresholds.iloc[j]:
          expected[j] = -1
          break
        if expected[i] == -1:
          break

    expected = pd.Series(expected, index=index, dtype="int32").rename("selected")
    with ut.expect_no_mutation(s):
      result = dsc.cusum(s=s, thresholds=thresholds)
    pd.testing.assert_series_equal(result["selected"], expected)

  def test_empty_series(self) -> None:
    index = self._make_index(4)
    s = pd.Series([0.0, 3.0, 0.0, 3.0], index=index, name="price")
    thresholds = pd.Series([2.0] * len(index), index=index)
    # actual data does not matter as we do [:0] on the series to get an empty series
    expected = pd.Series(s[:0], index=index[:0], dtype="int32").rename("selected")
    with ut.expect_no_mutation(s):
      result = dsc.cusum(s=s.iloc[:0], thresholds=thresholds.iloc[:0])
    pd.testing.assert_series_equal(result["selected"], expected)

  def test_index_mismatch_raises(self) -> None:
    index = self._make_index(4)
    s = pd.Series([0.0, 3.0, 0.0, 3.0], index=index)
    thresholds = pd.Series([2.0] * len(index), index=self._make_index(8)[4:])

    with pytest.raises(
      ValueError, match="s.index and thresholds.index must be the same"
    ):
      with ut.expect_no_mutation(s):
        dsc.cusum(s=s, thresholds=thresholds)
