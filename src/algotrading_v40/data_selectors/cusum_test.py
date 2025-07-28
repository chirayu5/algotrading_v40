import numpy as np
import pandas as pd
import pytest

from algotrading_v40.data_selectors.cusum import validate_and_run_cusum


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
    h = 2.0

    expected = pd.Series([0, 1, -1, 1], index=index, dtype="int32")

    result = validate_and_run_cusum(s, h)
    pd.testing.assert_series_equal(result, expected)

  def test_threshold_equal_does_not_trigger(self) -> None:
    index = self._make_index(3)
    s = pd.Series([0.0, 2.0, 3.0], index=index)
    h = 2.0

    expected = pd.Series([0, 0, 1], index=index, dtype="int32")
    result = validate_and_run_cusum(s, h)
    pd.testing.assert_series_equal(result, expected)

  def test_bad_values_in_s_raises(self) -> None:
    index = self._make_index(3)
    s = pd.Series([np.nan, 1.0, 2.0], index=index)
    h = 1.0

    with pytest.raises(ValueError, match="s must not have bad values"):
      validate_and_run_cusum(s, h)

  def test_non_positive_threshold_raises(self) -> None:
    index = self._make_index(3)
    s = pd.Series([0.0, 1.0, 2.0], index=index)
    h = 0.0

    with pytest.raises(ValueError, match="cusum threshold must be greater than 0"):
      validate_and_run_cusum(s, h)

  def test_large_series(self) -> None:
    index = self._make_index(300)
    s = pd.Series(np.random.randint(1, 120, len(index)), index=index)
    h = np.random.randint(1, 10)
    n = len(index)
    expected = [0] * n
    for j in range(1, n):
      # check if expected[j] should be 1
      for i in range(j - 1, -1, -1):
        if s.iloc[i] > s.iloc[j]:
          break
        if s.iloc[j] - s.iloc[i] > h:
          expected[j] = 1
          break
        if expected[i] == 1:
          break

      # check if expected[j] should be -1
      for i in range(j - 1, -1, -1):
        if s.iloc[i] < s.iloc[j]:
          break
        if s.iloc[j] - s.iloc[i] < -h:
          expected[j] = -1
          break
        if expected[i] == -1:
          break

    expected = pd.Series(expected, index=index, dtype="int32")
    result = validate_and_run_cusum(s, h)
    pd.testing.assert_series_equal(result, expected)
