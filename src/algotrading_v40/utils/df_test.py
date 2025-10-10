import datetime as dt
from datetime import date

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.structures.date_range as sdr
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.testing as ut


class TestGetDfSliceInDateRange:
  def test_basic(self):
    df = pd.DataFrame(
      {
        "open": [174.3, 174.3, 174.3, 174.2, 172.6, 3],
        "high": [174.30, 174.71, 174.30, 174.20, 172.60, 5],
        "low": [174.3, 174.3, 174.3, 172.6, 172.0, 2],
        "close": [174.3, 174.3, 174.3, 172.6, 172.6, 4],
        "volume": [1080, 220, 0, 2280, 1940, 10],
      },
      index=pd.DatetimeIndex(
        [
          "2016-01-03 03:52:59.999000+00:00",
          "2016-01-04 03:45:59.999000+00:00",
          "2016-01-04 03:46:59.999000+00:00",
          "2016-01-05 03:47:59.999000+00:00",
          "2016-01-05 03:48:59.999000+00:00",
          "2016-01-06 03:49:59.999000+00:00",
        ],
        name="date",
      ),
    )

    date_range = sdr.DateRange(date(2016, 1, 4), date(2016, 1, 5))
    dfb = df.copy()
    result = udf.get_df_slice_in_date_range(df, date_range)
    pd.testing.assert_frame_equal(dfb, df)

    expected_index = pd.DatetimeIndex(
      [
        "2016-01-04 03:45:59.999000+00:00",
        "2016-01-04 03:46:59.999000+00:00",
        "2016-01-05 03:47:59.999000+00:00",
        "2016-01-05 03:48:59.999000+00:00",
      ],
      name="date",
    )

    assert result.index.equals(expected_index)

  def test_single_day(self):
    df = pd.DataFrame(
      {
        "open": [174.3, 174.3, 174.3],
        "high": [174.30, 174.71, 174.30],
        "low": [174.3, 174.3, 174.3],
        "close": [174.3, 174.3, 174.3],
        "volume": [1080, 220, 0],
      },
      index=pd.DatetimeIndex(
        [
          "2016-01-04 03:45:59.999000+00:00",
          "2016-01-05 03:46:59.999000+00:00",
          "2016-01-06 03:47:59.999000+00:00",
        ],
        name="date",
      ),
    )

    date_range = sdr.DateRange(date(2016, 1, 5), date(2016, 1, 5))
    dfb = df.copy()
    result = udf.get_df_slice_in_date_range(df, date_range)
    pd.testing.assert_frame_equal(dfb, df)

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2016-01-05 03:46:59.999000+00:00")

  def test_empty_result(self):
    df = pd.DataFrame(
      {
        "open": [174.3, 174.3],
        "high": [174.30, 174.71],
        "low": [174.3, 174.3],
        "close": [174.3, 174.3],
        "volume": [1080, 220],
      },
      index=pd.DatetimeIndex(
        ["2016-01-04 03:45:59.999000+00:00", "2016-01-05 03:46:59.999000+00:00"],
        name="date",
      ),
    )

    date_range = sdr.DateRange(date(2016, 1, 10), date(2016, 1, 15))
    dfb = df.copy()
    result = udf.get_df_slice_in_date_range(df, date_range)
    pd.testing.assert_frame_equal(dfb, df)
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.DatetimeIndex)

  def test_invalid_index(self):
    df = pd.DataFrame(
      {
        "open": [174.3, 174.3],
        "high": [174.30, 174.71],
        "low": [174.3, 174.3],
        "close": [174.3, 174.3],
        "volume": [1080, 220],
      },
      index=[0, 1],
    )

    date_range = sdr.DateRange(date(2016, 1, 4), date(2016, 1, 5))
    dfb = df.copy()
    with pytest.raises(ValueError, match="df.index must be a DatetimeIndex"):
      udf.get_df_slice_in_date_range(df, date_range)
    pd.testing.assert_frame_equal(dfb, df)

  def test_non_utc_timezone_fails(self):
    df = pd.DataFrame(
      {
        "open": [174.3, 174.3, 174.3],
        "high": [174.30, 174.71, 174.30],
        "low": [174.3, 174.3, 174.3],
        "close": [174.3, 174.3, 174.3],
        "volume": [1080, 220, 0],
      },
      index=pd.DatetimeIndex(
        [
          "2016-01-04 03:45:59.999000+00:00",
          "2016-01-05 03:46:59.999000+00:00",
          "2016-01-06 03:47:59.999000+00:00",
        ],
        name="date",
        tz="Asia/Kolkata",
      ),
    )
    dfb = df.copy()
    date_range = sdr.DateRange(date(2016, 1, 5), date(2016, 1, 5))
    with pytest.raises(ValueError, match="DataFrame index must have UTC timezone"):
      udf.get_df_slice_in_date_range(df, date_range)
    pd.testing.assert_frame_equal(dfb, df)


class TestAnalyseNumericSeriesQuality:
  def test_normal_series_no_issues(self):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 5
    assert result.good_values_mask.equals(pd.Series([True, True, True, True, True]))
    assert result.n_values == 5
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_series_with_zeros_different_representations(self):
    s = pd.Series([1.0, 0.0, 0, -0.0, -0, 2.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 4  # 0.0, 0, -0.0, -0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 6
    assert result.good_values_mask.equals(
      pd.Series([True, True, True, True, True, True])
    )
    assert result.n_values == 6
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_series_with_negatives(self):
    s = pd.Series([1.0, -2.0, 3.0, -4.5, -0, -0.0, -np.inf, -np.nan, 5.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 2
    assert (
      result.n_negatives == 2
    )  # -0, -0.0, -np.inf and -np.nan are not counted as negatives
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 7
    assert result.good_values_mask.equals(
      pd.Series([True, True, True, True, True, True, False, False, True])
    )
    assert result.n_values == 9
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_nan_values(self):
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([True, False, True, False, True]))
    assert result.n_values == 5
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_inf_values(self):
    s = pd.Series([1.0, np.inf, 3.0, -np.inf, 5.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([True, False, True, False, True]))
    assert result.n_values == 5
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_mixed_bad_values(self):
    s = pd.Series([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0, None])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4  # nan, inf, -inf, None
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 1
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(
      pd.Series([True, False, True, False, False, True, False])
    )
    assert result.n_values == 7
    assert s.loc[result.good_values_mask].equals(
      pd.Series([1.0, 3.0, 5.0], index=[0, 2, 5])
    )
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_bad_values_at_start(self):
    s = pd.Series([np.nan, np.inf, 1.0, 2.0, 3.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([False, False, True, True, True]))
    assert result.n_values == 5
    assert result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_bad_values_at_end(self):
    s = pd.Series([1.0, 2.0, 3.0, np.nan, np.inf])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 2
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([True, True, True, False, False]))
    assert result.n_values == 5
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_series_with_bad_values_at_both_ends(self):
    s = pd.Series([np.nan, np.inf, 1.0, 2.0, 3.0, -np.inf, np.nan])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 2
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(
      pd.Series([False, False, True, True, True, False, False])
    )
    assert result.n_values == 7
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_series_with_bad_values_in_middle(self):
    s = pd.Series([1.0, 2.0, np.nan, np.inf, 3.0, 4.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 4
    assert result.good_values_mask.equals(
      pd.Series([True, True, False, False, True, True])
    )
    assert result.n_values == 6
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_all_bad_values(self):
    s = pd.Series([np.nan, np.inf, -np.inf, np.nan])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 4
    assert result.n_bad_values_at_end == 4
    assert result.n_good_values == 0
    assert result.good_values_mask.equals(pd.Series([False, False, False, False]))
    assert result.n_values == 4
    assert result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_empty_series(self):
    s = pd.Series([], dtype=float)
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 0
    assert result.good_values_mask.equals(pd.Series([], dtype=bool))
    assert result.n_values == 0
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_single_good_value(self):
    s = pd.Series([42.5])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 1
    assert result.good_values_mask.equals(pd.Series([True]))
    assert result.n_values == 1
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_single_bad_value(self):
    s = pd.Series([np.nan])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 1
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 1
    assert result.n_bad_values_at_end == 1
    assert result.n_good_values == 0
    assert result.good_values_mask.equals(pd.Series([False]))
    assert result.n_values == 1
    assert result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_single_zero_value(self):
    s = pd.Series([0.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 1
    assert result.good_values_mask.equals(pd.Series([True]))
    assert result.n_values == 1
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_single_negative_value(self):
    s = pd.Series([-5.0])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 1
    assert result.good_values_mask.equals(pd.Series([True]))
    assert result.n_values == 1
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_complex_mixed_case_with_result_dtypes(self):
    # Test with bad values at start/end, zeros, negatives, and good values
    s = pd.Series([np.nan, -np.inf, -2.0, 0.0, 1.0, -3.5, np.inf, np.nan])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4  # 2 nan, 2 inf
    assert result.n_zeros == 1
    assert (
      result.n_negatives == 2
    )  # -2.0, -3.5 (-inf values don't count as they're bad)
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 2
    assert result.n_good_values == 4
    assert result.good_values_mask.equals(
      pd.Series([False, False, True, True, True, True, False, False])
    )
    assert result.n_values == 8
    assert s.loc[result.good_values_mask].equals(
      pd.Series([-2.0, 0.0, 1.0, -3.5], index=[2, 3, 4, 5])
    )

    assert isinstance(result.n_bad_values, int)
    assert isinstance(result.n_zeros, int)
    assert isinstance(result.n_negatives, int)
    assert isinstance(result.n_bad_values_at_start, int)
    assert isinstance(result.n_bad_values_at_end, int)

    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end

  def test_integer_series(self):
    s = pd.Series([1, 2, -3, 0, 5], dtype=int)
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 5
    assert result.good_values_mask.equals(pd.Series([True, True, True, True, True]))
    assert result.n_values == 5
    assert not result.has_bad_values
    assert not result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_float32_series(self):
    s = pd.Series([1.0, -2.0, 0.0, np.nan], dtype=np.float32)
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 1
    assert result.n_zeros == 1
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 1
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([True, True, True, False]))
    assert result.n_values == 4
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert not result.has_bad_values_apart_from_end

  def test_non_numeric_series_raises_error(self):
    s = pd.Series(["a", "b", "c"])
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_object_series_with_numbers_raises_error(self):
    s = pd.Series([1, 2, "3"], dtype=object)
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_datetime_series_raises_error(self):
    s = pd.Series(pd.date_range("2020-01-01", periods=3))
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_string_series_raises_error(self):
    s = pd.Series(["1.0", "2.0", "3.0"])
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_boolean_series_works(self):
    s = pd.Series([True, False, True])
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1  # False
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 3
    assert result.good_values_mask.equals(pd.Series([True, True, True]))
    assert result.n_values == 3

  def test_pd_null_na_None_series_is_not_numeric(self):
    with pytest.raises(
      TypeError,
      match="float\\(\\) argument must be a string or a real number, not 'NAType'",
    ):
      # A pd.NA cannot be converted to a float32, so it raises a TypeError
      _ = pd.Series([1.0, None, 3.0, pd.NA, 5.0], dtype=np.float32)

    s = pd.Series([1.0, 2.0, 3.0, pd.NA])
    sb = s.copy()
    # If we try to analyse a series with pd.NA, it raises a ValueError
    with pytest.raises(ValueError, match="Series must be numeric"):
      udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_series_with_datetime_index(self):
    s = pd.Series([1.0, -np.inf, 3.0], index=pd.date_range("2020-01-01", periods=3))
    sb = s.copy()
    result = udf.analyse_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 1
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0
    assert result.n_good_values == 2
    assert result.good_values_mask.equals(pd.Series([True, False, True], index=s.index))
    assert result.n_values == 3
    assert s.loc[result.good_values_mask].equals(
      pd.Series(
        [1.0, 3.0], index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
      )
    )
    assert result.has_bad_values
    assert result.has_bad_values_apart_from_start
    assert result.has_bad_values_apart_from_end


class TestValidators:
  """Tests for data quality validator functions."""

  class TestCheckIndicesMatch:
    def test_matching_indices_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s1 = pd.Series([1, 2, 3], index=idx)
      s2 = pd.Series([4, 5, 6], index=idx)
      udf.check_indices_match(s1, s2)

    def test_mismatched_indices_fails(self):
      idx1 = pd.date_range("2020-01-01", periods=3, tz="UTC")
      idx2 = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04"], tz="UTC")
      s1 = pd.Series([1, 2, 3], index=idx1)
      s2 = pd.Series([4, 5, 6], index=idx2)
      with pytest.raises(ValueError, match="Series at position 1"):
        udf.check_indices_match(s1, s2)

  class TestCheckIndexUAndMi:
    def test_valid_index_passes(self):
      idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
      udf.check_index_u_and_mi(idx)

    def test_non_unique_fails(self):
      idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"], tz="UTC"
      )
      with pytest.raises(ValueError, match="Index must be unique"):
        udf.check_index_u_and_mi(idx)

    def test_non_monotonic_fails(self):
      idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"], tz="UTC"
      )
      with pytest.raises(ValueError, match="Index must be monotonically increasing"):
        udf.check_index_u_and_mi(idx)

  class TestCheckNoBadValues:
    def test_clean_series_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, 2.0, 3.0], index=idx)
      udf.check_no_bad_values(s)

    def test_series_with_nan_fails(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, np.nan, 3.0], index=idx)
      with pytest.raises(ValueError, match="bad values"):
        udf.check_no_bad_values(s)

  class TestCheckNoBadValuesApartFromStart:
    def test_bad_values_at_start_only_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([np.nan, 1.0, 2.0], index=idx)
      udf.check_no_bad_values_apart_from_start(s)

    def test_bad_values_in_middle_fails(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, np.nan, 3.0], index=idx)
      with pytest.raises(ValueError, match="apart from the start"):
        udf.check_no_bad_values_apart_from_start(s)

  class TestCheckAllGt0:
    def test_all_positive_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, 2.0, 3.0], index=idx)
      udf.check_all_gt0(s)

    def test_zero_value_fails(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, 0.0, 3.0], index=idx)
      with pytest.raises(ValueError, match="values that are <= 0"):
        udf.check_all_gt0(s)

  class TestCheckAllGte0:
    def test_non_negative_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([0.0, 1.0, 2.0], index=idx)
      udf.check_all_gte0(s)

    def test_negative_value_fails(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, -1.0, 3.0], index=idx)
      with pytest.raises(ValueError, match="values that are < 0"):
        udf.check_all_gte0(s)

  class TestCheckAllLt0:
    def test_all_negative_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([-1.0, -2.0, -3.0], index=idx)
      udf.check_all_lt0(s)

    def test_positive_value_fails(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([-1.0, 1.0, -3.0], index=idx)
      with pytest.raises(ValueError, match="values that are >= 0"):
        udf.check_all_lt0(s)
      del s

      s = pd.Series([-1.0, 0.0, -3.0], index=idx)
      with pytest.raises(ValueError, match="values that are >= 0"):
        udf.check_all_lt0(s)

  class TestCheckAllIn:
    def test_all_values_in_set_passes(self):
      idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
      s = pd.Series([1.0, 2.0, 3.0], index=idx)
      udf.check_all_in(s, (1.0, 2.0, 3.0))

    def test_value_not_in_set_fails(self):
      idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
      s = pd.Series([1.0, 4.0], index=idx)
      with pytest.raises(ValueError, match="values not in"):
        udf.check_all_in(s, (1.0, 2.0, 3.0))


class TestAnalyseNumericColumnsQuality:
  def test_empty_dataframe(self):
    df = pd.DataFrame()
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert result == {}

  def test_no_numeric_columns(self):
    df = pd.DataFrame(
      {"str_col": ["a", "b", "c"], "date_col": pd.date_range("2020-01-01", periods=3)}
    )
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert result == {}

  def test_single_numeric_column(self):
    df = pd.DataFrame({"numeric_col": [1.0, 2.0, 3.0], "str_col": ["a", "b", "c"]})
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result) == 1
    assert "numeric_col" in result
    assert result["numeric_col"].n_bad_values == 0
    assert result["numeric_col"].n_zeros == 0
    assert result["numeric_col"].n_negatives == 0

  def test_multiple_numeric_columns(self):
    df = pd.DataFrame(
      {"col1": [1.0, 0.0, -1.0], "col2": [np.inf, 2.0, 3.0], "str_col": ["a", "b", "c"]}
    )
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result) == 2
    assert "col1" in result
    assert "col2" in result
    assert result["col1"].n_zeros == 1
    assert result["col1"].n_negatives == 1
    assert result["col2"].n_bad_values == 1

  def test_mixed_column_types(self):
    df = pd.DataFrame(
      {
        "int_col": [1, 2, 3],
        "float_col": [1.0, 2.0, np.nan],
        "bool_col": [False, False, True],
        "str_col": ["a", "b", "c"],
        "date_col": pd.date_range("2020-01-01", periods=3),
      }
    )
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result) == 3
    assert "int_col" in result
    assert "float_col" in result
    assert "bool_col" in result
    assert "str_col" not in result
    assert "date_col" not in result
    assert result["float_col"].n_bad_values == 1
    assert result["float_col"].n_bad_values_at_end == 1
    assert result["bool_col"].n_zeros == 2

  def test_all_numeric_columns_with_issues(self):
    df = pd.DataFrame(
      {
        "col1": [np.inf, 0.0, -1.0],
        "col2": [1.0, np.nan, 3.0],
        "col3": [2.0, 0.0, -np.inf],
        "col4": [np.nan, None, np.nan],
      }
    )
    dfb = df.copy()
    result = udf.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert result.keys() == {"col1", "col2", "col3", "col4"}
    result["col1"].n_bad_values == 1
    result["col1"].n_zeros == 1
    result["col1"].n_negatives == 1
    result["col1"].n_bad_values_at_start == 1
    result["col1"].n_bad_values_at_end == 0

    result["col2"].n_bad_values == 1
    result["col2"].n_zeros == 0
    result["col2"].n_negatives == 0
    result["col2"].n_bad_values_at_start == 0
    result["col2"].n_bad_values_at_end == 0

    result["col3"].n_bad_values == 1
    result["col3"].n_zeros == 0
    result["col3"].n_negatives == 0
    result["col3"].n_bad_values_at_start == 0
    result["col3"].n_bad_values_at_end == 1

    result["col4"].n_bad_values == 3
    result["col4"].n_zeros == 0
    result["col4"].n_negatives == 0
    result["col4"].n_bad_values_at_start == 3
    result["col4"].n_bad_values_at_end == 3


class TestGetMostCommonIndexDelta:
  def test_empty_index(self):
    index = pd.DatetimeIndex([])
    result = udf.get_most_common_index_delta(index)

    assert result.most_common_index_delta is None
    pd.testing.assert_series_equal(result.index_delta_distribution, pd.Series([]))

  def test_single_element(self):
    index = pd.DatetimeIndex(["2024-01-01 09:15:59.999"])
    result = udf.get_most_common_index_delta(index)

    assert result.most_common_index_delta is None
    pd.testing.assert_series_equal(result.index_delta_distribution, pd.Series([]))

  def test_uniform_intervals(self):
    index = pd.DatetimeIndex(
      [
        "2024-01-01 09:15:59.999",
        "2024-01-01 09:16:59.999",
        "2024-01-01 09:17:59.999",
        "2024-01-01 09:18:59.999",
      ]
    )
    result = udf.get_most_common_index_delta(index)
    assert result.most_common_index_delta == 1
    pd.testing.assert_series_equal(
      result.index_delta_distribution, pd.Series([3], index=[1]), check_names=False
    )

  def test_mixed_intervals(self):
    index = pd.DatetimeIndex(
      [
        "2024-01-01 09:15:59.999",
        "2024-01-01 09:17:59.999",
        "2024-01-01 09:19:59.999",
        "2024-01-01 09:22:59.999",
        "2024-01-01 09:24:59.999",
      ]
    )
    result = udf.get_most_common_index_delta(index)

    assert result.most_common_index_delta == 2
    pd.testing.assert_series_equal(
      result.index_delta_distribution,
      pd.Series([3, 1], index=[2, 3]),
      check_names=False,
    )

  def test_tie_returns_first(self):
    index = pd.DatetimeIndex(
      [
        "2024-01-01 09:15:59.999",
        "2024-01-01 09:17:59.999",
        "2024-01-01 09:20:59.999",
      ]
    )
    result = udf.get_most_common_index_delta(index)
    assert result.most_common_index_delta == 2
    pd.testing.assert_series_equal(
      result.index_delta_distribution,
      pd.Series([1, 1], index=[2, 3]),
      check_names=False,
    )

  def test_non_datetime_index(self):
    index = pd.Index([1, 2, 3])
    # non-datetime index should error
    with pytest.raises(Exception):
      udf.get_most_common_index_delta(index)  # type: ignore


class TestGroupByBarGroup:
  def test_basic_grouping(self):
    df = pd.DataFrame(
      {
        "open": [100, 105, 110, 115, 120],
        "high": [102, 107, 112, 117, 122],
        "low": [99, 104, 109, 114, 119],
        "close": [101, 106, 111, 116, 121],
        "volume": [1000, 1500, 2000, 2500, 3000],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:46:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
          "2021-01-01 03:48:59.999+00:00",
          "2021-01-01 03:49:59.999+00:00",
        ]
      ),
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:47:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:47:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:49:59.999+00:00"),
      ],
      index=df.index,
    )

    with ut.expect_no_mutation(df):
      result = udf.group_by_bar_group(df, bar_group)

    expected_df = pd.DataFrame(
      {
        "open": [100, 110, 120],
        "high": [107, 117, 122],
        "low": [99, 109, 119],
        "close": [106, 116, 121],
        "volume": [2500, 4500, 3000],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
          "2021-01-01 03:49:59.999+00:00",
        ]
      ),
    )
    expected_df.index.name = "bar_group"

    expected_size = pd.Series(
      [2, 2, 1],
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
          "2021-01-01 03:49:59.999+00:00",
        ]
      ),
    ).astype("int32")
    expected_size.index.name = "bar_group"

    pd.testing.assert_frame_equal(result.df, expected_df)
    pd.testing.assert_series_equal(result.bar_group_size, expected_size)

  def test_single_bar_group(self):
    df = pd.DataFrame(
      {
        "open": [100, 105, 110],
        "high": [102, 107, 112],
        "low": [99, 104, 109],
        "close": [101, 106, 111],
        "volume": [1000, 1500, 2000],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:46:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
        ]
      ),
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
      ],
      index=df.index,
    )

    with ut.expect_no_mutation(df):
      result = udf.group_by_bar_group(df, bar_group)

    expected_df = pd.DataFrame(
      {
        "open": [100],
        "high": [112],
        "low": [99],
        "close": [111],
        "volume": [4500],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
        ]
      ),
    )
    expected_df.index.name = "bar_group"

    expected_size = pd.Series(
      [3],
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
        ]
      ),
    ).astype("int32")
    expected_size.index.name = "bar_group"

    pd.testing.assert_frame_equal(result.df, expected_df)
    pd.testing.assert_series_equal(result.bar_group_size, expected_size)

  def test_each_row_different_group(self):
    df = pd.DataFrame(
      {
        "open": [100, 105, 110, 115],
        "high": [102, 107, 112, 117],
        "low": [99, 104, 109, 114],
        "close": [101, 106, 111, 116],
        "volume": [1000, 1500, 2000, 2500],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:46:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
          "2021-01-01 03:48:59.999+00:00",
        ]
      ),
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:47:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:48:59.999+00:00"),
      ],
      index=df.index,
    )
    with ut.expect_no_mutation(df):
      result = udf.group_by_bar_group(df, bar_group)
      expected_df = df.copy()
      expected_df.index.name = "bar_group"

    expected_size = pd.Series(
      [1, 1, 1, 1],
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
          "2021-01-01 03:46:59.999+00:00",
          "2021-01-01 03:47:59.999+00:00",
          "2021-01-01 03:48:59.999+00:00",
        ]
      ),
    ).astype("int32")
    expected_size.index.name = "bar_group"

    pd.testing.assert_frame_equal(result.df, expected_df)
    pd.testing.assert_series_equal(result.bar_group_size, expected_size)

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "open": [100],
        "high": [102],
        "low": [99],
        "close": [101],
        "volume": [1000],
      },
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
        ]
      ),
    ).iloc[:0]  # empty dataframe
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
      ],
      index=pd.DatetimeIndex(
        [
          "2021-01-01 03:45:59.999+00:00",
        ]
      ),
    )[:0]

    with ut.expect_no_mutation(df):
      result = udf.group_by_bar_group(df, bar_group)
      expected_df = df.copy()
      expected_df.index.name = "bar_group"

    expected_size = pd.Series(
      [], dtype="int64", index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
    ).astype("int32")
    expected_size.index.name = "bar_group"

    pd.testing.assert_frame_equal(result.df, expected_df)
    pd.testing.assert_series_equal(result.bar_group_size, expected_size)


class TestCalculateGroupedValues:
  def test_basic_functionality(self):
    index = pd.DatetimeIndex(
      [
        "2021-01-01 03:45:59.999+00:00",
        "2021-01-01 03:46:29.999+00:00",
        "2021-01-01 03:46:59.999+00:00",
        "2021-01-01 03:47:29.999+00:00",
      ]
    )
    df = pd.DataFrame(
      {
        "open": [100, 101, 102, 103],
        "high": [105, 106, 107, 108],
        "low": [95, 96, 97, 98],
        "close": [104, 105, 106, 107],
        "volume": [1000, 1100, 1200, 1300],
      },
      index=index,
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
      ],
      index=index,
    )

    def compute_func(df_: pd.DataFrame) -> pd.DataFrame:
      return df_ + df_.shift(1)

    with ut.expect_no_mutation(df):
      result = udf.calculate_grouped_values(df, bar_group, compute_func)

    # Expected calculation:
    # 1. Group by bar_group: first group has [100,105,95,104,1000] and [101,106,96,105,1100]
    #    second group has [102,107,97,106,1200] and [103,108,98,107,1300]
    # 2. Aggregate each group:
    # first -> [100 (open so gets the first value),106 (high so gets the max),95 (low so gets the min),105 (close so gets the last value),2100 (volume so gets the sum)]
    # second -> [102,108,97,107,2500]
    # 3. Apply compute_func (df + df.shift(1)): first row becomes NaN, second row becomes first + second (100 + 102=202, 106 + 108=214, 95 + 97=192, 105 + 107=212, 2100 + 2500=4600)
    # 4. Reindex back to original index: only the third row (start of second group) gets the computed value
    expected = pd.DataFrame(
      {
        "open": [np.nan, np.nan, 202.0, np.nan],
        "high": [np.nan, np.nan, 214.0, np.nan],
        "low": [np.nan, np.nan, 192.0, np.nan],
        "close": [np.nan, np.nan, 212.0, np.nan],
        "volume": [np.nan, np.nan, 4600.0, np.nan],
      },
      index=index,
    )

    pd.testing.assert_frame_equal(result, expected)

  def test_no_grouping(self):
    df = ut.get_test_df(
      start_date=dt.date(2021, 1, 1),
      end_date=dt.date(2021, 1, 2),
    )
    bar_group = df.index.to_series().rename("bar_group")

    def compute_func(df_: pd.DataFrame) -> pd.DataFrame:
      return 2 * df_

    with ut.expect_no_mutation(df):
      result = udf.calculate_grouped_values(df, bar_group, compute_func)

    pd.testing.assert_frame_equal(result, 2 * df)

  def test_multiple_compute_functions(self):
    index = pd.DatetimeIndex(
      [
        "2021-01-01 03:45:59.999+00:00",
        "2021-01-01 03:46:29.999+00:00",
        "2021-01-01 03:46:59.999+00:00",
        "2021-01-01 03:47:29.999+00:00",
      ]
    )
    df = pd.DataFrame(
      {
        "open": [100, 101, 102, 103],
        "high": [105, 106, 107, 108],
        "low": [95, 96, 97, 98],
        "close": [104, 105, 106, 107],
        "volume": [1000, 1100, 1200, 1300],
      },
      index=index,
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
      ],
      index=index,
    )

    # First function: compute sum of open + close
    def compute_func1(df_: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame(
        {"sum_open_close": df_["open"] + df_["close"]},
        index=df_.index,
      )

    # Second function: compute ratio of high / low
    def compute_func2(df_: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame(
        {"ratio_high_low": df_["high"] / df_["low"]},
        index=df_.index,
      )

    with ut.expect_no_mutation(df):
      result = udf.calculate_grouped_values(
        df, bar_group, [compute_func1, compute_func2]
      )

    # Expected calculation:
    # see point 1 and 2 in test_basic_functionality
    expected = pd.DataFrame(
      {
        "sum_open_close": [205.0, np.nan, 209.0, np.nan],
        "ratio_high_low": [106.0 / 95.0, np.nan, 108.0 / 97.0, np.nan],
      },
      index=index,
    )

    pd.testing.assert_frame_equal(result, expected)

  def test_multiple_compute_functions_with_duplicate_columns_raises_error(self):
    index = pd.DatetimeIndex(
      [
        "2021-01-01 03:45:59.999+00:00",
        "2021-01-01 03:46:59.999+00:00",
      ]
    )
    df = pd.DataFrame(
      {
        "open": [100, 101],
        "high": [105, 106],
        "low": [95, 96],
        "close": [104, 105],
        "volume": [1000, 1100],
      },
      index=index,
    )
    bar_group = pd.Series(
      [
        pd.Timestamp("2021-01-01 03:45:59.999+00:00"),
        pd.Timestamp("2021-01-01 03:46:59.999+00:00"),
      ],
      index=index,
    )

    # Both functions return a column with the same name
    def compute_func1(df_: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame({"result": df_["open"] * 2}, index=df_.index)

    def compute_func2(df_: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame({"result": df_["close"] * 3}, index=df_.index)

    with pytest.raises(
      ValueError,
      match="Duplicate columns found across compute functions: \\['result'\\]",
    ):
      udf.calculate_grouped_values(df, bar_group, [compute_func1, compute_func2])

  def test_empty_dataframe(self):
    # the actual data does not matter as we do [:0] on the dataframe to get an empty dataframe
    df = pd.DataFrame(
      {
        "open": [100],
        "high": [102],
        "low": [99],
        "close": [101],
        "volume": [1000],
      },
      index=pd.DatetimeIndex(["2021-01-01 03:45:59.999+00:00"]),
    ).iloc[:0]
    bar_group = pd.Series(
      [pd.Timestamp("2021-01-01 03:45:59.999+00:00")],
      index=pd.DatetimeIndex(["2021-01-01 03:45:59.999+00:00"]),
    )[:0]
    assert df.empty

    def compute_func(df_: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame({"test_col": df_["open"] * 2}, index=df_.index)

    with ut.expect_no_mutation(df):
      result = udf.calculate_grouped_values(df, bar_group, compute_func)

    pd.testing.assert_index_equal(result.index, df.index)
