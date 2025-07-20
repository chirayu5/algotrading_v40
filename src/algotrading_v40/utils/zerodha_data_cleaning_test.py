from datetime import datetime

import pandas as pd
import pytest
import pytz

import algotrading_v40.utils.zerodha_data_cleaning as uzdc


class TestFixUnusualBars:
  def test_no_unusual_bars(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 17, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [100.0, 101.0, 102.0],
        "high": [100.5, 101.5, 102.5],
        "low": [99.5, 100.5, 101.5],
        "close": [100.0, 101.0, 102.0],
      }
    )
    dfb = df.copy()
    result = uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert isinstance(result, uzdc.FixUnusualBarsResult)
    assert len(result.df) == 3
    assert len(result.df_dropped) == 0
    assert result.n_dropped == 0
    pd.testing.assert_frame_equal(result.df, df)

  def test_with_unusual_bars_seconds(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 30)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [1, 2, 3],
        "high": [4, 5, 6],
        "low": [7, 8, 9],
        "close": [10, 11, 12],
      }
    )
    dfb = df.copy()
    result = uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result.df) == 2
    assert len(result.df_dropped) == 1
    assert result.n_dropped == 1

    expected_rounded_dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 0)),
    ]

    assert result.df.iloc[0]["date"] == expected_rounded_dates[0]
    assert result.df.iloc[1]["date"] == expected_rounded_dates[1]
    assert result.df[["open", "high", "low", "close"]].iloc[0].to_list() == [
      1,
      4,
      7,
      10,
    ]
    assert result.df[["open", "high", "low", "close"]].iloc[1].to_list() == [
      3,
      6,
      9,
      12,
    ]

  def test_duplicate_rounded_dates(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 30)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 45)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [1, 2, 3],
        "high": [4, 5, 6],
        "low": [7, 8, 9],
        "close": [10, 11, 12],
      }
    )
    dfb = df.copy()
    with pytest.raises(ValueError, match="Unusual bars have duplicate rounded dates"):
      uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "date": pd.Series([], dtype="datetime64[ns, Asia/Kolkata]"),
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
      }
    )
    dfb = df.copy()
    result = uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result.df) == 0
    assert len(result.df_dropped) == 0
    assert result.n_dropped == 0

  def test_single_row_dataframe(self):
    ist = pytz.timezone("Asia/Kolkata")
    df = pd.DataFrame(
      {
        "date": [ist.localize(datetime(2015, 2, 2, 9, 15, 30))],
        "open": [100.0],
        "high": [100.5],
        "low": [99.5],
        "close": [100.0],
      }
    )
    dfb = df.copy()
    result = uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result.df) == 1
    assert len(result.df_dropped) == 0
    assert result.n_dropped == 0
    assert result.df.iloc[0]["date"] == ist.localize(datetime(2015, 2, 2, 9, 16, 0))

  def test_mixed_scenario(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 30)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 45)),
      ist.localize(datetime(2015, 2, 2, 9, 17, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [1, 2, 3, 4],
        "high": [6, 7, 8, 9],
        "low": [11, 12, 13, 14],
        "close": [16, 17, 18, 19],
      }
    )
    dfb = df.copy()
    result = uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(result.df) == 3
    assert len(result.df_dropped) == 1
    assert result.n_dropped == 1

    assert result.df.iloc[0]["date"] == ist.localize(datetime(2015, 2, 2, 9, 15, 0))
    assert result.df.iloc[1]["date"] == ist.localize(datetime(2015, 2, 2, 9, 16, 0))
    assert result.df.iloc[2]["date"] == ist.localize(datetime(2015, 2, 2, 9, 17, 0))

    assert result.df[["open", "high", "low", "close"]].iloc[0].to_list() == [
      1,
      6,
      11,
      16,
    ]
    assert result.df[["open", "high", "low", "close"]].iloc[1].to_list() == [
      2,
      7,
      12,
      17,
    ]
    assert result.df[["open", "high", "low", "close"]].iloc[2].to_list() == [
      4,
      9,
      14,
      19,
    ]

  def test_duplicate_dates_raises_error(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 15, 2)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 2)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [100.0, 101.0, 102.0],
        "high": [100.5, 101.5, 102.5],
        "low": [99.5, 100.5, 101.5],
        "close": [100.0, 101.0, 102.0],
      }
    )
    dfb = df.copy()
    with pytest.raises(ValueError, match="DataFrame contains duplicate dates"):
      uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)

  def test_non_monotonic_dates_raises_error(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2015, 2, 2, 9, 17, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(datetime(2015, 2, 2, 9, 16, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [100.0, 101.0, 102.0],
        "high": [100.5, 101.5, 102.5],
        "low": [99.5, 100.5, 101.5],
        "close": [100.0, 101.0, 102.0],
      }
    )
    dfb = df.copy()
    with pytest.raises(
      ValueError, match="DataFrame dates are not in strictly ascending order"
    ):
      uzdc.fix_unusual_bars(df)
    pd.testing.assert_frame_equal(df, dfb)


class TestSetIndexToBarCloseTimestamp:
  def test_adds_59_999_seconds_and_converts_to_utc(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(datetime(2016, 1, 4, 9, 15, 0)),
      ist.localize(datetime(2016, 1, 4, 9, 16, 0)),
      ist.localize(datetime(2016, 1, 4, 15, 29, 0)),
    ]
    df = pd.DataFrame(
      {
        "date": dates,
        "open": [448.50, 446.55, 446.55],
        "high": [448.50, 446.65, 446.90],
        "low": [444.80, 446.25, 446.20],
        "close": [446.55, 446.55, 446.30],
        "volume": [51571, 24701, 35062],
      }
    )
    dfb = df.copy()
    result = uzdc.set_index_to_bar_close_timestamp(df)
    pd.testing.assert_frame_equal(df, dfb)

    expected_index = [
      pytz.UTC.localize(datetime(2016, 1, 4, 3, 45, 59, 999000)),
      pytz.UTC.localize(datetime(2016, 1, 4, 3, 46, 59, 999000)),
      pytz.UTC.localize(datetime(2016, 1, 4, 9, 59, 59, 999000)),
    ]
    expected_df = pd.DataFrame(
      {
        "open": [448.50, 446.55, 446.55],
        "high": [448.50, 446.65, 446.90],
        "low": [444.80, 446.25, 446.20],
        "close": [446.55, 446.55, 446.30],
        "volume": [51571, 24701, 35062],
      },
      index=pd.DatetimeIndex(expected_index, name="date"),
    )
    assert result.index.dtype == "datetime64[ns, UTC]"
    pd.testing.assert_frame_equal(result, expected_df)

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "date": pd.Series([], dtype="datetime64[ns, Asia/Kolkata]"),
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      }
    )

    result = uzdc.set_index_to_bar_close_timestamp(df)

    expected_df = pd.DataFrame(
      {
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      },
      index=pd.DatetimeIndex([], name="date", dtype="datetime64[ns, UTC]"),
    )

    pd.testing.assert_frame_equal(result, expected_df)
