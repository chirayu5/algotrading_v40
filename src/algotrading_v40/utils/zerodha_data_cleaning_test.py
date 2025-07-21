import datetime as dt

import numpy as np
import pandas as pd
import pytest
import pytz

import algotrading_v40.utils.zerodha_data_cleaning as uzdc


class TestFixUnusualBars:
  def test_no_unusual_bars(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 17, 0)),
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
    assert len(result.df_original_dropped) == 0
    assert len(result.df_original_date_needs_fix) == 0
    assert result.n_dropped == 0
    assert result.n_date_fixed == 0
    pd.testing.assert_frame_equal(result.df, df)

  def test_with_unusual_bars_seconds(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 30)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0)),
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
    assert len(result.df_original_dropped) == 1
    assert len(result.df_original_date_needs_fix) == 0
    assert result.n_dropped == 1
    assert result.n_date_fixed == 0

    expected_rounded_dates = [
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0)),
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
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 30)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 45)),
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
    assert len(result.df_original_dropped) == 0
    assert result.n_dropped == 0

  def test_single_row_dataframe(self):
    ist = pytz.timezone("Asia/Kolkata")
    df = pd.DataFrame(
      {
        "date": [ist.localize(dt.datetime(2015, 2, 2, 9, 15, 30))],
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
    assert len(result.df_original_dropped) == 0
    assert result.n_dropped == 0
    assert result.df.iloc[0]["date"] == ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0))

  def test_mixed_scenario(self):
    ist = pytz.timezone("Asia/Kolkata")
    dates = [
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 30)),  # should become 9:16:00
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 45)),  # should be dropped
      ist.localize(dt.datetime(2015, 2, 2, 9, 17, 0)),
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
    assert len(result.df_original_dropped) == 1
    assert len(result.df_original_date_needs_fix) == 1
    assert result.n_dropped == 1
    assert result.n_date_fixed == 1

    assert result.df.iloc[0]["date"] == ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0))
    assert result.df.iloc[1]["date"] == ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0))
    assert result.df.iloc[2]["date"] == ist.localize(dt.datetime(2015, 2, 2, 9, 17, 0))

    assert result.df_original_date_needs_fix.iloc[0]["date"] == ist.localize(
      dt.datetime(2015, 2, 2, 9, 15, 30)
    )
    assert result.df_original_date_needs_fix[["open", "high", "low", "close"]].iloc[
      0
    ].to_list() == [
      2,
      7,
      12,
      17,
    ]

    assert result.df_original_dropped.iloc[0]["date"] == ist.localize(
      dt.datetime(2015, 2, 2, 9, 16, 45)
    )
    assert result.df_original_dropped[["open", "high", "low", "close"]].iloc[
      0
    ].to_list() == [
      3,
      8,
      13,
      18,
    ]

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
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 2)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 2)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0)),
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
      ist.localize(dt.datetime(2015, 2, 2, 9, 17, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 15, 0)),
      ist.localize(dt.datetime(2015, 2, 2, 9, 16, 0)),
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
      ist.localize(dt.datetime(2016, 1, 4, 9, 15, 0)),
      ist.localize(dt.datetime(2016, 1, 4, 9, 16, 0)),
      ist.localize(dt.datetime(2016, 1, 4, 15, 29, 0)),
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
      pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 45, 59, 999000)),
      pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 46, 59, 999000)),
      pytz.UTC.localize(dt.datetime(2016, 1, 4, 9, 59, 59, 999000)),
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
    dfb = df.copy()
    result = uzdc.set_index_to_bar_close_timestamp(df)
    pd.testing.assert_frame_equal(df, dfb)
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


class TestDropNonStandardIndianTradingHours:
  # In UTC, Indian market hours are
  # 03:45:59.999000+00:00
  # 09:59:59.999000+00:00
  def test_all_outside_trading_hours(self):
    df = pd.DataFrame(
      {
        "open": [448.50, 446.55, 446.30],
        "high": [448.50, 446.65, 446.90],
        "low": [444.80, 446.25, 446.20],
        "close": [446.55, 446.55, 446.30],
        "volume": [51571, 24701, 35062],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 15, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 10, 0, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 15, 0, 59, 999000)),
        ],
        name="date",
      ),
    )
    dfb = df.copy()
    result = uzdc.drop_non_standard_indian_trading_hours(df)
    pd.testing.assert_frame_equal(df, dfb)
    expected_empty_df = pd.DataFrame(
      {
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      },
      index=pd.DatetimeIndex([], name="date", dtype="datetime64[ns, UTC]"),
    )

    pd.testing.assert_frame_equal(result.df, expected_empty_df)
    pd.testing.assert_frame_equal(result.df_original_dropped, df)
    assert result.n_dropped == 3

  def test_mixed_trading_hours(self):
    df = pd.DataFrame(
      {
        "open": [448.50, 446.55, 446.30, 445.00, 444.50],
        "high": [448.50, 446.65, 446.90, 445.20, 444.80],
        "low": [444.80, 446.25, 446.20, 444.50, 444.00],
        "close": [446.55, 446.55, 446.30, 444.80, 444.20],
        "volume": [51571, 24701, 35062, 20000, 15000],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 15, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 45, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 5, 30, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 9, 59, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 10, 0, 59, 999000)),
        ],
        name="date",
      ),
    )
    dfb = df.copy()
    result = uzdc.drop_non_standard_indian_trading_hours(df)
    pd.testing.assert_frame_equal(df, dfb)
    expected_kept_df = pd.DataFrame(
      {
        "open": [446.55, 446.30, 445.00],
        "high": [446.65, 446.90, 445.20],
        "low": [446.25, 446.20, 444.50],
        "close": [446.55, 446.30, 444.80],
        "volume": [24701, 35062, 20000],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 45, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 5, 30, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 9, 59, 59, 999000)),
        ],
        name="date",
      ),
    )

    expected_dropped_df = pd.DataFrame(
      {
        "open": [448.50, 444.50],
        "high": [448.50, 444.80],
        "low": [444.80, 444.00],
        "close": [446.55, 444.20],
        "volume": [51571, 15000],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 15, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 10, 0, 59, 999000)),
        ],
        name="date",
      ),
    )

    pd.testing.assert_frame_equal(result.df, expected_kept_df)
    pd.testing.assert_frame_equal(result.df_original_dropped, expected_dropped_df)
    assert result.n_dropped == 2

  def test_index_not_utc(self):
    df = pd.DataFrame(
      {
        "open": [448.50, 446.55, 446.30],
        "high": [448.50, 446.65, 446.90],
        "low": [444.80, 446.25, 446.20],
        "close": [446.55, 446.55, 446.30],
        "volume": [51571, 24701, 35062],
      },
      index=pd.DatetimeIndex(
        [
          pytz.timezone("Asia/Kolkata").localize(
            dt.datetime(2016, 1, 4, 3, 15, 59, 999000)
          ),
          pytz.timezone("Asia/Kolkata").localize(
            dt.datetime(2016, 1, 4, 3, 45, 59, 999000)
          ),
          pytz.timezone("Asia/Kolkata").localize(
            dt.datetime(2016, 1, 4, 9, 59, 59, 999000)
          ),
        ],
        name="date",
      ),
    )
    dfb = df.copy()
    with pytest.raises(ValueError, match="DataFrame index must have UTC timezone"):
      uzdc.drop_non_standard_indian_trading_hours(df)
    pd.testing.assert_frame_equal(df, dfb)

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      },
      index=pd.DatetimeIndex([], name="date", dtype="datetime64[ns, UTC]"),
    )
    dfb = df.copy()
    result = uzdc.drop_non_standard_indian_trading_hours(df)
    pd.testing.assert_frame_equal(df, dfb)
    pd.testing.assert_frame_equal(result.df, df)
    pd.testing.assert_frame_equal(result.df_original_dropped, df)
    assert result.n_dropped == 0


class TestFixHighLowValues:
  def test_no_fixes_needed(self):
    df = pd.DataFrame(
      {
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 106.0],
        "low": [95.0, 103.0, 100.0],
        "close": [108.0, 106.0, 104.0],
        "volume": [1000, 2000, 1500],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 45, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 46, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 47, 59, 999000)),
        ],
        name="date",
      ),
    )
    dfb = df.copy()
    result = uzdc.fix_high_low_values(df)
    pd.testing.assert_frame_equal(df, dfb)
    pd.testing.assert_frame_equal(result.df, df)
    assert len(result.df_original_high_needs_fix) == 0
    assert len(result.df_original_low_needs_fix) == 0
    assert result.n_high_fixed == 0
    assert result.n_low_fixed == 0

  def test_both_high_and_low_need_fix(self):
    df = pd.DataFrame(
      {
        "open": [100.0, 105.0, 102.0, 3, 3],
        "high": [95.0, 103.0, 108.0, 2, 4],
        "low": [105.0, 110.0, 95.0, 1, 1],
        "close": [98.0, 108.0, 104.0, 2, 0.5],
        "volume": [1000, 2000, 1500, 500, 500],
      },
      index=pd.DatetimeIndex(
        [
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 45, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 46, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 47, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 48, 59, 999000)),
          pytz.UTC.localize(dt.datetime(2016, 1, 4, 3, 49, 59, 999000)),
        ],
        name="date",
      ),
    )
    dfb = df.copy()
    result = uzdc.fix_high_low_values(df)
    pd.testing.assert_frame_equal(df, dfb)
    expected_df = df.copy()
    expected_df.loc[expected_df.index[0], "high"] = 105.0
    expected_df.loc[expected_df.index[0], "low"] = 95.0
    expected_df.loc[expected_df.index[1], "high"] = 110.0
    expected_df.loc[expected_df.index[1], "low"] = 103.0
    expected_df.loc[expected_df.index[3], "high"] = 3
    expected_df.loc[expected_df.index[3], "low"] = 1
    expected_df.loc[expected_df.index[4], "high"] = 4
    expected_df.loc[expected_df.index[4], "low"] = 0.5

    expected_high_needs_fix = df.iloc[[0, 1, 3]]
    expected_low_needs_fix = df.iloc[[0, 1, 4]]

    pd.testing.assert_frame_equal(result.df, expected_df)
    pd.testing.assert_frame_equal(
      result.df_original_high_needs_fix, expected_high_needs_fix
    )
    pd.testing.assert_frame_equal(
      result.df_original_low_needs_fix, expected_low_needs_fix
    )
    assert result.n_high_fixed == 3
    assert result.n_low_fixed == 3

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      },
      index=pd.DatetimeIndex([], name="date", dtype="datetime64[ns, UTC]"),
    )

    result = uzdc.fix_high_low_values(df)

    pd.testing.assert_frame_equal(result.df, df)
    pd.testing.assert_frame_equal(result.df_original_high_needs_fix, df)
    pd.testing.assert_frame_equal(result.df_original_low_needs_fix, df)
    assert result.n_high_fixed == 0
    assert result.n_low_fixed == 0


class TestCountBarsPerTradingDay:
  def test_normal_case(self):
    dates1 = pd.date_range(
      start="2023-01-01 03:45:59.999000+00:00",
      end="2023-01-01 09:59:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )
    dates2 = pd.date_range(
      start="2023-04-02 03:45:59.999000+00:00",
      end="2023-04-02 09:59:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )
    dates = dates1.union(dates2)
    df = pd.DataFrame(
      {
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000] * len(dates),
      },
      index=dates,
    )
    df.index.name = "date"
    dfb = df.copy()
    result = uzdc.count_bars_per_trading_day(df)
    pd.testing.assert_frame_equal(df, dfb)
    expected_df = pd.DataFrame({"count": [375, 375]})
    expected_df.index = pd.Index(
      [
        dt.date(2023, 1, 1),
        dt.date(2023, 4, 2),
      ],
      name="date",
    )
    pd.testing.assert_frame_equal(result.df, expected_df)
    assert result.n_dates_with_less_than_375_bars == 0
    assert result.fraction_dates_with_less_than_375_bars == 0
    assert result.n_dates == 2
    assert result.dates_with_less_than_375_bars == []

  def test_some_days_with_less_than_375_bars(self):
    dates1 = pd.date_range(
      start="2023-01-01 03:45:59.999000+00:00",
      end="2023-01-01 09:59:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )
    dates2 = pd.date_range(
      start="2023-01-02 03:45:59.999000+00:00",
      end="2023-01-02 05:00:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )
    dates3 = pd.date_range(
      start="2023-01-03 03:45:59.999000+00:00",
      end="2023-01-03 09:59:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )

    all_dates = dates1.union(dates2).union(dates3)
    df = pd.DataFrame(
      {
        "open": [100.0] * len(all_dates),
        "high": [105.0] * len(all_dates),
        "low": [95.0] * len(all_dates),
        "close": [102.0] * len(all_dates),
        "volume": [1000] * len(all_dates),
      },
      index=all_dates,
    )
    df.index.name = "date"
    dfb = df.copy()
    result = uzdc.count_bars_per_trading_day(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert len(dates2) == 76
    expected_df = pd.DataFrame({"count": [375, 76, 375]})
    expected_df.index = pd.Index(
      [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)], name="date"
    )

    pd.testing.assert_frame_equal(result.df, expected_df)
    assert result.n_dates_with_less_than_375_bars == 1
    assert np.isclose(result.fraction_dates_with_less_than_375_bars, 1.0 / 3)
    assert result.n_dates == 3
    assert result.dates_with_less_than_375_bars == [dt.date(2023, 1, 2)]

  def test_single_day_less_than_375_bars(self):
    dates = pd.date_range(
      start="2023-01-01 03:45:59.999000+00:00",
      end="2023-01-01 09:58:59.999000+00:00",
      freq="1min",
      tz="UTC",
    )
    df = pd.DataFrame(
      {
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000] * len(dates),
      },
      index=dates,
    )
    df.index.name = "date"
    dfb = df.copy()
    result = uzdc.count_bars_per_trading_day(df)
    pd.testing.assert_frame_equal(df, dfb)
    expected_df = pd.DataFrame({"count": [374]})
    expected_df.index = pd.Index([dt.date(2023, 1, 1)], name="date")

    pd.testing.assert_frame_equal(result.df, expected_df)
    assert result.n_dates_with_less_than_375_bars == 1
    assert np.isclose(result.fraction_dates_with_less_than_375_bars, 1.0)
    assert result.n_dates == 1
    assert result.dates_with_less_than_375_bars == [dt.date(2023, 1, 1)]

  def test_empty_dataframe(self):
    df = pd.DataFrame(
      {
        "open": pd.Series([], dtype="float64"),
        "high": pd.Series([], dtype="float64"),
        "low": pd.Series([], dtype="float64"),
        "close": pd.Series([], dtype="float64"),
        "volume": pd.Series([], dtype="int64"),
      },
      index=pd.DatetimeIndex([], name="date", dtype="datetime64[ns, UTC]"),
    )

    result = uzdc.count_bars_per_trading_day(df)

    expected_df = pd.DataFrame({"count": []})
    expected_df.index = pd.Index([], name="date", dtype="object")
    expected_df["count"] = expected_df["count"].astype("int64")

    pd.testing.assert_frame_equal(result.df, expected_df)
    assert result.n_dates_with_less_than_375_bars == 0
    assert np.isclose(result.fraction_dates_with_less_than_375_bars, 0.0)
    assert result.n_dates == 0
    assert result.dates_with_less_than_375_bars == []


class TestAnalyzeNumericSeriesQuality:
  def test_normal_series_no_issues(self):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_series_with_zeros_different_representations(self):
    s = pd.Series([1.0, 0.0, 0, -0.0, -0, 2.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 4  # 0.0, 0, -0.0, -0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_series_with_negatives(self):
    s = pd.Series([1.0, -2.0, 3.0, -4.5, -0, -0.0, -np.inf, -np.nan, 5.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 2
    assert (
      result.n_negatives == 2
    )  # -0, -0.0, -np.inf and -np.nan are not counted as negatives
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_series_with_nan_values(self):
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_series_with_inf_values(self):
    s = pd.Series([1.0, np.inf, 3.0, -np.inf, 5.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_series_with_mixed_bad_values(self):
    s = pd.Series([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0, None])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4  # nan, inf, -inf, None
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 1

  def test_series_with_bad_values_at_start(self):
    s = pd.Series([np.nan, np.inf, 1.0, 2.0, 3.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 0

  def test_series_with_bad_values_at_end(self):
    s = pd.Series([1.0, 2.0, 3.0, np.nan, np.inf])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 2

  def test_series_with_bad_values_at_both_ends(self):
    s = pd.Series([np.nan, np.inf, 1.0, 2.0, 3.0, -np.inf, np.nan])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 2

  def test_series_with_bad_values_in_middle(self):
    s = pd.Series([1.0, 2.0, np.nan, np.inf, 3.0, 4.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 2
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_all_bad_values(self):
    s = pd.Series([np.nan, np.inf, -np.inf, np.nan])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 4
    assert result.n_bad_values_at_end == 4

  def test_empty_series(self):
    s = pd.Series([], dtype=float)
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_single_good_value(self):
    s = pd.Series([42.5])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_single_bad_value(self):
    s = pd.Series([np.nan])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 1
    assert result.n_zeros == 0
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 1
    assert result.n_bad_values_at_end == 1

  def test_single_zero_value(self):
    s = pd.Series([0.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_single_negative_value(self):
    s = pd.Series([-5.0])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 0
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_complex_mixed_case_with_result_dtypes(self):
    # Test with bad values at start/end, zeros, negatives, and good values
    s = pd.Series([np.nan, -np.inf, -2.0, 0.0, 1.0, -3.5, np.inf, np.nan])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 4  # 2 nan, 2 inf
    assert result.n_zeros == 1
    assert (
      result.n_negatives == 2
    )  # -2.0, -3.5 (-inf values don't count as they're bad)
    assert result.n_bad_values_at_start == 2
    assert result.n_bad_values_at_end == 2

    assert isinstance(result.n_bad_values, int)
    assert isinstance(result.n_zeros, int)
    assert isinstance(result.n_negatives, int)
    assert isinstance(result.n_bad_values_at_start, int)
    assert isinstance(result.n_bad_values_at_end, int)
    for field_name in result.__dataclass_fields__:
      assert isinstance(getattr(result, field_name), int)

  def test_integer_series(self):
    s = pd.Series([1, 2, -3, 0, 5], dtype=int)
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_float32_series(self):
    s = pd.Series([1.0, -2.0, 0.0, np.nan], dtype=np.float32)
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 1
    assert result.n_zeros == 1
    assert result.n_negatives == 1
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 1

  def test_non_numeric_series_raises_error(self):
    s = pd.Series(["a", "b", "c"])
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_object_series_with_numbers_raises_error(self):
    s = pd.Series([1, 2, "3"], dtype=object)
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_datetime_series_raises_error(self):
    s = pd.Series(pd.date_range("2020-01-01", periods=3))
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_string_series_raises_error(self):
    s = pd.Series(["1.0", "2.0", "3.0"])
    sb = s.copy()
    with pytest.raises(ValueError, match="Series must be numeric"):
      uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)

  def test_boolean_series_works(self):
    s = pd.Series([True, False, True])
    sb = s.copy()
    result = uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)
    assert result.n_bad_values == 0
    assert result.n_zeros == 1  # False
    assert result.n_negatives == 0
    assert result.n_bad_values_at_start == 0
    assert result.n_bad_values_at_end == 0

  def test_pd_null_na_None_series_is_not_numeric(self):
    with pytest.raises(
      TypeError,
      match="float\\(\\) argument must be a string or a real number, not 'NAType'",
    ):
      # A pd.NA cannot be converted to a float32, so it raises a TypeError
      _ = pd.Series([1.0, None, 3.0, pd.NA, 5.0], dtype=np.float32)

    s = pd.Series([1.0, 2.0, 3.0, pd.NA])
    sb = s.copy()
    # If we try to analyze a series with pd.NA, it raises a ValueError
    with pytest.raises(ValueError, match="Series must be numeric"):
      uzdc.analyze_numeric_series_quality(s)
    pd.testing.assert_series_equal(s, sb)


class TestAnalyseNumericColumnsQuality:
  def test_empty_dataframe(self):
    df = pd.DataFrame()
    dfb = df.copy()
    result = uzdc.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert result == {}

  def test_no_numeric_columns(self):
    df = pd.DataFrame(
      {"str_col": ["a", "b", "c"], "date_col": pd.date_range("2020-01-01", periods=3)}
    )
    dfb = df.copy()
    result = uzdc.analyse_numeric_columns_quality(df)
    pd.testing.assert_frame_equal(df, dfb)
    assert result == {}

  def test_single_numeric_column(self):
    df = pd.DataFrame({"numeric_col": [1.0, 2.0, 3.0], "str_col": ["a", "b", "c"]})
    dfb = df.copy()
    result = uzdc.analyse_numeric_columns_quality(df)
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
    result = uzdc.analyse_numeric_columns_quality(df)
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
    result = uzdc.analyse_numeric_columns_quality(df)
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
    result = uzdc.analyse_numeric_columns_quality(df)
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
