import copy
import datetime as dt
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest
import pytz

import algotrading_v40.structures.instrument_desc as sid
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
      index=pd.DatetimeIndex(
        expected_index, name="bar_close_timestamp", dtype="datetime64[ns, UTC]"
      ),
    )
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
      index=pd.DatetimeIndex(
        [], name="bar_close_timestamp", dtype="datetime64[ns, UTC]"
      ),
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


def _assert_instrument_desc_to_df_matches(
  a: dict[sid.InstrumentDesc, pd.DataFrame],
  b: dict[sid.InstrumentDesc, pd.DataFrame],
):
  assert a.keys() == b.keys()
  for instrument_desc in a.keys():
    pd.testing.assert_frame_equal(a[instrument_desc], b[instrument_desc])


class TestCountMissingTradingSessions:
  def _create_test_dataframe(self, dates: Sequence[dt.date]) -> pd.DataFrame:
    datetime_index = pd.DatetimeIndex(
      [
        pd.Timestamp(date)
        .tz_localize("UTC")
        .replace(hour=3, minute=45, second=59, microsecond=999000)
        for date in dates
      ]
    )

    return pd.DataFrame(
      {
        "open": [100.0] * len(dates),
        "high": [105.0] * len(dates),
        "low": [95.0] * len(dates),
        "close": [102.0] * len(dates),
        "volume": [1000] * len(dates),
      },
      index=datetime_index,
    )

  def _create_instrument_descs(self) -> Dict[str, sid.InstrumentDesc]:
    return {
      "ADANIENT": sid.EquityDesc(market=sid.Market.INDIAN_MARKET, symbol="ADANIENT"),
      "ADANIPORTS": sid.EquityDesc(
        market=sid.Market.INDIAN_MARKET, symbol="ADANIPORTS"
      ),
      "NIFTY_FIN": sid.IndexDesc(
        market=sid.Market.INDIAN_MARKET, symbol="NIFTY FIN SERVICE"
      ),
      "NIFTY_IT": sid.IndexDesc(market=sid.Market.INDIAN_MARKET, symbol="NIFTY IT"),
    }

  def test_no_missing_sessions_single_instrument(self):
    instrument_descs = self._create_instrument_descs()
    dates = [dt.date(2023, 1, 2), dt.date(2023, 1, 4), dt.date(2023, 1, 5)]

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(dates)
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    assert len(result.instrument_desc_to_missing_sessions) == 1
    assert (
      len(result.instrument_desc_to_missing_sessions[instrument_descs["ADANIENT"]]) == 0
    )
    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["ADANIENT"]] == 0
    )
    np.testing.assert_array_equal(result.all_dates, np.array(sorted(set(dates))))

  def test_no_missing_sessions_multiple_instruments(self):
    instrument_descs = self._create_instrument_descs()
    dates = [dt.date(2023, 1, 2), dt.date(2023, 1, 3), dt.date(2023, 1, 4)]

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(dates),
      instrument_descs["ADANIPORTS"]: self._create_test_dataframe(dates),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    assert len(result.instrument_desc_to_missing_sessions) == 2
    for instrument_desc in instrument_desc_to_df.keys():
      assert len(result.instrument_desc_to_missing_sessions[instrument_desc]) == 0
      assert result.instrument_desc_to_n_missing_sessions[instrument_desc] == 0
    np.testing.assert_array_equal(result.all_dates, np.array(sorted(set(dates))))

  def test_missing_sessions_multiple_instruments_different_ranges(self):
    instrument_descs = self._create_instrument_descs()
    # all dates are 2 3 5 6 7 8
    adanient_dates = [
      dt.date(2023, 1, 2),
      dt.date(2023, 1, 3),
      # 5 missing
      dt.date(2023, 1, 6),
      dt.date(2023, 1, 7),
    ]

    adaniports_dates = [
      dt.date(2023, 1, 3),
      dt.date(2023, 1, 5),
      # 6 missing
      # 7 missing
      dt.date(2023, 1, 8),
    ]

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(adanient_dates),
      instrument_descs["ADANIPORTS"]: self._create_test_dataframe(adaniports_dates),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    # ADANIENT should be missing 2023-01-05 (within its range)
    adanient_missing = result.instrument_desc_to_missing_sessions[
      instrument_descs["ADANIENT"]
    ]
    assert len(adanient_missing) == 1
    assert adanient_missing[0] == dt.date(2023, 1, 5)

    # ADANIPORTS should be missing 2023-01-06 (within its range)
    adaniports_missing = result.instrument_desc_to_missing_sessions[
      instrument_descs["ADANIPORTS"]
    ]
    assert len(adaniports_missing) == 2
    np.testing.assert_array_equal(
      adaniports_missing, np.array([dt.date(2023, 1, 6), dt.date(2023, 1, 7)])
    )

    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["ADANIENT"]] == 1
    )
    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["ADANIPORTS"]] == 2
    )
    np.testing.assert_array_equal(
      result.all_dates, np.array(sorted(set(adanient_dates + adaniports_dates)))
    )

  def test_empty_input(self):
    instrument_desc_to_df = {}
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    assert len(result.instrument_desc_to_missing_sessions) == 0
    assert len(result.instrument_desc_to_n_missing_sessions) == 0

  def test_single_date_per_instrument(self):
    instrument_descs = self._create_instrument_descs()

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe([dt.date(2023, 1, 2)]),
      instrument_descs["NIFTY_IT"]: self._create_test_dataframe([dt.date(2023, 1, 3)]),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    adanient_missing = result.instrument_desc_to_missing_sessions[
      instrument_descs["ADANIENT"]
    ]
    assert len(adanient_missing) == 0

    nifty_it_missing = result.instrument_desc_to_missing_sessions[
      instrument_descs["NIFTY_IT"]
    ]
    assert len(nifty_it_missing) == 0
    np.testing.assert_array_equal(
      result.all_dates, np.array([dt.date(2023, 1, 2), dt.date(2023, 1, 3)])
    )

  def test_non_overlapping_date_ranges(self):
    instrument_descs = self._create_instrument_descs()

    # ADANIENT: January 2-4
    adanient_dates = [dt.date(2023, 1, 2), dt.date(2023, 1, 3), dt.date(2023, 1, 4)]

    # ADANIPORTS: January 6-8
    adaniports_dates = [dt.date(2023, 1, 6), dt.date(2023, 1, 7), dt.date(2023, 1, 8)]

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(adanient_dates),
      instrument_descs["ADANIPORTS"]: self._create_test_dataframe(adaniports_dates),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    # Each instrument should have no missing sessions within their own ranges
    assert (
      len(result.instrument_desc_to_missing_sessions[instrument_descs["ADANIENT"]]) == 0
    )
    assert (
      len(result.instrument_desc_to_missing_sessions[instrument_descs["ADANIPORTS"]])
      == 0
    )
    np.testing.assert_array_equal(
      result.all_dates, np.array(sorted(set(adanient_dates + adaniports_dates)))
    )

  def test_three_descs(self):
    instrument_descs = self._create_instrument_descs()
    # all dates are 2 5 7 8 10 14
    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(
        [
          dt.date(2023, 1, 2),
          # 5 missing
          dt.date(2023, 1, 7),
          dt.date(2023, 1, 8),
          dt.date(2023, 1, 10),
        ]
      ),
      instrument_descs["ADANIPORTS"]: self._create_test_dataframe(
        [
          dt.date(2023, 1, 5),
          # 7 missing
          dt.date(2023, 1, 8),
          # 10 missing
          dt.date(2023, 1, 14),
        ]
      ),
      instrument_descs["NIFTY_IT"]: self._create_test_dataframe(
        [
          dt.date(2023, 1, 5),
          # 7 missing
          # 8 missing
          # 10 missing
          dt.date(2023, 1, 14),
        ]
      ),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_sessions(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    np.testing.assert_array_equal(
      result.instrument_desc_to_missing_sessions[instrument_descs["ADANIENT"]],
      np.array([dt.date(2023, 1, 5)]),
    )
    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["ADANIENT"]] == 1
    )
    np.testing.assert_array_equal(
      result.instrument_desc_to_missing_sessions[instrument_descs["ADANIPORTS"]],
      np.array([dt.date(2023, 1, 7), dt.date(2023, 1, 10)]),
    )
    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["ADANIPORTS"]] == 2
    )
    np.testing.assert_array_equal(
      result.instrument_desc_to_missing_sessions[instrument_descs["NIFTY_IT"]],
      np.array([dt.date(2023, 1, 7), dt.date(2023, 1, 8), dt.date(2023, 1, 10)]),
    )
    assert (
      result.instrument_desc_to_n_missing_sessions[instrument_descs["NIFTY_IT"]] == 3
    )
    np.testing.assert_array_equal(
      result.all_dates,
      np.array(
        [
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 5),
          dt.date(2023, 1, 7),
          dt.date(2023, 1, 8),
          dt.date(2023, 1, 10),
          dt.date(2023, 1, 14),
        ]
      ),
    )


class TestCountMissingTradingBars:
  def _create_test_dataframe(self, timestamps: Sequence[pd.Timestamp]) -> pd.DataFrame:
    datetime_index = pd.DatetimeIndex(timestamps)

    return pd.DataFrame(
      {
        "open": [100.0] * len(timestamps),
        "high": [105.0] * len(timestamps),
        "low": [95.0] * len(timestamps),
        "close": [102.0] * len(timestamps),
        "volume": [1000] * len(timestamps),
      },
      index=datetime_index,
    )

  def _create_instrument_descs(self) -> Dict[str, sid.InstrumentDesc]:
    return {
      "ADANIENT": sid.EquityDesc(market=sid.Market.INDIAN_MARKET, symbol="ADANIENT"),
      "ADANIPORTS": sid.EquityDesc(
        market=sid.Market.INDIAN_MARKET, symbol="ADANIPORTS"
      ),
      "NIFTY_FIN": sid.IndexDesc(
        market=sid.Market.INDIAN_MARKET, symbol="NIFTY FIN SERVICE"
      ),
      "NIFTY_IT": sid.IndexDesc(market=sid.Market.INDIAN_MARKET, symbol="NIFTY IT"),
    }

  def test_three_descs(self):
    instrument_descs = self._create_instrument_descs()

    # Create timestamps for the same day (2023-01-02) at different times
    # 0 1 2 3 4 5
    base_date = dt.date(2023, 1, 2)
    all_timestamps = [
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=3, minute=45, second=59, microsecond=999000),
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=4, minute=15, second=30, microsecond=999000),
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=5, minute=30, second=15, microsecond=999000),
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=6, minute=45, second=45, microsecond=999000),
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=7, minute=0, second=0, microsecond=999000),
      pd.Timestamp(base_date)
      .tz_localize("UTC")
      .replace(hour=8, minute=30, second=30, microsecond=999000),
    ]

    instrument_desc_to_df = {
      instrument_descs["ADANIENT"]: self._create_test_dataframe(
        [
          all_timestamps[0],  # 03:45:59.999000
          # all_timestamps[1] missing (04:15:30.999000)
          all_timestamps[2],  # 05:30:15.999000
          all_timestamps[3],  # 06:45:45.999000
          all_timestamps[4],  # 07:00:00.999000
        ]
      ),
      instrument_descs["ADANIPORTS"]: self._create_test_dataframe(
        [
          all_timestamps[1],  # 04:15:30.999000
          # all_timestamps[2] missing (05:30:15.999000)
          all_timestamps[3],  # 06:45:45.999000
          # all_timestamps[4] missing (07:00:00.999000)
          all_timestamps[5],  # 08:30:30.999000
        ]
      ),
      instrument_descs["NIFTY_IT"]: self._create_test_dataframe(
        [
          all_timestamps[1],  # 04:15:30.999000
          # all_timestamps[2] missing (05:30:15.999000)
          # all_timestamps[3] missing (06:45:45.999000)
          # all_timestamps[4] missing (07:00:00.999000)
          all_timestamps[5],  # 08:30:30.999000
        ]
      ),
    }
    b = copy.deepcopy(instrument_desc_to_df)
    result = uzdc.count_missing_trading_bars(instrument_desc_to_df)
    _assert_instrument_desc_to_df_matches(instrument_desc_to_df, b)
    assert result.instrument_desc_to_missing_bars[instrument_descs["ADANIENT"]].equals(
      pd.DatetimeIndex([all_timestamps[1]])
    )
    assert result.instrument_desc_to_n_missing_bars[instrument_descs["ADANIENT"]] == 1
    assert result.instrument_desc_to_missing_bars[
      instrument_descs["ADANIPORTS"]
    ].equals(pd.DatetimeIndex([all_timestamps[2], all_timestamps[4]]))
    assert result.instrument_desc_to_n_missing_bars[instrument_descs["ADANIPORTS"]] == 2
    assert result.instrument_desc_to_missing_bars[instrument_descs["NIFTY_IT"]].equals(
      pd.DatetimeIndex([all_timestamps[2], all_timestamps[3], all_timestamps[4]])
    )
    assert result.instrument_desc_to_n_missing_bars[instrument_descs["NIFTY_IT"]] == 3
    assert result.all_timestamps.equals(pd.DatetimeIndex(all_timestamps))
