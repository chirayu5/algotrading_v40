"""Functions to clean Zerodha Kite historical data."""

import dataclasses
import datetime as dt
from typing import Callable

import numpy as np
import pandas as pd
import pytz

import algotrading_v40.structures.instrument_desc as sid


@dataclasses.dataclass(frozen=True)
class FixUnusualBarsResult:
  df: pd.DataFrame
  df_original_dropped: pd.DataFrame
  df_original_date_needs_fix: pd.DataFrame

  @property
  def n_dropped(self) -> int:
    return len(self.df_original_dropped)

  @property
  def n_date_fixed(self) -> int:
    return len(self.df_original_date_needs_fix)


def fix_unusual_bars(
  df: pd.DataFrame,
) -> FixUnusualBarsResult:
  df = df.copy()
  if df["date"].duplicated().any():
    raise ValueError("DataFrame contains duplicate dates")

  if not df["date"].is_monotonic_increasing:
    raise ValueError("DataFrame dates are not in strictly ascending order")

  df["rounded_date"] = df["date"].dt.round("min")
  rounded_date_same_as_date = df["rounded_date"] == df["date"]

  df_unusual = df.loc[~rounded_date_same_as_date]
  if df_unusual["rounded_date"].duplicated(keep=False).any():
    raise ValueError("Unusual bars have duplicate rounded dates")

  rounded_date_found_in_date_column = df["rounded_date"].isin(df["date"])
  df_original_dropped = df.loc[
    (~rounded_date_same_as_date) & (rounded_date_found_in_date_column)
  ]
  df = df.loc[rounded_date_same_as_date | (~rounded_date_found_in_date_column)].copy()

  date_needs_fix = df["date"] != df["rounded_date"]
  df_original_date_needs_fix = df.loc[date_needs_fix].copy()
  df.loc[date_needs_fix, "date"] = df.loc[date_needs_fix, "rounded_date"]
  df = df.drop(columns=["rounded_date"])
  return FixUnusualBarsResult(
    df=df,
    df_original_dropped=df_original_dropped,
    df_original_date_needs_fix=df_original_date_needs_fix,
  )


def set_index_to_bar_close_timestamp(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["date"] = df["date"] + pd.Timedelta(seconds=59.999)
  df["date"] = df["date"].dt.tz_convert("UTC")
  df = df.set_index("date", drop=True)
  return df


@dataclasses.dataclass(frozen=True)
class DropNonStandardIndianTradingHoursResult:
  df: pd.DataFrame
  df_original_dropped: pd.DataFrame

  @property
  def n_dropped(self) -> int:
    return len(self.df_original_dropped)


def drop_non_standard_indian_trading_hours(
  df: pd.DataFrame,
) -> DropNonStandardIndianTradingHoursResult:
  if str(df.index.tz) != "UTC":  # type: ignore
    raise ValueError("DataFrame index must have UTC timezone")

  # In UTC, Indian market hours are
  # 03:45:59.999000+00:00
  # 09:59:59.999000+00:00
  ist = pytz.timezone("Asia/Kolkata")
  df_ist = df.copy()
  df_ist.index = df_ist.index.tz_convert(ist)  # type: ignore

  start_time_ist = dt.time(9, 15, 59, 999000)
  end_time_ist = dt.time(15, 29, 59, 999000)

  mask = (df_ist.index.time >= start_time_ist) & (df_ist.index.time <= end_time_ist)  # type: ignore
  return DropNonStandardIndianTradingHoursResult(
    df=df.loc[mask],
    df_original_dropped=df.loc[~mask],
  )


@dataclasses.dataclass(frozen=True)
class FixHighLowValuesResult:
  df: pd.DataFrame
  df_original_high_needs_fix: pd.DataFrame
  df_original_low_needs_fix: pd.DataFrame

  @property
  def n_high_fixed(self) -> int:
    return len(self.df_original_high_needs_fix)

  @property
  def n_low_fixed(self) -> int:
    return len(self.df_original_low_needs_fix)


def fix_high_low_values(
  df: pd.DataFrame,
) -> FixHighLowValuesResult:
  df = df.copy()
  df_original = df.copy()
  correct_high = df[["open", "high", "low", "close"]].max(axis=1)
  correct_low = df[["open", "high", "low", "close"]].min(axis=1)

  high_needs_fix = df["high"] != correct_high
  low_needs_fix = df["low"] != correct_low
  df.loc[high_needs_fix, "high"] = correct_high[high_needs_fix]
  df.loc[low_needs_fix, "low"] = correct_low[low_needs_fix]

  return FixHighLowValuesResult(
    df=df,
    df_original_high_needs_fix=df_original.loc[high_needs_fix],
    df_original_low_needs_fix=df_original.loc[low_needs_fix],
  )


@dataclasses.dataclass(frozen=True)
class CountBarsPerTradingDayResult:
  df: pd.DataFrame
  n_dates: int
  dates_with_less_than_375_bars: list[dt.date]

  @property
  def n_dates_with_less_than_375_bars(self) -> int:
    return len(self.dates_with_less_than_375_bars)

  @property
  def fraction_dates_with_less_than_375_bars(self) -> float:
    return (
      self.n_dates_with_less_than_375_bars / self.n_dates if self.n_dates > 0 else 0.0
    )


def count_bars_per_trading_day(
  df: pd.DataFrame,
) -> CountBarsPerTradingDayResult:
  date_counts = df.groupby(df.index.date).size()  # type: ignore
  dates_with_less_than_375_bars = list(date_counts[date_counts < 375].index)

  result_df = pd.DataFrame(
    {"date": date_counts.index, "count": date_counts.values}
  ).set_index("date")

  return CountBarsPerTradingDayResult(
    df=result_df,
    n_dates=len(date_counts),
    dates_with_less_than_375_bars=dates_with_less_than_375_bars,
  )


@dataclasses.dataclass(frozen=True)
class CountMissingTradingSessionsResult:
  instrument_desc_to_missing_sessions: dict[sid.InstrumentDesc, np.ndarray[dt.date]]
  all_dates: np.ndarray[dt.date]

  # def __post_init__(self):
  #   for instrument_desc, sessions in self.instrument_desc_to_missing_sessions.items():
  #     if not isinstance(sessions, np.ndarray):
  #       raise TypeError(
  #         f"Expected np.ndarray for missing sessions, got {type(sessions)}"
  #       )
  #     if sessions.dtype != object or (
  #       len(sessions) > 0 and not isinstance(sessions[0], dt.date)
  #     ):
  #       raise TypeError("Expected np.ndarray of dt.date objects for missing sessions")

  #   if not isinstance(self.all_dates, np.ndarray):
  #     raise TypeError(f"Expected np.ndarray for all_dates, got {type(self.all_dates)}")
  #   if self.all_dates.dtype != object or (
  #     len(self.all_dates) > 0 and not isinstance(self.all_dates[0], dt.date)
  #   ):
  #     raise TypeError("Expected np.ndarray of dt.date objects for all_dates")

  @property
  def instrument_desc_to_n_missing_sessions(self) -> dict[sid.InstrumentDesc, int]:
    return {
      instrument_desc: len(sessions)
      for instrument_desc, sessions in self.instrument_desc_to_missing_sessions.items()
    }


@dataclasses.dataclass(frozen=True)
class CountMissingTradingBarsResult:
  instrument_desc_to_missing_bars: dict[sid.InstrumentDesc, pd.DatetimeIndex]
  all_timestamps: pd.DatetimeIndex

  # def __post_init__(self):
  #   for instrument_desc, bars in self.instrument_desc_to_missing_bars.items():
  #     if not isinstance(bars, pd.DatetimeIndex):
  #       raise TypeError(f"Expected pd.DatetimeIndex for missing bars, got {type(bars)}")
  #   if not isinstance(self.all_timestamps, pd.DatetimeIndex):
  #     raise TypeError(
  #       f"Expected pd.DatetimeIndex for all_timestamps, got {type(self.all_timestamps)}"
  #     )

  @property
  def instrument_desc_to_n_missing_bars(self) -> dict[sid.InstrumentDesc, int]:
    return {
      instrument_desc: len(bars)
      for instrument_desc, bars in self.instrument_desc_to_missing_bars.items()
    }


def _count_missing_trading_items(
  instrument_desc_to_df: dict[sid.InstrumentDesc, pd.DataFrame],
  index_extractor: Callable[[pd.DataFrame], pd.DatetimeIndex],
) -> tuple[dict[sid.InstrumentDesc, pd.DatetimeIndex], pd.DatetimeIndex]:
  """Generic function to count missing trading items (sessions or bars)."""
  # Collect all DatetimeIndexes and union them
  all_items = pd.DatetimeIndex([])
  for df in instrument_desc_to_df.values():
    df_items = index_extractor(df)
    all_items = all_items.union(df_items)
  all_items = pd.DatetimeIndex(all_items)
  instrument_desc_to_missing_items = {}
  for instrument_desc, df in instrument_desc_to_df.items():
    df_items = index_extractor(df)
    df_min_item = df_items.min()
    df_max_item = df_items.max()

    # Filter all_items to only include items within the instrument's range
    applicable_all_items = all_items[
      (all_items >= df_min_item) & (all_items <= df_max_item)
    ]

    # Find missing items by getting the difference
    missing_items = applicable_all_items.difference(df_items)
    instrument_desc_to_missing_items[instrument_desc] = missing_items.sort_values()

  return instrument_desc_to_missing_items, all_items.sort_values()


def count_missing_trading_sessions(
  instrument_desc_to_df: dict[sid.InstrumentDesc, pd.DataFrame],
) -> CountMissingTradingSessionsResult:
  missing_items, all_items = _count_missing_trading_items(
    instrument_desc_to_df, lambda df: pd.to_datetime(df.index.date)
  )
  missing_items = {id: v.date for id, v in missing_items.items()}
  return CountMissingTradingSessionsResult(
    instrument_desc_to_missing_sessions=missing_items,
    all_dates=all_items.date,
  )


def count_missing_trading_bars(
  instrument_desc_to_df: dict[sid.InstrumentDesc, pd.DataFrame],
) -> CountMissingTradingBarsResult:
  missing_items, all_items = _count_missing_trading_items(
    instrument_desc_to_df, lambda df: df.index
  )
  return CountMissingTradingBarsResult(
    instrument_desc_to_missing_bars=missing_items,
    all_timestamps=all_items,
  )


@dataclasses.dataclass(frozen=True)
class AnalyzeSeriesQualityResult:
  n_bad_values: int
  n_zeros: int
  n_negatives: int
  n_bad_values_at_start: int
  n_bad_values_at_end: int


def analyze_numeric_series_quality(s: pd.Series) -> AnalyzeSeriesQualityResult:
  import numpy as np

  if not pd.api.types.is_numeric_dtype(s):
    raise ValueError("Series must be numeric")

  bad_mask = ~np.isfinite(s.to_numpy())
  good_mask = ~bad_mask

  n_bad_values = bad_mask.sum()

  good_values = s[good_mask]
  n_zeros = (good_values == 0).sum()
  n_negatives = (good_values < 0).sum()

  cumsum_good = (~bad_mask).cumsum()
  n_bad_values_at_start = (cumsum_good == 0).sum()

  cumsum_good_reversed = (~bad_mask[::-1]).cumsum()
  n_bad_values_at_end = (cumsum_good_reversed == 0).sum()

  return AnalyzeSeriesQualityResult(
    n_bad_values=int(n_bad_values),
    n_zeros=int(n_zeros),
    n_negatives=int(n_negatives),
    n_bad_values_at_start=int(n_bad_values_at_start),
    n_bad_values_at_end=int(n_bad_values_at_end),
  )


def analyse_numeric_columns_quality(
  df: pd.DataFrame,
) -> dict[str, AnalyzeSeriesQualityResult]:
  result = {}
  for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
      continue
    result[col] = analyze_numeric_series_quality(df[col])  # type: ignore
  return result
