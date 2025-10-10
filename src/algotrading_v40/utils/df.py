import dataclasses
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

import algotrading_v40.structures.date_range as sdr


def get_df_slice_in_date_range(
  df: pd.DataFrame,
  date_range: sdr.DateRange,
) -> pd.DataFrame:
  if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("df.index must be a DatetimeIndex")
  if str(df.index.tz) != "UTC":  # type: ignore
    raise ValueError("DataFrame index must have UTC timezone")
  return df[
    (df.index.date >= date_range.start_date) & (df.index.date <= date_range.end_date)
  ]


@dataclasses.dataclass(frozen=True)
class AnalyseSeriesQualityResult:
  n_zeros: int
  n_negatives: int
  n_bad_values_at_start: int
  n_bad_values_at_end: int
  good_values_mask: pd.Series
  n_values: int

  @property
  def n_bad_values(self) -> int:
    return self.n_values - self.n_good_values

  @property
  def n_good_values(self) -> int:
    return int(self.good_values_mask.sum())

  @property
  def has_bad_values(self) -> bool:
    return self.n_bad_values > 0

  @property
  def has_bad_values_apart_from_start(self) -> bool:
    return self.n_bad_values_at_start != self.n_bad_values

  @property
  def has_bad_values_apart_from_end(self) -> bool:
    return self.n_bad_values_at_end != self.n_bad_values


def analyse_numeric_series_quality(s: pd.Series) -> AnalyseSeriesQualityResult:
  if not pd.api.types.is_numeric_dtype(s):
    raise ValueError("Series must be numeric")

  bad_values_mask = ~np.isfinite(s.to_numpy())
  good_values_mask = ~bad_values_mask

  good_values = s[good_values_mask]
  n_zeros = (good_values == 0).sum()
  n_negatives = (good_values < 0).sum()

  cumsum_good = (~bad_values_mask).cumsum()
  n_bad_values_at_start = (cumsum_good == 0).sum()

  cumsum_good_reversed = (~bad_values_mask[::-1]).cumsum()
  n_bad_values_at_end = (cumsum_good_reversed == 0).sum()

  return AnalyseSeriesQualityResult(
    n_zeros=int(n_zeros),
    n_negatives=int(n_negatives),
    n_bad_values_at_start=int(n_bad_values_at_start),
    n_bad_values_at_end=int(n_bad_values_at_end),
    good_values_mask=pd.Series(good_values_mask, index=s.index),
    n_values=len(s),
  )


def analyse_numeric_columns_quality(
  df: pd.DataFrame,
) -> dict[str, AnalyseSeriesQualityResult]:
  result = {}
  for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
      continue
    result[col] = analyse_numeric_series_quality(df[col])  # type: ignore
  return result


@dataclasses.dataclass(frozen=True)
class GetMostCommonIndexDeltaResult:
  most_common_index_delta: int | None  # in minutes
  index_delta_distribution: pd.Series


def get_most_common_index_delta(
  index: pd.DatetimeIndex,
) -> GetMostCommonIndexDeltaResult:
  """Get the most common index delta in minutes."""
  if len(index) <= 1:
    return GetMostCommonIndexDeltaResult(
      most_common_index_delta=None,
      index_delta_distribution=pd.Series([]),
    )
  index_deltas = ((index[1:] - index[:-1]).total_seconds() // 60).astype(int)
  vcs = index_deltas.value_counts(dropna=True).sort_values(ascending=False)
  return GetMostCommonIndexDeltaResult(
    most_common_index_delta=int(vcs.index[0]),
    index_delta_distribution=vcs,
  )


############# DATA QUALITY VALIDATORS ##############
def check_indices_match(*series: pd.Series) -> None:
  """Check that all series have the same index."""
  if len(series) == 0:
    return

  first_index = series[0].index
  for i, s in enumerate(series[1:], start=1):
    if not first_index.equals(s.index):
      raise ValueError(
        f"Series at position {i} has different index than series at position 0"
      )


def check_index_u_and_mi(index: pd.Index) -> None:
  """Check that index is unique and monotonically increasing."""
  # Check uniqueness - vectorized
  if not index.is_unique:
    raise ValueError("Index must be unique")

  # Check monotonic increasing - vectorized
  if not index.is_monotonic_increasing:
    raise ValueError("Index must be monotonically increasing")


def check_no_bad_values(*series: pd.Series) -> None:
  """Check that all series have no bad values (NaN, inf)."""
  for i, s in enumerate(series):
    result = analyse_numeric_series_quality(s)
    if result.has_bad_values:
      raise ValueError(
        f"Series at position {i} has {result.n_bad_values} bad values "
        f"(NaN or inf) out of {result.n_values} total values"
      )


def check_no_bad_values_apart_from_start(*series: pd.Series) -> None:
  """Check that all series have no bad values except at the start."""
  for i, s in enumerate(series):
    result = analyse_numeric_series_quality(s)
    if result.has_bad_values_apart_from_start:
      n_bad_apart_from_start = result.n_bad_values - result.n_bad_values_at_start
      raise ValueError(
        f"Series at position {i} has {n_bad_apart_from_start} bad values "
        f"(NaN or inf) apart from the start out of {result.n_values} total values"
      )


def check_all_gt0(*series: pd.Series) -> None:
  """Check that ALL values in all series are greater than 0. Throws if bad values present."""
  for i, s in enumerate(series):
    result = analyse_numeric_series_quality(s)

    # First check for bad values
    if result.has_bad_values:
      raise ValueError(
        f"Series at position {i} has {result.n_bad_values} bad values "
        f"(NaN or inf) out of {result.n_values} total values"
      )

    # Then check all values are > 0 (vectorized)
    arr = s.to_numpy()
    if not np.all(arr > 0):
      n_violations = np.sum(arr <= 0)
      raise ValueError(
        f"Series at position {i} has {n_violations} values that are <= 0"
      )


def check_all_gte0(*series: pd.Series) -> None:
  """Check that ALL values in all series are >= 0. Throws if bad values present."""
  for i, s in enumerate(series):
    result = analyse_numeric_series_quality(s)

    # First check for bad values
    if result.has_bad_values:
      raise ValueError(
        f"Series at position {i} has {result.n_bad_values} bad values "
        f"(NaN or inf) out of {result.n_values} total values"
      )

    # Then check all values are >= 0 (vectorized)
    arr = s.to_numpy()
    if not np.all(arr >= 0):
      n_violations = np.sum(arr < 0)
      raise ValueError(f"Series at position {i} has {n_violations} values that are < 0")


def check_all_lt0(*series: pd.Series) -> None:
  """Check that ALL values in all series are < 0. Throws if bad values present."""
  for i, s in enumerate(series):
    result = analyse_numeric_series_quality(s)

    # First check for bad values
    if result.has_bad_values:
      raise ValueError(
        f"Series at position {i} has {result.n_bad_values} bad values "
        f"(NaN or inf) out of {result.n_values} total values"
      )

    # Then check all values are < 0 (vectorized)
    arr = s.to_numpy()
    if not np.all(arr < 0):
      n_violations = np.sum(arr >= 0)
      raise ValueError(
        f"Series at position {i} has {n_violations} values that are >= 0"
      )


def check_all_in(series: pd.Series, values: tuple) -> None:
  """Check that ALL values in series are in the given set. Throws if bad values present."""
  result = analyse_numeric_series_quality(series)

  # First check for bad values
  if result.has_bad_values:
    raise ValueError(
      f"Series has {result.n_bad_values} bad values "
      f"(NaN or inf) out of {result.n_values} total values"
    )

  # Then check all values are in the allowed set (vectorized)
  arr = series.to_numpy()
  values_array = np.array(values)
  if not np.all(np.isin(arr, values_array)):
    n_violations = np.sum(~np.isin(arr, values_array))
    unique_bad = np.unique(arr[~np.isin(arr, values_array)])
    raise ValueError(
      f"Series has {n_violations} values not in {values}. "
      f"Found unexpected values: {unique_bad.tolist()}"
    )


#################################################

############## GROUPING BY BAR GROUP ##############
# Allows working with x minute bars instead of only 1 minute bars
# Also allows using dollar bars


@dataclasses.dataclass(frozen=True)
class GroupByBarGroupResult:
  df: pd.DataFrame
  bar_group_size: pd.Series


def group_by_bar_group(df: pd.DataFrame, bar_group: pd.Series) -> GroupByBarGroupResult:
  if not bar_group.index.equals(df.index):
    raise ValueError("bar_group index must be equal to df index")
  dfg = df.groupby(by=bar_group).agg(
    {
      "open": "first",
      "high": "max",
      "low": "min",
      "close": "last",
      "volume": "sum",
    }
  )
  dfg.index.name = "bar_group"
  bar_group_size = df.groupby(by=bar_group).size().astype("int32")
  bar_group_size.index.name = "bar_group"

  return GroupByBarGroupResult(
    df=dfg,
    bar_group_size=bar_group_size,
  )


#################################################

#################### CALCULATE GROUPED VERSION OF VALUES ############


def calculate_grouped_values(
  df: pd.DataFrame,
  bar_group: pd.Series,
  compute_func: Callable[[pd.DataFrame], pd.DataFrame]
  | list[Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
  """Calculate grouped values using one or more compute functions.

  Args:
    df: DataFrame with a 'bar_group' column
    compute_func: Either a single callable or a list of callables.
                  Each callable takes a grouped DataFrame and returns a DataFrame.

  Returns:
    DataFrame with all computed columns. When multiple functions are provided,
    results are concatenated into a single DataFrame.

  Raises:
    ValueError: If functions return DataFrames with overlapping column names.

  Examples:
    # Single function (backward compatible)
    result = calculate_grouped_values(df, my_compute_func)

    # Multiple functions (efficient - groups only once)
    result = calculate_grouped_values(df, [
      compute_func1,
      compute_func2,
      compute_func3,
    ])
  """
  dfg = group_by_bar_group(df, bar_group=bar_group)
  if not isinstance(compute_func, list):
    compute_func = [compute_func]
  if len(compute_func) == 0:
    raise ValueError("compute_func list cannot be empty")

  results = [func(dfg.df).reindex(df.index) for func in compute_func]

  all_columns = []
  for result in results:
    all_columns.extend(result.columns.tolist())

  if len(all_columns) != len(set(all_columns)):
    column_counts = Counter(all_columns)
    duplicates = [col for col, count in column_counts.items() if count > 1]
    raise ValueError(f"Duplicate columns found across compute functions: {duplicates}")

  # Concatenate all results (indices are guaranteed to be equal due to reindex)
  return pd.concat(results, axis=1)
