import dataclasses

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
class AnalyzeSeriesQualityResult:
  n_bad_values: int
  n_zeros: int
  n_negatives: int
  n_bad_values_at_start: int
  n_bad_values_at_end: int


def analyze_numeric_series_quality(s: pd.Series) -> AnalyzeSeriesQualityResult:
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
