import dataclasses
import datetime as dt

import algotrading_v40_cpp.bar_groupers as av40c_bg
import pandas as pd

import algotrading_v40.constants as ctnts


@dataclasses.dataclass(frozen=True)
class GetTimeBasedUniformBarGroupForIndianMarketResult:
  bar_groups: pd.Series
  offsets: pd.Series


def get_time_based_uniform_bar_group_for_indian_market(
  df: pd.DataFrame,
  *,
  group_size_minutes: int,
  offset_minutes: int,
) -> GetTimeBasedUniformBarGroupForIndianMarketResult:
  if offset_minutes < 0 or offset_minutes >= group_size_minutes:
    raise ValueError(
      f"offset_minutes must be between 0 and {group_size_minutes - 1} (inclusive), "
      f"got {offset_minutes}"
    )

  if (not df.index.is_unique) or (not df.index.is_monotonic_increasing):
    raise ValueError("df.index must be unique and monotonic increasing")
  if str(df.index.tz) != "UTC":  # type: ignore
    raise ValueError("DataFrame index must have UTC timezone")

  if len(df) == 0:
    return GetTimeBasedUniformBarGroupForIndianMarketResult(
      bar_groups=pd.Series(index=df.index),
      offsets=pd.Series(index=df.index),
    )
  # Convert index to "minutes since first row"
  minutes_from_first_row = df.index.astype(int)
  minutes_from_first_row = (
    minutes_from_first_row - minutes_from_first_row[0]
  ) // ctnts.NANOS_PER_MINUTE

  # Day offsets relative to first row's calendar date
  day_offsets = pd.Series(df.index.date - df.index.date[0]).dt.days.values

  overnight_gap_minutes = int(
    (
      pd.Timestamp.combine(
        dt.date(2025, 5, 12),
        ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC,
      ).tz_localize("UTC")
      - pd.Timestamp.combine(
        dt.date(2025, 5, 11),
        ctnts.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC,
      ).tz_localize("UTC")
    ).total_seconds()
    // 60
    - 1
  )
  # overnight_gap_minutes is 1065 for standard Indian-market timings

  out = av40c_bg.time_based_uniform_cpp(
    minutes_from_first_row,
    day_offsets,
    group_size_minutes,
    offset_minutes,
    overnight_gap_minutes,
  )
  group_start_positions = out["group_start_positions"]
  offsets = out["offsets"]

  return GetTimeBasedUniformBarGroupForIndianMarketResult(
    bar_groups=pd.Series(
      df.index[group_start_positions], index=df.index, name="bar_group"
    ),
    offsets=pd.Series(offsets, index=df.index),
  )
