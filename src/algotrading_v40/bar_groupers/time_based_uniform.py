import dataclasses
import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.constants as ctnts
import algotrading_v40.trading_time_elapsed_calculators.with_overnight_gaps_only as ttec_wogo


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
  # tte: trading time elapsed
  tte = ttec_wogo.with_overnight_gaps_only(
    index=df.index,
    overnight_gap_minutes=overnight_gap_minutes,
  )
  tte += group_size_minutes - offset_minutes
  grouped_tte = group_size_minutes * (tte // group_size_minutes)
  group_start_positions = (
    pd.Series(np.arange(len(df)), index=df.index)
    .where(grouped_tte != grouped_tte.shift(1))
    .ffill()
    .fillna(0)
    .astype(int)
  )

  return GetTimeBasedUniformBarGroupForIndianMarketResult(
    bar_groups=pd.Series(
      df.index[group_start_positions], index=df.index, name="bar_group"
    ),
    offsets=pd.Series(tte % group_size_minutes, index=df.index),
  )
