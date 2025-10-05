import algotrading_v40_cpp.trading_time_elapsed_calculators as av40c_ttec
import pandas as pd

import algotrading_v40.constants as ctnts


def _validate_inputs(
  index: pd.DatetimeIndex,
  overnight_gap_minutes: int,
):
  if index.tz is None:
    raise ValueError("index must be timezone-aware (expected UTC)")
  if str(index.tz) != "UTC":
    raise ValueError("index must be in UTC timezone")

  if overnight_gap_minutes < 0:
    # a zero overnight gap is considered valid as it can be used for
    # the case of continously trading markets (e.g. BTCUSDT Perpetual Futures on Binance)
    raise ValueError("overnight_gap_minutes must be non-negative")


def with_overnight_gaps_only(
  index: pd.DatetimeIndex,
  overnight_gap_minutes: int,
) -> pd.Series:
  _validate_inputs(index, overnight_gap_minutes)

  if len(index) == 0:
    return pd.Series(index=index)

  # Convert index to minutes since first row.
  minutes_from_first_data_point = index.astype(int)
  minutes_from_first_data_point = (
    minutes_from_first_data_point - minutes_from_first_data_point[0]
  ) // ctnts.NANOS_PER_MINUTE

  # Day offsets relative to first row's calendar date.
  days_from_first_day = pd.Series(index.date - index.date[0]).dt.days.values

  return pd.Series(
    av40c_ttec.with_overnight_gaps_only_cpp(
      minutes_from_first_data_point,
      days_from_first_day,
      overnight_gap_minutes,
    ),
    index=index,
  ).astype(int)
