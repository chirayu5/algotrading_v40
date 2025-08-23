import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.constants as ac


def get_daily_vol(close: pd.Series, span0: int, shift: int) -> pd.Series:
  """SNIPPET 3.1 DAILY VOLATILITY ESTIMATES"""
  dr = close / close.shift(shift) - 1
  dv = dr.ewm(span=span0).std()
  return dv


def get_bar_number_in_session(index: pd.DatetimeIndex) -> pd.Series:
  if len(index) == 0:
    return pd.Series([], dtype=int, index=index)

  first_minute_bar_time = ac.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  first_minute_bar_seconds = (
    first_minute_bar_time.hour * 3600
    + first_minute_bar_time.minute * 60
    + first_minute_bar_time.second
  )

  index_seconds = index.hour * 3600 + index.minute * 60 + index.second
  bar_numbers = (index_seconds - first_minute_bar_seconds) // 60
  return pd.Series(bar_numbers, index=index, dtype=int)


def get_indian_market_session_info(
  index: pd.DatetimeIndex,
) -> pd.DataFrame:
  """
  Get session information for a given index.
  The index must be timezone-aware (expected UTC).
  Logic is written with the assumption that the index is sampled at
  1 minute intervals with maybe some missing bars.
  """
  if not isinstance(index, pd.DatetimeIndex):
    raise ValueError("index must be a DatetimeIndex")
  if index.tz is None:
    raise ValueError("index must be timezone-aware (expected UTC)")
  if str(index.tz) != "UTC":
    raise ValueError("index must be in UTC timezone")

  index_times = index.time
  valid_time_range = (
    index_times >= ac.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ) & (index_times <= ac.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC)
  if not valid_time_range.all():
    invalid_times = index_times[~valid_time_range]
    raise ValueError(
      f"All times must be between "
      f"{ac.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC} and "
      f"{ac.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC} (inclusive). "
      f"Found invalid times: {invalid_times[:5].tolist()}"
    )

  df = pd.DataFrame(index=index)
  shifted_index = pd.Series(index).shift(1)
  delta_to_previous_index_minutes = (
    pd.Series(index) - shifted_index
  ).dt.total_seconds() // 60
  delta_to_previous_index_minutes.index = index

  df["is_first_bar_of_session"] = delta_to_previous_index_minutes.notna() & (
    delta_to_previous_index_minutes >= 1060
  )  # ~1066 minutes is the overnight gap
  df["is_first_bar_of_session"] = df["is_first_bar_of_session"] | (
    df.index.time == ac.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  )

  df["is_last_bar_of_session"] = (
    df.index.time == ac.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  )

  df["session_date"] = df.index.date
  df["session_id"] = df["is_first_bar_of_session"].cumsum()

  df["bar_number_in_session"] = get_bar_number_in_session(index=index)

  # Monday=0, Sunday=6
  df["weekday"] = df.index.weekday
  return df


######## BAR GROUPING CALCULATION #########


def get_time_based_bar_group_for_indian_market(
  df: pd.DataFrame, group_size_minutes: int
) -> pd.Series:
  # using a random date as it does not matter
  if not 1 <= group_size_minutes <= 375:
    raise ValueError(
      f"group_size_minutes must be between 1 and 375 (inclusive). "
      f"Got {group_size_minutes}"
    )

  # PRECOMPUTATION; does not depend on the input dataframe df
  reference = pd.date_range(
    start=dt.datetime.combine(
      dt.date(1990, 1, 1), ac.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
    ),
    periods=375,
    freq="1min",
  ).time
  ref2 = group_size_minutes * (np.arange(0, 375) // group_size_minutes)
  if (ref2 == ref2[-1]).sum() < (group_size_minutes / 2):
    # if the last group is less than half of the group size,
    # then set the last group to the second last group
    # this is to ensure that the last group is at least half of the group size
    ref2[ref2 == ref2[-1]] = ref2[-1] - group_size_minutes
  time_to_grouped_time = {t: reference[ref2[i]] for i, t in enumerate(reference)}
  # END PRECOMPUTATION

  bar_group = df.index.map(
    lambda x: dt.datetime.combine(x.date(), time_to_grouped_time[x.time()])
  )
  return pd.Series(bar_group, index=df.index)


#########################################
