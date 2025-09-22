import dataclasses
import datetime as dt

import pandas as pd

import algotrading_v40.constants as ctnts


@dataclasses.dataclass(frozen=True)
class GroupInfoForIndianMarketResult:
  first_ideal_ts: pd.Timestamp
  duration: int


def group_info_for_indian_market(
  ts: pd.Timestamp,
  group_size_minutes: int,
  offset_minutes: int,
  prev_date: dt.date,
):
  """
  Given a timestamp, identify the group it belongs to.
  Return the first ideal timestamp and duration of the group.
  """
  raise NotImplementedError("This function is deprecated. Do not use it.")
  if group_size_minutes <= 0 or group_size_minutes > 375:
    raise ValueError(
      f"group_size_minutes must be between 1 and 375 (inclusive). Got {group_size_minutes}"
    )

  remainder = 375 % group_size_minutes
  if remainder == 0:
    n_groups = 375 // group_size_minutes
    last_group_size = group_size_minutes
  else:
    if remainder < (group_size_minutes / 2):
      n_groups = 375 // group_size_minutes
      last_group_size = group_size_minutes + remainder
    else:
      n_groups = (375 // group_size_minutes) + 1
      last_group_size = remainder

  if offset_minutes < 0 or offset_minutes >= last_group_size:
    raise ValueError(
      f"offset_minutes must be between 0 and {last_group_size - 1} (inclusive). Got {offset_minutes}"
    )

  if (prev_date is not None) and (prev_date >= ts.date()):
    raise ValueError("prev_date must be strictly before ts.date()")

  # G G G ... G L
  # (n_groups-1) G of size group_size_minutes
  # last group L of size last_group_size

  # first minute bar close timestamp after applying the offset_minutes for the trading day
  fts: pd.Timestamp = pd.Timestamp.combine(
    ts.date(), ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC") + pd.Timedelta(minutes=offset_minutes)

  elapsed: int = (ts - fts).total_seconds() // 60

  if elapsed < 0:
    # belongs to the previous day's last group
    # prev fts: previous day's first minute bar close timestamp after applying the offset_minutes
    if prev_date is None:
      return GroupInfoForIndianMarketResult(
        first_ideal_ts=None,
        duration=None,
      )
    prev_fts: pd.Timestamp = pd.Timestamp.combine(
      prev_date, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
    ).tz_localize("UTC") + pd.Timedelta(minutes=offset_minutes)
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=prev_fts
      + pd.Timedelta(minutes=(n_groups - 1) * group_size_minutes),
      duration=last_group_size,
    )

  if elapsed < group_size_minutes * (n_groups - 1):
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=fts
      + pd.Timedelta(minutes=(elapsed // group_size_minutes) * group_size_minutes),
      duration=group_size_minutes,
    )
  else:
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=fts + pd.Timedelta(minutes=(n_groups - 1) * group_size_minutes),
      duration=last_group_size,
    )


def get_time_based_bar_group_for_indian_market(
  df: pd.DataFrame,
  group_size_minutes: int,
  offset_minutes: int,
) -> pd.Series:
  """
  Get the time-based bar group for the Indian market.

  For a trading day, it assumes
  - first minute bar close time is: 03:45:59.999000+00:00
  - last minute bar close time is: 09:59:59.999000+00:00
  (375 minutes in total)

  The `offset_minutes` parameter is used to shift the bar group by a certain number of minutes.

  Example without offset:
    If one trading day is:
    0 1 2 3 4 5 6 7 8 ... 371 372 373 374
    with group_size_minutes=4 and offset_minutes=0,
    then the bar groups are:
    0 0 0 0 | 4 4 4 4 | 8 ... 368 | 372 372 372 (last group has 3 bars since 375 % 4 = 3)

  Example with offset:
    If one trading day is:
    0 1 2 3 4 5 6 7 8 ... 371 372 373 374
    with group_size_minutes=4 and offset_minutes=1,
    then the bar groups are:
    pd373:(0) | 1:(1 2 3 4) | 5:(5 6 7 8) ... 369:(369 370 371 372) | 373:(373 374 |||| nd0) | nd1:(nd1 nd2 nd3 nd4)
    (pd: previous day, nd: next day)
    (last group will have one bar from the next day since offset_minutes is 1)

  Args:
    df: DataFrame with index as bar_close_timestamp (UTC timezone)
    offset_minutes: offset_minutes in minutes from the start of the day
    group_size_minutes: size of the bar group in minutes
  """
  raise NotImplementedError(
    "This function is deprecated. Use bg_tbu.get_time_based_uniform_bar_group_for_indian_market instead."
  )
  if group_size_minutes <= 0 or group_size_minutes > 375:
    raise ValueError(
      f"group_size_minutes must be between 1 and 375 (inclusive). Got {group_size_minutes}"
    )

  if (not df.index.is_unique) or (not df.index.is_monotonic_increasing):
    raise ValueError("df.index must be unique and monotonic increasing")
  if str(df.index.tz) != "UTC":  # type: ignore
    raise ValueError("DataFrame index must have UTC timezone")

  n = len(df)
  bg = [None] * n

  bg[0] = df.index[0]
  curr_date = df.index[0].date()
  prev_date = curr_date - dt.timedelta(days=1)
  gi = group_info_for_indian_market(
    ts=df.index[0],
    group_size_minutes=group_size_minutes,
    offset_minutes=offset_minutes,
    prev_date=prev_date,
  )
  cgfits = gi.first_ideal_ts
  cgd = gi.duration
  cgfrts = df.index[0]

  for i in range(1, n):
    curr_ts = df.index[i]
    # cgfits: current group's first IDEAL time stamp
    # cgfrts: current group's first REAL time stamp (can be different from IDEAL if there are missing bars)
    # cgd: current group's duration

    if curr_ts.date() != curr_date:
      prev_date = curr_date
      curr_date = curr_ts.date()

    def _diff_minutes(ts1, ts2):
      if ts1 > ts2:
        raise ValueError("ts1 must be before ts2")

      if ts1.date() == ts2.date():
        return (ts2 - ts1).total_seconds() // 60

      # ts1 and ts2 are on different days
      # part1: minutes from ts1 to the end of the trading session
      part1 = (
        pd.Timestamp.combine(
          ts1.date(), ctnts.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
        ).tz_localize("UTC")
        - ts1
      ).total_seconds() // 60
      # part2: minutes from the start of the trading session to ts2
      part2 = (
        1
        + (
          ts2
          - pd.Timestamp.combine(
            ts2.date(), ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
          ).tz_localize("UTC")
        ).total_seconds()
        // 60
      )
      return part1 + part2

    if _diff_minutes(ts1=cgfits, ts2=curr_ts) <= cgd - 1:
      bg[i] = cgfrts
    else:
      cgfrts = curr_ts
      bg[i] = curr_ts
      gi = group_info_for_indian_market(
        ts=curr_ts,
        group_size_minutes=group_size_minutes,
        offset_minutes=offset_minutes,
        prev_date=prev_date,
      )
      cgfits = gi.first_ideal_ts
      cgd = gi.duration

  return pd.Series(bg, index=df.index)
