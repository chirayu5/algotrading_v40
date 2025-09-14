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
  group_size: int,
  offset: int,
  prev_date: dt.date,
):
  """
  Given a timestamp, identify the group it belongs to.
  Return the first ideal timestamp and duration of the group.
  """
  if group_size <= 0 or group_size > 375:
    raise ValueError(
      f"group_size must be between 1 and 375 (inclusive). Got {group_size}"
    )

  remainder = 375 % group_size
  if remainder == 0:
    n_groups = 375 // group_size
    last_group_size = group_size
  else:
    if remainder < (group_size / 2):
      n_groups = 375 // group_size
      last_group_size = group_size + remainder
    else:
      n_groups = (375 // group_size) + 1
      last_group_size = remainder

  if offset < 0 or offset >= last_group_size:
    raise ValueError(
      f"offset must be between 0 and {last_group_size - 1} (inclusive). Got {offset}"
    )

  if (prev_date is not None) and (prev_date >= ts.date()):
    raise ValueError("prev_date must be strictly before ts.date()")

  # G G G ... G L
  # (n_groups-1) G of size group_size
  # last group L of size last_group_size

  # first minute bar close timestamp after applying the offset for the trading day
  fts: pd.Timestamp = pd.Timestamp.combine(
    ts.date(), ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC") + pd.Timedelta(minutes=offset)

  elapsed: int = (ts - fts).total_seconds() // 60

  if elapsed < 0:
    # belongs to the previous day's last group
    # prev fts: previous day's first minute bar close timestamp after applying the offset
    if prev_date is None:
      return GroupInfoForIndianMarketResult(
        first_ideal_ts=None,
        duration=None,
      )
    prev_fts: pd.Timestamp = pd.Timestamp.combine(
      prev_date, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
    ).tz_localize("UTC") + pd.Timedelta(minutes=offset)
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=prev_fts + pd.Timedelta(minutes=(n_groups - 1) * group_size),
      duration=last_group_size,
    )

  if elapsed < group_size * (n_groups - 1):
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=fts + pd.Timedelta(minutes=(elapsed // group_size) * group_size),
      duration=group_size,
    )
  else:
    return GroupInfoForIndianMarketResult(
      first_ideal_ts=fts + pd.Timedelta(minutes=(n_groups - 1) * group_size),
      duration=last_group_size,
    )


def get_time_based_bar_group_for_indian_market(
  df: pd.DataFrame,
  group_size: int,
  offset: int,
) -> pd.Series:
  """
  Get the time-based bar group for the Indian market.

  For a trading day, it assumes
  - first minute bar close time is: 03:45:59.999000+00:00
  - last minute bar close time is: 09:59:59.999000+00:00
  (375 minutes in total)


  The `offset` parameter is used to shift the bar group by a certain number of minutes.

  Args:
    df: DataFrame with index as bar_close_timestamp (UTC timezone)
    offset: offset in minutes from the start of the day
    group_size: size of the bar group in minutes
  """
  if group_size <= 0 or group_size > 375:
    raise ValueError(
      f"group_size must be between 1 and 375 (inclusive). Got {group_size}"
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
    group_size=group_size,
    offset=offset,
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
        group_size=group_size,
        offset=offset,
        prev_date=prev_date,
      )
      cgfits = gi.first_ideal_ts
      cgd = gi.duration

  return pd.Series(bg, index=df.index)
