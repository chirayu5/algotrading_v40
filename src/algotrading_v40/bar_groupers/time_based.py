import dataclasses

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
):
  """
  Given a timestamp, identify the group it belongs to.
  Return the first ideal timestamp and duration of the group.
  """
  if offset != 0:
    raise ValueError("offset must be 0")
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

  # G G G ... G L
  # (n_groups-1) G of size group_size
  # last group L of size last_group_size
  if group_size * (n_groups - 1) + last_group_size != 375:
    raise ValueError(
      f"group_size * (n_groups-1) + last_group_size must be equal to 375. Got {group_size * (n_groups - 1) + last_group_size}"
    )

  # first minute bar close timestamp after applying the offset for the trading day
  fts: pd.Timestamp = pd.Timestamp.combine(
    ts.date(), ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC") + pd.Timedelta(minutes=offset)

  elapsed: int = (ts - fts).total_seconds()
  if elapsed % 60 != 0:
    raise ValueError(f"elapsed must be a multiple of 60. Got {elapsed}")
  elapsed = elapsed // 60

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
  offset: int = 0,
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
  if offset != 0:
    raise ValueError("offset must be 0")
  if group_size <= 0 or group_size > 375:
    raise ValueError(
      f"group_size must be between 1 and 375 (inclusive). Got {group_size}"
    )

  n = len(df)
  bg = [None] * n

  bg[0] = df.index[0]
  gi = group_info_for_indian_market(
    ts=df.index[0], group_size=group_size, offset=offset
  )
  cgfits = gi.first_ideal_ts
  cgd = gi.duration
  del gi
  cgfrts = df.index[0]

  for i in range(1, n):
    curr_ts = df.index[i]
    # cgfits: current group's first IDEAL time stamp
    # cgfrts: current group's first REAL time stamp (can be different from IDEAL if there are missing bars)
    # cgd: current group's duration
    if curr_ts - cgfits <= pd.Timedelta(minutes=cgd - 1):
      bg[i] = cgfrts
    else:
      cgfrts = curr_ts
      bg[i] = curr_ts
      gi = group_info_for_indian_market(
        ts=curr_ts, group_size=group_size, offset=offset
      )
      if gi.first_ideal_ts == cgfits:
        raise ValueError("Unexpected")
      cgfits = gi.first_ideal_ts
      cgd = gi.duration
      del gi

  return pd.Series(bg, index=df.index)
