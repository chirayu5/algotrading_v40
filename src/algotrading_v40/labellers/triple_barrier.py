import numpy as np
import pandas as pd

import algotrading_v40.utils.df as udf


def _validate_inputs(
  s: pd.Series,
  selected: pd.Series,  # whether to run the search on this index
  tpb: pd.Series,  # take profit barriers
  slb: pd.Series,  # stop loss barriers
  vb: pd.Series,  # absolute integer index of vertical barriers
  side: pd.Series,  # 1 for long bet, -1 for short bet
):
  series_list = [selected, tpb, slb, vb, side]
  for series in series_list:
    if not s.index.equals(series.index):
      raise ValueError("All series must have the same index")
  n = len(s)

  for series, name in [
    (s, "s"),
    (selected, "selected"),
    (tpb, "tpb"),
    (slb, "slb"),
    (vb, "vb"),
    (side, "side"),
  ]:
    if udf.analyse_numeric_series_quality(series).n_bad_values != 0:
      raise ValueError(f"{name} must not have bad values")

  for series, name in [(selected, "selected"), (vb, "vb"), (side, "side")]:
    if not pd.api.types.is_integer_dtype(series):
      raise TypeError(f"{name} must be integer type")

  if not set(side.unique()).issubset({1, -1}):
    raise ValueError("side must only contain values 1 or -1")
  if not set(selected.unique()).issubset({0, 1}):
    raise ValueError("selected must only contain values 0 or 1")

  if not np.all(vb.values >= np.arange(n)):
    raise ValueError(
      "All vertical barriers must be greater than or equal to their index"
    )

  if not np.all(tpb > 0):
    raise ValueError("All take profit barriers must be greater than 0")
  if not np.all(slb < 0):
    raise ValueError("All stop loss barriers must be less than 0")


def _triple_barrier(
  s: np.ndarray,
  selected: np.ndarray,  # whether to run the search on this index
  tpb: np.ndarray,  # take profit barriers
  slb: np.ndarray,  # stop loss barriers
  vb: np.ndarray,  # absolute integer index of vertical barriers
  side: np.ndarray,  # 1 for long bet, -1 for short bet
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  n = len(s)
  # take profit hit at
  # stop loss hit at
  # vertical barrier hit at
  tpha, slha, vbha, first_touch_type, first_touch_return = (
    np.empty(n, dtype=object),
    np.empty(n, dtype=object),
    np.empty(n, dtype=object),
    np.empty(n, dtype=object),
    np.empty(n, dtype=object),
  )

  for i in range(n):
    if (selected[i] == 0) or (vb[i] == i):
      tpha[i] = np.nan
      slha[i] = np.nan
      vbha[i] = np.nan
      continue

    ret = ((s[i:] / s[i]) - 1) * side[i]

    # find the first index where take profit barrier is hit
    tp_idx = np.nan
    for j in range(len(ret)):
      if ret[j] >= tpb[i]:
        tp_idx = i + j
        break

    # find the first index where stop loss barrier is hit
    sl_idx = np.nan
    for j in range(len(ret)):
      if ret[j] <= slb[i]:
        sl_idx = i + j
        break
    del ret
    tpha[i] = tp_idx
    slha[i] = sl_idx

    # first index where vertical barrier is hit
    if vb[i] < n:
      vbha[i] = vb[i]
    else:
      vbha[i] = np.nan

  first_touch_at = np.nanmin([tpha, slha, vbha], axis=0)
  # first_touch_type:
  #   1: take profit barrier hit
  #   -1: stop loss barrier hit
  #   0: vertical barrier hit
  #   np.nan: no barrier hit
  for i in range(n):
    if np.isnan(first_touch_at[i]):
      first_touch_type[i] = np.nan
      first_touch_return[i] = np.nan
    elif tpha[i] == first_touch_at[i]:
      first_touch_type[i] = 1
      first_touch_return[i] = s[tpha[i]] / s[i] - 1
    elif slha[i] == first_touch_at[i]:
      first_touch_type[i] = -1
      first_touch_return[i] = s[slha[i]] / s[i] - 1
    elif vbha[i] == first_touch_at[i]:
      first_touch_type[i] = 0
      first_touch_return[i] = s[vbha[i]] / s[i] - 1

  return tpha, slha, vbha, first_touch_at, first_touch_type, first_touch_return


def triple_barrier(
  s: pd.Series,
  selected: pd.Series,  # whether to run the search on this index
  tpb: pd.Series,  # take profit barriers
  slb: pd.Series,  # stop loss barriers
  vb: pd.Series,  # absolute integer index of vertical barriers
  side: pd.Series,  # 1 for long bet, -1 for short bet
) -> pd.DataFrame:
  _validate_inputs(s, selected, tpb, slb, vb, side)

  tpha, slha, vbha, first_touch_at, first_touch_type, first_touch_return = (
    _triple_barrier(
      s=s.values,
      selected=selected.values,
      tpb=tpb.values,
      slb=slb.values,
      vb=vb.values,
      side=side.values,
    )
  )

  result = pd.DataFrame(
    {
      "tpha": tpha,
      "slha": slha,
      "vbha": vbha,
      "first_touch_at": first_touch_at,
      "first_touch_type": first_touch_type,
      "first_touch_return": first_touch_return,
    },
    index=s.index,
  ).astype(
    {
      "tpha": "Int32",
      "slha": "Int32",
      "vbha": "Int32",
      "first_touch_at": "Int32",
      "first_touch_type": "Int32",
      "first_touch_return": "float32",
    }
  )
  return result
