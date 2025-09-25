import algotrading_v40_cpp.labellers as av40c_l
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


def triple_barrier(
  s: pd.Series,
  selected: pd.Series,  # whether to run the search on this index
  tpb: pd.Series,  # take profit barriers
  slb: pd.Series,  # stop loss barriers
  vb: pd.Series,  # absolute integer index of vertical barriers
  side: pd.Series,  # 1 for long bet, -1 for short bet
) -> pd.DataFrame:
  _validate_inputs(s, selected, tpb, slb, vb, side)

  result = pd.DataFrame(
    av40c_l.triple_barrier_cpp(
      s=s.values,
      selected=selected.values,
      tpb=tpb.values,
      slb=slb.values,
      vb=vb.values,
      side=side.values,
    ),
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
