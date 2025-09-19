import numpy as np
import pandas as pd


def get_int_index_based_bar_group(
  df: pd.DataFrame,
  group_size: int,
  offset: int,
) -> pd.Series:
  if (not df.index.is_unique) or (not df.index.is_monotonic_increasing):
    raise ValueError("df.index must be unique and monotonic increasing")
  if str(df.index.tz) != "UTC":  # type: ignore
    raise ValueError("DataFrame index must have UTC timezone")
  if offset < 0 or offset >= group_size:
    raise ValueError(
      f"offset must be between 0 and {group_size - 1} (inclusive). Got {offset}"
    )
  n = len(df)
  indices = pd.Series(group_size * (np.arange(n) // group_size), index=df.index)
  if offset > 0:
    indices = (offset + indices).shift(offset).fillna(0).astype(int)
  bg = df.index[indices.values]
  return pd.Series(bg, index=df.index)
