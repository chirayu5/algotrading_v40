import dataclasses
from typing import Callable

import numpy as np
import pandas as pd


class DfStreamer:
  def __init__(self, df: pd.DataFrame):
    expected_columns = {"open", "high", "low", "close", "volume"}
    if set(df.columns) != expected_columns:
      raise ValueError(
        f"DataFrame must have columns {expected_columns}, got {set(df.columns)}"
      )
    self.df = df
    self.pending_idx = 0

  def next(self) -> pd.DataFrame:
    df = self.df.iloc[: (self.pending_idx + 1)].copy()
    for col in df.columns:
      if col != "open":
        df.loc[df.index[-1], col] = np.nan
    self.pending_idx += 1
    return df


@dataclasses.dataclass(frozen=True)
class CompareBatchAndStreamResult:
  df_batch: pd.DataFrame
  df_batch_mismatch: pd.DataFrame | None
  df_stream_mismatch: pd.DataFrame | None
  dfs_match: bool

  def __post_init__(self):
    if (self.df_batch_mismatch is None) != (self.df_stream_mismatch is None):
      raise ValueError(
        "df_batch_mismatch and df_stream_mismatch must both be None or both be not None"
      )
    if (not self.dfs_match) and (self.df_batch_mismatch is None):
      raise ValueError("df_batch_mismatch must be not None if dfs_match is False")


def compare_batch_and_stream(
  df: pd.DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame]
) -> CompareBatchAndStreamResult:
  streamer = DfStreamer(df)
  df_batch = func(df)
  for i in range(len(df)):
    df_ = streamer.next()
    df_stream = func(df_)
    if not df_batch.iloc[: i + 1].equals(df_stream):
      return CompareBatchAndStreamResult(
        df_batch=df_batch,
        df_batch_mismatch=df_batch.iloc[: i + 1],
        df_stream_mismatch=df_stream,
        dfs_match=False,
      )
  return CompareBatchAndStreamResult(
    df_batch=df_batch,
    df_batch_mismatch=None,
    df_stream_mismatch=None,
    dfs_match=True,
  )
