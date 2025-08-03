import algotrading_v40_cpp as av40c
import pandas as pd

import algotrading_v40.utils.df as udf


def rsi_(
  prices: pd.Series,
  lookback: int,
) -> pd.Series:
  if udf.analyse_numeric_series_quality(prices).n_bad_values > 0:
    raise ValueError("prices must not have bad values")
  return pd.Series(
    data=av40c.rsi_cpp(prices.values, lookback),
    index=prices.index,
  )


def rsi(
  df: pd.DataFrame,
  lookback: int,
) -> pd.DataFrame:
  close_delayed = df["close"].shift(1)
  rsi_series = rsi_(close_delayed.iloc[1:], lookback)
  rsi_series = rsi_series.reindex(df.index)
  return pd.DataFrame(
    data=rsi_series,
    index=df.index,
    columns=[f"rsi_{lookback}"],
  )
