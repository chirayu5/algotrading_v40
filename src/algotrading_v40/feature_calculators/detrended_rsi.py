import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def detrended_rsi_(
  prices: pd.Series,
  short_length: int,
  long_length: int,
  length: int,
) -> pd.Series:
  if udf.analyse_numeric_series_quality(prices).n_bad_values > 0:
    raise ValueError("prices must not have bad values")
  return pd.Series(
    data=av40c_fc.detrended_rsi_cpp(
      prices.values,
      short_length,
      long_length,
      length,
    ),
    index=prices.index,
  )


def detrended_rsi(
  df: pd.DataFrame,
  short_length: int,
  long_length: int,
  length: int,
) -> pd.DataFrame:
  close_delayed = df["close"].shift(1)
  detrended_rsi_series = detrended_rsi_(
    prices=close_delayed.iloc[1:],
    short_length=short_length,
    long_length=long_length,
    length=length,
  )
  detrended_rsi_series = detrended_rsi_series.reindex(df.index)
  return pd.DataFrame(
    data=detrended_rsi_series,
    index=df.index,
    columns=[f"detrended_rsi_{short_length}_{long_length}_{length}"],
  )
