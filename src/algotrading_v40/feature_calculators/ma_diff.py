import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def ma_diff_(
  high: pd.Series,
  low: pd.Series,
  close: pd.Series,
  short_length: int,
  long_length: int,
  lag: int,
) -> pd.Series:
  for name, s in [("high", high), ("low", low), ("close", close)]:
    if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
      raise ValueError(f"{name} must not have bad values")

  return pd.Series(
    data=av40c_fc.ma_diff_cpp(
      high.values,
      low.values,
      close.values,
      short_length,
      long_length,
      lag,
    ),
    index=close.index,
  )


def ma_diff(
  df: pd.DataFrame,
  short_length: int,
  long_length: int,
  lag: int,
) -> pd.DataFrame:
  high_d = df["high"].shift(1)
  low_d = df["low"].shift(1)
  close_d = df["close"].shift(1)

  series = ma_diff_(
    high=high_d.iloc[1:],
    low=low_d.iloc[1:],
    close=close_d.iloc[1:],
    short_length=short_length,
    long_length=long_length,
    lag=lag,
  ).reindex(df.index)

  col_name = f"ma_diff_{short_length}_{long_length}_{lag}"
  return pd.DataFrame(series, columns=[col_name])
