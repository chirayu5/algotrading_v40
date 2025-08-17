import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def price_intensity_(
  open: pd.Series,
  high: pd.Series,
  low: pd.Series,
  close: pd.Series,
  n_to_smooth: int,
) -> pd.Series:
  for name, s in [("open", open), ("high", high), ("low", low), ("close", close)]:
    if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
      raise ValueError(f"{name} must not have bad values")
  return pd.Series(
    data=av40c_fc.price_intensity_cpp(
      open=open.values,
      high=high.values,
      low=low.values,
      close=close.values,
      n_to_smooth=n_to_smooth,
    ),
    index=open.index,
  )


def price_intensity(
  df: pd.DataFrame,
  n_to_smooth: int,
) -> pd.DataFrame:
  open_d = df["open"].shift(1)
  high_d = df["high"].shift(1)
  low_d = df["low"].shift(1)
  close_d = df["close"].shift(1)
  series = price_intensity_(
    open=open_d.iloc[1:],
    high=high_d.iloc[1:],
    low=low_d.iloc[1:],
    close=close_d.iloc[1:],
    n_to_smooth=n_to_smooth,
  ).reindex(df.index)
  col_name = f"price_intensity_{n_to_smooth}"
  return pd.DataFrame(series, columns=[col_name])
