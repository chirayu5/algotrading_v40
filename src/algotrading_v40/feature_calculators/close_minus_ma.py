import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def close_minus_ma_(
  high: pd.Series,
  low: pd.Series,
  close: pd.Series,
  lookback: int,
  atr_length: int,
) -> pd.Series:
  for name, s in [("high", high), ("low", low), ("close", close)]:
    if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
      raise ValueError(f"{name} must not have bad values")

  return pd.Series(
    data=av40c_fc.close_minus_ma_cpp(
      high=high.values,
      low=low.values,
      close=close.values,
      lookback=lookback,
      atr_length=atr_length,
    ),
    index=close.index,
  )


def close_minus_ma(
  df: pd.DataFrame,
  lookback: int,
  atr_length: int,
) -> pd.DataFrame:
  high_d = df["high"].shift(1)
  low_d = df["low"].shift(1)
  close_d = df["close"].shift(1)

  series = close_minus_ma_(
    high=high_d.iloc[1:],
    low=low_d.iloc[1:],
    close=close_d.iloc[1:],
    lookback=lookback,
    atr_length=atr_length,
  ).reindex(df.index)

  col_name = f"close_minus_ma_{lookback}_{atr_length}"
  return pd.DataFrame(series, columns=[col_name])
