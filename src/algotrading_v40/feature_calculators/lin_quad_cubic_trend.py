import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def lin_quad_cubic_trend_(
  high: pd.Series,
  low: pd.Series,
  close: pd.Series,
  poly_degree: int,
  lookback: int,
  atr_length: int,
) -> pd.Series:
  for name, s in [("high", high), ("low", low), ("close", close)]:
    if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
      raise ValueError(f"{name} must not have bad values")
  return pd.Series(
    data=av40c_fc.lin_quad_cubic_trend_cpp(
      high=high.values,
      low=low.values,
      close=close.values,
      poly_degree=poly_degree,
      lookback=lookback,
      atr_length=atr_length,
    ),
    index=high.index,
  )


def lin_quad_cubic_trend(
  df: pd.DataFrame,
  poly_degree: int,
  lookback: int,
  atr_length: int,
) -> pd.DataFrame:
  if poly_degree not in [1, 2]:
    raise ValueError(
      "poly_degree must be 1 or 2 (3 is available but is not expected to be useful...so disabled right now)"
    )

  high_d = df["high"].shift(1)
  low_d = df["low"].shift(1)
  close_d = df["close"].shift(1)
  series = lin_quad_cubic_trend_(
    high=high_d.iloc[1:],
    low=low_d.iloc[1:],
    close=close_d.iloc[1:],
    poly_degree=poly_degree,
    lookback=lookback,
    atr_length=atr_length,
  ).reindex(df.index)
  col_name = f"lin_quad_cubic_trend_{poly_degree}_{lookback}_{atr_length}"
  return pd.DataFrame(series, columns=[col_name])
