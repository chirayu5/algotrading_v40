import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def stochastic_(
  close: pd.Series,
  high: pd.Series,
  low: pd.Series,
  lookback: int,
  n_to_smooth: int,
) -> pd.Series:
  if udf.analyse_numeric_series_quality(close).n_bad_values > 0:
    raise ValueError("close must not have bad values")
  if udf.analyse_numeric_series_quality(high).n_bad_values > 0:
    raise ValueError("high must not have bad values")
  if udf.analyse_numeric_series_quality(low).n_bad_values > 0:
    raise ValueError("low must not have bad values")
  return pd.Series(
    data=av40c_fc.stochastic_cpp(
      close=close.values,
      high=high.values,
      low=low.values,
      lookback=lookback,
      n_to_smooth=n_to_smooth,
    ),
    index=close.index,
  )


def stochastic(
  df: pd.DataFrame,
  lookback: int,
  n_to_smooth: int,
) -> pd.DataFrame:
  close_delayed = df["close"].shift(1)
  high_delayed = df["high"].shift(1)
  low_delayed = df["low"].shift(1)
  stochastic_series = stochastic_(
    close=close_delayed.iloc[1:],
    high=high_delayed.iloc[1:],
    low=low_delayed.iloc[1:],
    lookback=lookback,
    n_to_smooth=n_to_smooth,
  )
  stochastic_series = stochastic_series.reindex(df.index)
  return pd.DataFrame(
    data=stochastic_series,
    index=df.index,
    columns=[f"stochastic_{lookback}_{n_to_smooth}"],
  )
