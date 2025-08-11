import algotrading_v40_cpp.feature_calculators as av40c_fc
import pandas as pd

import algotrading_v40.utils.df as udf


def stochastic_rsi_(
  prices: pd.Series,
  rsi_lookback: int,
  stoch_lookback: int,
  n_to_smooth: int,
) -> pd.Series:
  if udf.analyse_numeric_series_quality(prices).n_bad_values > 0:
    raise ValueError("prices must not have bad values")
  return pd.Series(
    data=av40c_fc.stochastic_rsi_cpp(
      prices.values, rsi_lookback, stoch_lookback, n_to_smooth
    ),
    index=prices.index,
  )


def stochastic_rsi(
  df: pd.DataFrame,
  rsi_lookback: int,
  stoch_lookback: int,
  n_to_smooth: int,
) -> pd.DataFrame:
  close_delayed = df["close"].shift(1)
  srsi_series = stochastic_rsi_(
    prices=close_delayed.iloc[1:],
    rsi_lookback=rsi_lookback,
    stoch_lookback=stoch_lookback,
    n_to_smooth=n_to_smooth,
  ).reindex(df.index)
  return pd.DataFrame(
    data=srsi_series,
    index=df.index,
    columns=[f"stochastic_rsi_{rsi_lookback}_{stoch_lookback}_{n_to_smooth}"],
  )
