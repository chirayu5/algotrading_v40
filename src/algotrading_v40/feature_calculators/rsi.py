import algotrading_v40_cpp as av40c
import pandas as pd


def rsi(
  prices: pd.Series,
  lookback: int,
) -> pd.Series:
  return pd.Series(
    data=av40c.rsi_cpp(prices.values, lookback),
    index=prices.index,
  )
