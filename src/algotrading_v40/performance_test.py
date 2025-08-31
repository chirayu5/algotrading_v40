# ... existing code ...

import datetime as dt

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import Strategy

import algotrading_v40.data_accessors.cleaned as dac
import algotrading_v40.performance as perf
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid


class PositionStrategy(Strategy):
  """
  Identical to the strategy used in the Jupyter notebook:
  each bar we rebalance so our position equals the most recent value of
  `final_ba_position`.
  """

  def init(self):
    super().init()

  def next(self):
    desired = int(self.data.final_ba_position[-1])
    current = self.position.size
    delta = desired - current
    if delta > 0:
      self.buy(size=delta)
    elif delta < 0:
      self.sell(size=-delta)


def _prepare_df() -> pd.DataFrame:
  instruments = [
    sid.EquityDesc(symbol="ICICIBANK", market=sid.Market.INDIAN_MARKET),
  ]
  date_rng = sdr.DateRange(dt.date(2021, 1, 1), dt.date(2021, 10, 10))
  data = dac.get_cleaned_data(instruments, date_rng)
  df_icici = data.get_full_df_for_instrument_desc(
    sid.EquityDesc(symbol="ICICIBANK", market=sid.Market.INDIAN_MARKET)
  )
  np.random.seed(75016)
  df_icici["final_ba_position"] = (
    np.random.uniform(-50, 50, len(df_icici)).round().astype(int)
  )
  df_icici = df_icici.rename(
    columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
  )
  df_icici = df_icici.iloc[1590:1630].copy()
  df_icici.loc[df_icici.index[-1], "final_ba_position"] = 0
  return df_icici


def test_compute_backtesting_return_matches_backtesting_library():
  df = _prepare_df()
  _, pct_np = perf.compute_backtesting_return(df)
  bt = Backtest(
    df,
    PositionStrategy,
    commission=0.0005,
    trade_on_close=True,
    hedging=False,
    exclusive_orders=False,
  )
  stats = bt.run()
  pct_lib = stats["Return [%]"]
  np.testing.assert_allclose(pct_np, pct_lib)
