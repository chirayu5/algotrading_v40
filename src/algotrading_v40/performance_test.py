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
  date_rng = sdr.DateRange(dt.date(2021, 1, 1), dt.date(2021, 10, 2))
  data = dac.get_cleaned_data(instruments, date_rng)
  df_icici = data.get_full_df_for_instrument_desc(
    sid.EquityDesc(symbol="ICICIBANK", market=sid.Market.INDIAN_MARKET)
  )
  df_icici["final_ba_position"] = (
    np.random.uniform(-50, 50, len(df_icici)).round().astype(int)
  )
  df_icici = df_icici.rename(
    columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
  )
  start = np.random.randint(0, len(df_icici) - 100)
  df_icici = df_icici.iloc[start : start + 20].copy()
  return df_icici


def test_compute_backtesting_return_matches_backtesting_library():
  np.random.seed(257055)

  df = _prepare_df()
  df["volume"] = 1
  df.index.name = "bar_close_timestamp"
  eq_np, pct_np = perf.compute_backtesting_return(
    df, initial_cash=10000, commission_rate=0.0005
  )
  bt = Backtest(
    df,
    PositionStrategy,
    commission=0.0005,
    trade_on_close=True,
    hedging=False,
    exclusive_orders=False,
    cash=10000,
  )
  stats = bt.run()
  np.testing.assert_allclose(eq_np, stats["Equity Final [$]"])
  np.testing.assert_allclose(pct_np, stats["Return [%]"])
