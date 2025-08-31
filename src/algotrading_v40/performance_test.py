import datetime as dt

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import Strategy

import algotrading_v40.data_accessors.cleaned as dac
import algotrading_v40.performance as perf
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.testing as ut


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
  date_rng = sdr.DateRange(dt.date(2021, 1, 1), dt.date(2021, 1, 5))
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
  return df_icici


def test_compute_backtesting_return_matches_backtesting_library_0():
  np.random.seed(4)
  df = _prepare_df()
  df["volume"] = 1
  df.index.name = "bar_close_timestamp"
  assert len(df) == 1125
  with ut.expect_no_mutation(df):
    eq_np, pct_np = perf.compute_backtesting_return(
      df, initial_cash=10000, commission_rate=0.0005
    )
  with ut.expect_no_mutation(df):
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


def test_compute_backtesting_return_matches_backtesting_library_1():
  df = pd.DataFrame(
    {
      "Open": [551.05, 550.90, 550.70, 551.00, 551.30],
      "High": [551.05, 550.95, 551.05, 551.45, 551.35],
      "Low": [550.75, 550.60, 550.65, 551.00, 551.20],
      "Close": [550.75, 550.70, 551.05, 551.25, 551.30],
      "volume": [1, 1, 1, 1, 1],
      "final_ba_position": [37, 5, -18, -48, 37],
    },
    index=pd.to_datetime(
      [
        "2021-01-14 09:14:59.999000+00:00",
        "2021-01-14 09:15:59.999000+00:00",
        "2021-01-14 09:16:59.999000+00:00",
        "2021-01-14 09:17:59.999000+00:00",
        "2021-01-14 09:18:59.999000+00:00",
      ]
    ),
  )
  df.index.name = "bar_close_timestamp"
  with ut.expect_no_mutation(df):
    eq_np, pct_np = perf.compute_backtesting_return(
      df, initial_cash=10000, commission_rate=0.0005
    )
  with ut.expect_no_mutation(df):
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
