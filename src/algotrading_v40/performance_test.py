import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.performance as perf
import algotrading_v40.utils.testing as ut


def test_compute_backtesting_return_matches_backtesting_library_0():
  # a large random test case that passes
  # ensures future fixes don't break existing functionality
  np.random.seed(4)
  df = ut.get_test_df(
    start_date=dt.datetime(2021, 1, 14), end_date=dt.datetime(2021, 3, 14)
  ).rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low"})
  df["final_ba_position"] = np.random.uniform(-10, 10, len(df)).round().astype(int)
  df["volume"] = 1
  df.loc[df.index[-1], "final_ba_position"] = 0
  df.index.name = "bar_close_timestamp"

  initial_cash = 1e5
  commission = 0.0005

  with ut.expect_no_mutation(df):
    eq_np, pct_np = perf.compute_backtesting_return(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=True,
    )
  with ut.expect_no_mutation(df):
    stats = perf.compute_backtesting_return_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  print("stats: ", stats)
  print("eq_np: ", eq_np)
  print("pct_np: ", pct_np)
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
      "final_ba_position": [37, 5, -18, -48, 0],
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

  initial_cash = 1e4
  commission = 0.0005

  with ut.expect_no_mutation(df):
    eq_np, pct_np = perf.compute_backtesting_return(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=False,
    )
  with ut.expect_no_mutation(df):
    stats = perf.compute_backtesting_return_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  print("stats: ", stats)
  print("eq_np: ", eq_np)
  print("pct_np: ", pct_np)
  np.testing.assert_allclose(eq_np, stats["Equity Final [$]"])
  np.testing.assert_allclose(pct_np, stats["Return [%]"])
