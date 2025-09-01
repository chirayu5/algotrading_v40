import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.performance as perf
import algotrading_v40.utils.testing as ut


def test_compute_backtesting_return_matches_backtesting_library_0():
  # a large random test case with error_on_order_rejection=True that passes
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
      "Open": [
        737.53,
        737.14,
        738.45,
      ],
      "High": [
        738.77,
        738.55,
        741.07,
      ],
      "Low": [
        737.17,
        737.14,
        738.45,
      ],
      "Close": [
        738.21,
        738.17,
        740.93,
      ],
      "volume": [1, 1, 1],
      "final_ba_position": [-61, -135, 0],
    },
    index=pd.to_datetime(
      [
        "2021-01-14 03:48:59.999000+00:00",
        "2021-01-14 03:49:59.999000+00:00",
        "2021-01-14 03:50:59.999000+00:00",
      ]
    ),
  )
  df.index.name = "bar_close_timestamp"
  print()
  print(df.to_string())

  initial_cash = 1e5
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


# for ii in range(2000):
# print(f"ii: {ii}")
# df = ut.get_test_df(
#   start_date=dt.datetime(2021, 1, 14), end_date=dt.datetime(2021, 3, 14)
# ).rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low"})
# df["final_ba_position"] = np.random.uniform(-500, 500, len(df)).round().astype(int)
# df["volume"] = 1
# df.index.name = "bar_close_timestamp"
# df = df.iloc[:10]
# df.loc[df.index[-1], "final_ba_position"] = 0
# assert len(df) == 10
# df = pd.read_parquet(
#   "test_compute_backtesting_return_matches_backtesting_library_1_2sep.parquet"
# )


# for _ in range(1000):
#   try:
# df = ut.get_test_df(
#   start_date=dt.datetime(2021, 1, 14), end_date=dt.datetime(2021, 12, 14)
# ).rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low"})
# df = df.sample(n=10).sort_values(by="bar_close_timestamp")
# assert df.shape[0] == 10
# df["final_ba_position"] = np.random.uniform(-50, 50, len(df)).round().astype(int)
# df["volume"] = 1
# df.loc[df.index[-1], "final_ba_position"] = 0
# df.index.name = "bar_close_timestamp"

# df.to_parquet(
#   "test_compute_backtesting_return_matches_backtesting_library_2.parquet"
# )
# df = pd.read_parquet(
#   "test_compute_backtesting_return_matches_backtesting_library_2.parquet"
# )
# print()
# print(df.to_string())


def test_compute_backtesting_return_matches_backtesting_library_2():
  # tests the affordability criterion
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


def test_compute_backtesting_return_matches_backtesting_library_3():
  # tests that trading is stopped if the account is out of money
  df = pd.DataFrame(
    {
      "Open": [
        30.26,
        34.95,
        44.22,
        184.11,
        210.83,
        1197.93,
        1294.06,
        1828.09,
        4665.46,
        75880.18,
      ],
      "High": [
        30.27,
        34.95,
        44.22,
        184.74,
        211.76,
        1200.31,
        1296.94,
        1830.63,
        4665.46,
        75938.31,
      ],
      "Low": [
        30.17,
        34.78,
        44.08,
        184.02,
        210.83,
        1196.13,
        1293.69,
        1825.07,
        4635.16,
        75739.69,
      ],
      "Close": [
        30.17,
        34.80,
        44.14,
        184.74,
        211.76,
        1198.44,
        1296.94,
        1826.57,
        4636.84,
        75938.31,
      ],
      "volume": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "final_ba_position": [-8, 42, -35, 4, -19, -21, -32, -19, -28, 0],
    },
    index=pd.to_datetime(
      [
        "2021-01-21 04:32:59.999000+00:00",
        "2021-01-25 06:09:59.999000+00:00",
        "2021-02-09 07:10:59.999000+00:00",
        "2021-03-24 09:15:59.999000+00:00",
        "2021-03-30 08:29:59.999000+00:00",
        "2021-06-02 05:07:59.999000+00:00",
        "2021-06-07 05:02:59.999000+00:00",
        "2021-06-14 07:32:59.999000+00:00",
        "2021-07-02 05:34:59.999000+00:00",
        "2021-10-25 05:50:59.999000+00:00",
      ]
    ),
  )
  df.index.name = "bar_close_timestamp"
  initial_cash = 1e5
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
  np.testing.assert_allclose(eq_np, stats["Equity Final [$]"])
  np.testing.assert_allclose(pct_np, stats["Return [%]"])
