import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.performance as perf
import algotrading_v40.utils.testing as ut


def _check_against_reference(
  backtest_result: perf.BacktestResult, expected_stats: dict, input_df: pd.DataFrame
):
  np.testing.assert_allclose(
    backtest_result.equity_final, expected_stats["Equity Final [$]"]
  )
  np.testing.assert_allclose(backtest_result.return_pct, expected_stats["Return [%]"])

  equity_curve = backtest_result.equity_curve

  expected_equity_curve = (
    expected_stats["_equity_curve"]["Equity"].shift(-1).rename("equity").iloc[1:-1]
  )
  expected_equity_curve.index.name = "bar_close_timestamp"

  np.testing.assert_allclose(backtest_result.equity_final, equity_curve.iloc[-1])

  pd.testing.assert_index_equal(
    equity_curve.index,
    expected_equity_curve.index,
  )

  pd.testing.assert_index_equal(
    backtest_result.equity_curve.index,
    input_df.index,
  )

  pd.testing.assert_series_equal(
    equity_curve,
    expected_equity_curve,
  )


def test_random_case_matches_reference_with_order_rejection_true():
  # a large random test case with error_on_order_rejection=True that passes
  # ensures future fixes don't break existing functionality
  np.random.seed(4)
  df = ut.get_test_df(
    start_date=dt.datetime(2021, 1, 14), end_date=dt.datetime(2021, 3, 14)
  )
  df["valuation_price"] = df["close"].shift(-1)
  df["final_ba_position"] = np.random.uniform(-10, 10, len(df)).round().astype(int)
  df["volume"] = 1
  df.loc[df.index[-1], "final_ba_position"] = 0
  df.index.name = "bar_close_timestamp"

  initial_cash = 1e5
  commission = 0.0005

  with ut.expect_no_mutation(df):
    backtest_result = perf.backtest(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=True,
    )
  with ut.expect_no_mutation(df):
    stats = perf.backtest_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  _check_against_reference(
    backtest_result=backtest_result, expected_stats=stats, input_df=df
  )


def test_affordability_uses_margin_check_price():
  # tests that affordability criterion uses valuation_price and not close price
  df = pd.DataFrame(
    {
      "open": [
        737.53,
        737.14,
        738.45,
      ],
      "high": [
        738.77,
        738.55,
        741.07,
      ],
      "low": [
        737.17,
        737.14,
        738.45,
      ],
      "close": [
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
  df["valuation_price"] = df["close"].shift(-1)

  initial_cash = 1e5
  commission = 0.0005

  with ut.expect_no_mutation(df):
    backtest_result = perf.backtest(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=False,
    )
  with ut.expect_no_mutation(df):
    stats = perf.backtest_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  _check_against_reference(
    backtest_result=backtest_result, expected_stats=stats, input_df=df
  )


def test_unaffordable_trades_do_not_open():
  # tests the existence of the affordability criterion
  df = pd.DataFrame(
    {
      "open": [551.05, 550.90, 550.70, 551.00, 551.30],
      "high": [551.05, 550.95, 551.05, 551.45, 551.35],
      "low": [550.75, 550.60, 550.65, 551.00, 551.20],
      "close": [550.75, 550.70, 551.05, 551.25, 551.30],
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
  df["valuation_price"] = df["close"].shift(-1)

  initial_cash = 1e4
  commission = 0.0005

  with ut.expect_no_mutation(df):
    backtest_result = perf.backtest(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=False,
    )
  with ut.expect_no_mutation(df):
    stats = perf.backtest_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  _check_against_reference(
    backtest_result=backtest_result, expected_stats=stats, input_df=df
  )


def test_trading_stops_when_account_out_of_cash():
  # tests that trading is stopped if the account is out of money
  df = pd.DataFrame(
    {
      "open": [
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
      "high": [
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
      "low": [
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
      "close": [
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
  df["valuation_price"] = df["close"].shift(-1)
  initial_cash = 1e5
  commission = 0.0005

  with ut.expect_no_mutation(df):
    backtest_result = perf.backtest(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=False,
    )
  with ut.expect_no_mutation(df):
    stats = perf.backtest_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  _check_against_reference(
    backtest_result=backtest_result, expected_stats=stats, input_df=df
  )


def test_random_case_order_rejection_false_matches_reference():
  # a large random test case where an order will be rejected (so the code throws an error when error_on_order_rejection=True)
  # with error_on_order_rejection=False, the code not error and match backtesting library's result
  # np.random.seed(4)
  df = ut.get_test_df(
    start_date=dt.datetime(2021, 1, 14), end_date=dt.datetime(2021, 12, 14)
  )
  df["valuation_price"] = df["close"].shift(-1)
  df["final_ba_position"] = np.random.uniform(-100, 100, len(df)).round().astype(int)
  df["volume"] = 1
  df.loc[df.index[-1], "final_ba_position"] = 0
  df.index.name = "bar_close_timestamp"

  initial_cash = 1e3
  commission = 0.0005

  with ut.expect_no_mutation(df):
    backtest_result = perf.backtest(
      df,
      initial_cash=initial_cash,
      commission_rate=commission,
      error_on_order_rejection=False,
    )
  with ut.expect_no_mutation(df):
    stats = perf.backtest_reference(
      df, initial_cash=initial_cash, commission_rate=commission
    )
  _check_against_reference(
    backtest_result=backtest_result, expected_stats=stats, input_df=df
  )
