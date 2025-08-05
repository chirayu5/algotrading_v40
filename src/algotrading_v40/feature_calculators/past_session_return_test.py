import numpy as np
import pandas as pd

import algotrading_v40.feature_calculators.past_session_return as fc_psr
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut

TEST_INDEX = pd.DatetimeIndex(
  [
    # Session 1 (2023-01-02)
    pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),  # first bar
    pd.Timestamp("2023-01-02 03:46:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-02 03:47:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-02 03:48:59.999000", tz="UTC"),
    pd.Timestamp(
      "2023-01-02 09:52:59.999000", tz="UTC"
    ),  # last bar (but is_last_bar_of_session will be false since the time is not 09:59:59.999000)
    # Session 2 (2023-01-04)
    # first bar due to overnight gap and not time-based
    # can happen if first bar is missing
    pd.Timestamp("2023-01-04 03:50:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-04 03:51:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-04 03:52:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-04 03:53:59.999000", tz="UTC"),
    pd.Timestamp("2023-01-04 09:59:59.999000", tz="UTC"),  # last bar
    # Session 3 (2023-01-10)
    # incomplete session
    pd.Timestamp("2023-01-10 03:45:59.999000", tz="UTC"),  # first bar
    pd.Timestamp("2023-01-10 03:46:59.999000", tz="UTC"),
  ]
)


class TestPastSessionReturn:
  def test_lag_0(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = fc_psr.past_session_return_indian_market(df, lag=0)
    past_session_return_A = (
      df["close"].loc[TEST_INDEX[4]] / df["open"].loc[TEST_INDEX[0]]
    ) - 1
    past_session_return_B = (
      df["close"].loc[TEST_INDEX[9]] / df["open"].loc[TEST_INDEX[5]]
    ) - 1
    pd.testing.assert_frame_equal(
      result,
      pd.DataFrame(
        data=[
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          past_session_return_A,
          past_session_return_A,
          past_session_return_A,
          past_session_return_A,
          past_session_return_A,
          past_session_return_B,
          past_session_return_B,
        ],
        index=TEST_INDEX,
        columns=["past_session_return_0"],
      ),
    )

  def test_lag_1(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = fc_psr.past_session_return_indian_market(df, lag=1)
    past_session_return_A = (
      df["close"].loc[TEST_INDEX[4]] / df["open"].loc[TEST_INDEX[0]]
    ) - 1
    pd.testing.assert_frame_equal(
      result,
      pd.DataFrame(
        data=[
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          past_session_return_A,
          past_session_return_A,
        ],
        index=TEST_INDEX,
        columns=["past_session_return_1"],
      ),
    )

  def test_streamability(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_psr.past_session_return_indian_market(df_, lag=0),
    )
    result_quality = udf.analyse_numeric_series_quality(
      result.df_batch["past_session_return_0"]
    )
    # First 7 values should be good and match with streaming
    assert result_quality.n_good_values >= 7
    assert result.dfs_match
