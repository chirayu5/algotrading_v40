import numpy as np
import pandas as pd

import algotrading_v40.feature_calculators.open_gap as fc_open_gap
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
    pd.Timestamp("2023-01-02 09:52:59.999000", tz="UTC"),  # last bar
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


class TestOpenGap:
  def test_lag_0(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = fc_open_gap.open_gap_indian_market(df, lag=0)
    open_gap_A = (df["open"].loc[TEST_INDEX[5]] / df["close"].loc[TEST_INDEX[4]]) - 1
    open_gap_B = (df["open"].loc[TEST_INDEX[10]] / df["close"].loc[TEST_INDEX[9]]) - 1
    pd.testing.assert_frame_equal(
      result,
      pd.DataFrame(
        data=[
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          np.nan,
          open_gap_A,
          open_gap_A,
          open_gap_A,
          open_gap_A,
          open_gap_A,
          open_gap_B,
          open_gap_B,
        ],
        index=TEST_INDEX,
        columns=["open_gap_0"],
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
      result = fc_open_gap.open_gap_indian_market(df, lag=1)
    open_gap_A = (df["open"].loc[TEST_INDEX[5]] / df["close"].loc[TEST_INDEX[4]]) - 1
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
          open_gap_A,
          open_gap_A,
        ],
        index=TEST_INDEX,
        columns=["open_gap_1"],
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
      df, lambda df_: fc_open_gap.open_gap_indian_market(df_, lag=0)
    )
    result_quality = udf.analyse_numeric_series_quality(result.df_batch["open_gap_0"])
    # First 7 values should be good and match with streaming
    assert result_quality.n_good_values >= 7
    # there should be no bad values at the end
    assert result_quality.n_bad_values_at_end == 0
    # all bad values should be at the start
    assert result_quality.n_bad_values_at_start == result_quality.n_bad_values
    assert result.dfs_match
