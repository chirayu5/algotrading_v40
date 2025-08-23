import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.utils.features as uf
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


class TestGetIndianMarketSessionInfo:
  def test_two_sessions_basic_properties(self) -> None:
    with ut.expect_no_mutation(TEST_INDEX):
      df = uf.get_indian_market_session_info(TEST_INDEX)

    expected_df = pd.DataFrame(
      {
        "is_first_bar_of_session": [
          True,
          False,
          False,
          False,
          False,
          True,
          False,
          False,
          False,
          False,
          True,
          False,
        ],
        "is_last_bar_of_session": [
          False,
          False,
          False,
          False,
          # is_last_bar_of_session is strictly time based to ensure streamability.
          # so "2023-01-02 09:52:59.999000" will be false since it is not
          # "2023-01-02 09:59:59.999000"
          False,
          False,
          False,
          False,
          False,
          True,
          False,
          False,
        ],
        "session_date": [
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 2),
          dt.date(2023, 1, 4),
          dt.date(2023, 1, 4),
          dt.date(2023, 1, 4),
          dt.date(2023, 1, 4),
          dt.date(2023, 1, 4),
          dt.date(2023, 1, 10),
          dt.date(2023, 1, 10),
        ],
        "session_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3],
        "bar_number_in_session": [0, 1, 2, 3, 367, 5, 6, 7, 8, 374, 0, 1],
        "weekday": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1],
      },
      index=TEST_INDEX,
    ).astype({"weekday": "int32"})
    pd.testing.assert_frame_equal(df, expected_df)

  def test_streamability(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    result = us.compare_batch_and_stream(
      df, lambda df_: uf.get_indian_market_session_info(df_.index)
    )
    assert result.dfs_match

  def test_raises_for_non_utc_timezone(self) -> None:
    start = pd.Timestamp("2023-01-02 03:45:59.999000", tz="Asia/Kolkata")
    idx = pd.date_range(start=start, periods=10, freq="min")
    with pytest.raises(ValueError, match="UTC"):
      uf.get_indian_market_session_info(idx)

  def test_raises_for_time_outside_session_window(self) -> None:
    # one minute before market open
    invalid_start = pd.Timestamp("2023-01-02 03:44:59.999000", tz="UTC")
    idx = pd.date_range(start=invalid_start, periods=5, freq="min", tz="UTC")
    with pytest.raises(ValueError, match="All times must be between"):
      uf.get_indian_market_session_info(idx)


class TestGetTimeBasedBarGroupForIndianMarket:
  def test_1_minute_groups(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = uf.get_time_based_bar_group_for_indian_market(df, 1)
    df["bar_group"] = result
    np.testing.assert_array_equal(df.index.values, df["bar_group"].values)

  def test_7_minute_groups(self) -> None:
    test_index = pd.date_range(
      start="2023-01-02 03:45:59.999000",
      end="2023-01-02 04:00:59.999000",
      freq="1min",
      tz="UTC",
    )
    df = pd.DataFrame(index=test_index)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = uf.get_time_based_bar_group_for_indian_market(df, 7)
    df["bar_group"] = result

    expected_times = pd.Series(
      [
        # first group
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        # second group
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:52:59.999000", tz="UTC"),
        # third group (small but not the last group...so not merged)
        pd.Timestamp("2023-01-02 03:59:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:59:59.999000", tz="UTC"),
      ],
      index=df.index,
    )
    np.testing.assert_array_equal(df["bar_group"].values, expected_times.values)

  def test_last_small_group_is_merged(self) -> None:
    test_index = pd.date_range(
      start="2023-01-02 09:43:59.999000",
      end="2023-01-02 09:59:59.999000",
      freq="1min",
      tz="UTC",
    )
    df = pd.DataFrame(index=test_index)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = uf.get_time_based_bar_group_for_indian_market(df, 11)
    df["bar_group"] = result

    expected_times = pd.Series(
      [
        # 32nd group
        pd.Timestamp("2023-01-02 09:37:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:37:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:37:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:37:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:37:59.999000", tz="UTC"),
        # 33rd group
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
        # single remaining bar included in the 33rd group only (small last group...merged)
        pd.Timestamp("2023-01-02 09:48:59.999000", tz="UTC"),
      ],
      index=df.index,
    )
    np.testing.assert_array_equal(df["bar_group"].values, expected_times.values)

  def test_last_adequate_group_is_not_merged(self) -> None:
    test_index = pd.date_range(
      start="2023-01-02 09:43:59.999000",
      end="2023-01-02 09:59:59.999000",
      freq="1min",
      tz="UTC",
    )
    df = pd.DataFrame(index=test_index)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = uf.get_time_based_bar_group_for_indian_market(df, 9)
    df["bar_group"] = result

    expected_times = pd.Series(
      [
        # 40th group (appears only partially in the input)
        pd.Timestamp("2023-01-02 09:36:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:36:59.999000", tz="UTC"),
        # 41st group
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:45:59.999000", tz="UTC"),
        # incomplete adequately sized last 42nd group...so not merged
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 09:54:59.999000", tz="UTC"),
      ],
      index=df.index,
    )
    np.testing.assert_array_equal(df["bar_group"].values, expected_times.values)

  def test_streamability(self) -> None:
    test_index_1 = TEST_INDEX
    test_index_2 = pd.date_range(
      start="2023-01-02 03:45:59.999000",
      end="2023-01-02 04:00:59.999000",
      freq="1min",
      tz="UTC",
    )
    test_index_3 = pd.date_range(
      start="2023-01-02 09:43:59.999000",
      end="2023-01-02 09:59:59.999000",
      freq="1min",
      tz="UTC",
    )
    for test_index, group_size in [
      (test_index_1, 1),
      (test_index_2, 7),
      (test_index_3, 11),
      (test_index_3, 9),
    ]:
      df = pd.DataFrame(index=test_index)
      df["open"] = np.random.uniform(100, 200, len(df))
      df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
      df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
      df["close"] = np.random.uniform(df["low"], df["high"])
      df["volume"] = np.random.uniform(1000, 10000, len(df))
      result = us.compare_batch_and_stream(
        df, lambda df_: uf.get_time_based_bar_group_for_indian_market(df_, group_size)
      )
      assert result.dfs_match

  def test_375_minute_groups(self) -> None:
    df = pd.DataFrame(index=TEST_INDEX)
    df["open"] = np.random.uniform(100, 200, len(df))
    df["high"] = df["open"] + np.random.uniform(0, 10, len(df))
    df["low"] = df["open"] - np.random.uniform(0, 10, len(df))
    df["close"] = np.random.uniform(df["low"], df["high"])
    df["volume"] = np.random.uniform(1000, 10000, len(df))
    with ut.expect_no_mutation(df):
      result = uf.get_time_based_bar_group_for_indian_market(df, 375)
    df["bar_group"] = result

    expected_times = pd.Series(
      [
        # Session 1 (2023-01-02)
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),  # first bar
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-02 03:45:59.999000", tz="UTC"),
        # Session 2 (2023-01-04)
        # first bar due to overnight gap and not time-based
        # can happen if first bar is missing
        pd.Timestamp("2023-01-04 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-04 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-04 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-04 03:45:59.999000", tz="UTC"),
        pd.Timestamp("2023-01-04 03:45:59.999000", tz="UTC"),  # last bar
        # Session 3 (2023-01-10)
        # incomplete session
        pd.Timestamp("2023-01-10 03:45:59.999000", tz="UTC"),  # first bar
        pd.Timestamp("2023-01-10 03:45:59.999000", tz="UTC"),
      ],
      index=df.index,
    )
    np.testing.assert_array_equal(df["bar_group"].values, expected_times.values)
