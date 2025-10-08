import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.feature_calculators.detrended_rsi as fc_drsi
import algotrading_v40.feature_calculators.utils as fc_utils
import algotrading_v40.utils.df as u_df
import algotrading_v40.utils.streaming as u_s
import algotrading_v40.utils.testing as u_t


def _get_test_df() -> pd.DataFrame:
  timestamps = pd.to_datetime(
    [
      # BAR GROUPINGS WITH OFFSET 0
      "2023-01-02 03:46:59.999000+00:00",  # 0
      "2023-01-02 03:47:59.999000+00:00",  # 1
      # "2023-01-02 03:48:59.999000+00:00",
      ####################################
      # "2023-01-02 03:49:59.999000+00:00",
      "2023-01-02 03:50:59.999000+00:00",  # 2
      "2023-01-02 03:51:59.999000+00:00",  # 3
      ####################################
      # "2023-01-02 03:52:59.999000+00:00",
      "2023-01-02 03:53:59.999000+00:00",  # 4
      "2023-01-02 03:54:59.999000+00:00",  # 5
      ####################################
      # "2023-01-02 03:55:59.999000+00:00",
      # "2023-01-02 03:56:59.999000+00:00",
      "2023-01-02 03:57:59.999000+00:00",  # 6
      ####################################
      "2023-01-02 03:58:59.999000+00:00",  # 7
    ]
  )

  np.random.seed(42)
  df = pd.DataFrame(
    {
      "open": np.random.uniform(100, 110, len(timestamps)),
      "high": np.random.uniform(111, 115, len(timestamps)),
      "low": np.random.uniform(80, 95, len(timestamps)),
      "close": np.random.uniform(100, 110, len(timestamps)),
      "volume": np.random.randint(1000, 2000, len(timestamps)),
    },
    index=timestamps,
  )
  df.index.name = "bar_close_timestamp"
  return df


class TestCalculateFeaturesWithCompleteCoverage:
  # def test_against_slow_but_definitely_correct_implementation(self):
  #   def _f_calc_with_initial_nans_1(df: pd.DataFrame):
  #     dfr = pd.DataFrame(
  #       index=df.index, columns=["close_rolling_sum", "open_rolling_sum"]
  #     )
  #     dfr["close_rolling_sum"] = df["close"].shift(1).rolling(window=6).sum()
  #     dfr["open_rolling_sum"] = df["open"].rolling(window=3).sum()
  #     dfr["high_rolling_sum"] = df["high"].shift(1).rolling(window=9).sum()
  #     dfr["low_rolling_sum"] = df["low"].shift(1).rolling(window=2).sum()
  #     return dfr

  #   timestamps = pd.DatetimeIndex(
  #     [
  #       "2023-01-02 04:14:59.999000+00:00",
  #       "2023-01-02 04:55:59.999000+00:00",
  #       "2023-01-02 05:11:59.999000+00:00",
  #       "2023-01-02 07:06:59.999000+00:00",
  #       "2023-01-02 07:24:59.999000+00:00",
  #       "2023-01-02 07:41:59.999000+00:00",
  #       "2023-01-02 09:06:59.999000+00:00",
  #       "2023-01-02 09:21:59.999000+00:00",
  #       "2023-01-03 04:44:59.999000+00:00",
  #       "2023-01-03 04:57:59.999000+00:00",
  #       "2023-01-03 05:22:59.999000+00:00",
  #       "2023-01-03 05:56:59.999000+00:00",
  #       "2023-01-03 06:06:59.999000+00:00",
  #       "2023-01-03 06:51:59.999000+00:00",
  #       "2023-01-03 07:02:59.999000+00:00",
  #       "2023-01-03 07:03:59.999000+00:00",
  #       "2023-01-03 07:06:59.999000+00:00",
  #       "2023-01-03 08:20:59.999000+00:00",
  #       "2023-01-03 08:40:59.999000+00:00",
  #       "2023-01-03 09:11:59.999000+00:00",
  #       "2023-01-03 09:31:59.999000+00:00",
  #       "2023-01-04 04:04:59.999000+00:00",
  #       "2023-01-04 06:24:59.999000+00:00",
  #       "2023-01-04 07:11:59.999000+00:00",
  #     ]
  #   )

  #   df = pd.DataFrame(
  #     {
  #       "open": np.random.uniform(100, 200, len(timestamps)),
  #       "high": np.random.uniform(300, 400, len(timestamps)),
  #       "low": np.random.uniform(10, 50, len(timestamps)),
  #       "close": np.random.uniform(100, 200, len(timestamps)),
  #       "volume": np.random.randint(1000, 2000, len(timestamps)),
  #     },
  #     index=timestamps,
  #   )
  #   df.index.name = "bar_close_timestamp"

  #   group_size_minutes = 80

  #   with u_t.expect_no_mutation(df):
  #     result_df = fc_utils.calculate_features_with_complete_coverage(
  #       df=df, f_calc=_f_calc_with_initial_nans_1, group_size_minutes=group_size_minutes
  #     )

  #   r = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
  #     df=df,
  #     group_size_minutes=group_size_minutes,
  #     offset_minutes=0,
  #   )

  #   with u_t.expect_no_mutation(df):
  #     result_df_expected = pd.DataFrame(
  #       index=df.index,
  #       columns=[
  #         "close_rolling_sum",
  #         "open_rolling_sum",
  #         "high_rolling_sum",
  #         "low_rolling_sum",
  #       ],
  #     )

  #     for o in range(group_size_minutes):
  #       r = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
  #         df=df,
  #         group_size_minutes=group_size_minutes,
  #         offset_minutes=o,
  #       )
  #       df["bar_group"] = r.bar_groups
  #       df["offset"] = r.offsets
  #       with u_t.expect_no_mutation(df):
  #         df_ = u_df.calculate_grouped_values(
  #           df=df, compute_func=_f_calc_with_initial_nans_1
  #         )
  #       assert result_df_expected.loc[df["offset"] == 0, :].isna().all().all(), (
  #         "Expected all values to be NaN before assignment"
  #       )
  #       result_df_expected.loc[df["offset"] == 0, :] = df_.loc[df["offset"] == 0, :]
  #       df.drop(columns=["bar_group", "offset"], inplace=True)

  #   result_df_expected["close_rolling_sum"] = result_df_expected[
  #     "close_rolling_sum"
  #   ].astype(float)
  #   result_df_expected["open_rolling_sum"] = result_df_expected[
  #     "open_rolling_sum"
  #   ].astype(float)
  #   result_df_expected["high_rolling_sum"] = result_df_expected[
  #     "high_rolling_sum"
  #   ].astype(float)
  #   result_df_expected["low_rolling_sum"] = result_df_expected[
  #     "low_rolling_sum"
  #   ].astype(float)

  #   for col in result_df_expected.columns:
  #     last_nan = None
  #     for i in range(len(result_df_expected[col])):
  #       if np.isnan(result_df_expected[col].iloc[i]):
  #         last_nan = i
  #     if last_nan is not None:
  #       result_df_expected.loc[: result_df_expected.index[last_nan], col] = np.nan

  #   pd.testing.assert_frame_equal(result_df, result_df_expected)

  #   bsr = u_s.compare_batch_and_stream(
  #     df,
  #     lambda df_: fc_utils.calculate_features_with_complete_coverage(
  #       df=df_,
  #       f_calc=_f_calc_with_initial_nans_1,
  #       group_size_minutes=group_size_minutes,
  #     ),
  #   )
  #   # THIS TEST CASE IS ENGINEERED TO BE STREAM UNSAFE
  #   assert not bsr.dfs_match

  def test_against_handcrafted_logic(self):
    def _f_calc_with_initial_nans_2(df: pd.DataFrame):
      ob = df.loc[df.index[-1], "open"]
      df.loc[df.index[-1], :] = np.nan
      df.loc[df.index[-1], "open"] = ob
      dfr = pd.DataFrame(
        index=df.index, columns=["close_squared", "high_cubed", "open_sqrt"]
      )
      dfr["close_squared"] = df["close"].shift(1).pow(2)
      dfr["high_cubed"] = df["high"].shift(1).pow(3)
      dfr["open_sqrt"] = df["open"].pow(0.5)
      return dfr

    # # BAR GROUPINGS WITH OFFSET 0
    #     # BAR GROUPINGS WITH OFFSET 0
    #     "2023-01-02 03:46:59.999000+00:00",  # 0
    #     "2023-01-02 03:47:59.999000+00:00",  # 1
    #     # "2023-01-02 03:48:59.999000+00:00",
    #     ####################################
    #     # "2023-01-02 03:49:59.999000+00:00",
    #     "2023-01-02 03:50:59.999000+00:00",  # 2
    #     "2023-01-02 03:51:59.999000+00:00",  # 3
    #     ####################################
    #     # "2023-01-02 03:52:59.999000+00:00",
    #     "2023-01-02 03:53:59.999000+00:00",  # 4
    #     "2023-01-02 03:54:59.999000+00:00",  # 5
    #     ####################################
    #     # "2023-01-02 03:55:59.999000+00:00",
    #     # "2023-01-02 03:56:59.999000+00:00",
    #     "2023-01-02 03:57:59.999000+00:00",  # 6
    #     ####################################
    #     "2023-01-02 03:58:59.999000+00:00",  # 7

    # BAR GROUPINGS WITH OFFSET 1
    #     "2023-01-02 03:46:59.999000+00:00",  # 0
    #     ####################################
    #     "2023-01-02 03:47:59.999000+00:00",  # 1
    #     # "2023-01-02 03:48:59.999000+00:00",
    #     # "2023-01-02 03:49:59.999000+00:00",
    #     ####################################
    #     "2023-01-02 03:50:59.999000+00:00",  # 2
    #     "2023-01-02 03:51:59.999000+00:00",  # 3
    #     # "2023-01-02 03:52:59.999000+00:00",
    #     ####################################
    #     "2023-01-02 03:53:59.999000+00:00",  # 4
    #     "2023-01-02 03:54:59.999000+00:00",  # 5
    #     # "2023-01-02 03:55:59.999000+00:00",
    #     ####################################
    #     # "2023-01-02 03:56:59.999000+00:00",
    #     "2023-01-02 03:57:59.999000+00:00",  # 6
    #     "2023-01-02 03:58:59.999000+00:00",  # 7

    # BAR GROUPINGS WITH OFFSET 2
    #     "2023-01-02 03:46:59.999000+00:00",  # 0
    #     "2023-01-02 03:47:59.999000+00:00",  # 1
    #     ####################################
    #     # "2023-01-02 03:48:59.999000+00:00",
    #     # "2023-01-02 03:49:59.999000+00:00",
    #     "2023-01-02 03:50:59.999000+00:00",  # 2
    #     ####################################
    #     "2023-01-02 03:51:59.999000+00:00",  # 3
    #     # "2023-01-02 03:52:59.999000+00:00",
    #     "2023-01-02 03:53:59.999000+00:00",  # 4
    #     ####################################
    #     "2023-01-02 03:54:59.999000+00:00",  # 5
    #     # "2023-01-02 03:55:59.999000+00:00",
    #     # "2023-01-02 03:56:59.999000+00:00",
    #     ####################################
    #     "2023-01-02 03:57:59.999000+00:00",  # 6
    #     "2023-01-02 03:58:59.999000+00:00",  # 7

    df = _get_test_df()
    timestamps = df.index

    open_ = df["open"].values
    close_ = df["close"].values
    high_ = df["high"].values
    low_ = df["low"].values
    dfg0 = pd.DataFrame(
      data={
        "open": [open_[0], open_[2], open_[4], open_[6], open_[7]],
        "high": [
          max(high_[0], high_[1]),
          max(high_[2], high_[3]),
          max(high_[4], high_[5]),
          max(high_[6], high_[6]),
          max(high_[7], high_[7]),
        ],
        "low": [
          min(low_[0], low_[1]),
          min(low_[2], low_[3]),
          min(low_[4], low_[5]),
          min(low_[6], low_[6]),
          min(low_[7], low_[7]),
        ],
        "close": [close_[1], close_[3], close_[5], close_[6], close_[7]],
      },
      index=[timestamps[0], timestamps[2], timestamps[4], timestamps[6], timestamps[7]],
    )

    dfg1 = pd.DataFrame(
      data={
        "open": [open_[0], open_[1], open_[2], open_[4], open_[6]],
        "high": [
          max(high_[0], high_[0]),
          max(high_[1], high_[1]),
          max(high_[2], high_[3]),
          max(high_[4], high_[5]),
          max(high_[6], high_[7]),
        ],
        "low": [
          min(low_[0], low_[0]),
          min(low_[1], low_[1]),
          min(low_[2], low_[3]),
          min(low_[4], low_[5]),
          min(low_[6], low_[7]),
        ],
        "close": [close_[0], close_[1], close_[3], close_[5], close_[7]],
      },
      index=[timestamps[0], timestamps[1], timestamps[2], timestamps[4], timestamps[6]],
    )

    dfg2 = pd.DataFrame(
      data={
        "open": [open_[0], open_[2], open_[3], open_[5], open_[6]],
        "high": [
          max(high_[0], high_[1]),
          max(high_[2], high_[2]),
          max(high_[3], high_[4]),
          max(high_[5], high_[5]),
          max(high_[6], high_[7]),
        ],
        "low": [
          min(low_[0], low_[1]),
          min(low_[2], low_[2]),
          min(low_[3], low_[4]),
          min(low_[5], low_[5]),
          min(low_[6], low_[7]),
        ],
        "close": [close_[1], close_[2], close_[4], close_[5], close_[7]],
      },
      index=[timestamps[0], timestamps[2], timestamps[3], timestamps[5], timestamps[6]],
    )

    dfg0_r = _f_calc_with_initial_nans_2(dfg0)
    dfg1_r = _f_calc_with_initial_nans_2(dfg1)
    dfg2_r = _f_calc_with_initial_nans_2(dfg2)

    group_size_minutes = 3
    with u_t.expect_no_mutation(df):
      result_df = fc_utils.calculate_features_with_complete_coverage(
        df=df, f_calc=_f_calc_with_initial_nans_2, group_size_minutes=group_size_minutes
      )

    pd.testing.assert_series_equal(result_df.iloc[0], dfg0_r.iloc[0])
    pd.testing.assert_series_equal(result_df.iloc[1], dfg1_r.iloc[1])
    pd.testing.assert_series_equal(result_df.iloc[2], dfg1_r.iloc[2])
    pd.testing.assert_series_equal(result_df.iloc[3], dfg2_r.iloc[2])
    pd.testing.assert_series_equal(result_df.iloc[4], dfg1_r.iloc[3])
    pd.testing.assert_series_equal(result_df.iloc[5], dfg2_r.iloc[3])
    pd.testing.assert_series_equal(result_df.iloc[6], dfg2_r.iloc[4])
    pd.testing.assert_series_equal(result_df.iloc[7], dfg0_r.iloc[4])

  def test_streaming_matches_batch(self):
    def _f_calc_with_initial_nans_2(df: pd.DataFrame):
      ob = df.loc[df.index[-1], "open"]
      df.loc[df.index[-1], :] = np.nan
      df.loc[df.index[-1], "open"] = ob
      dfr = pd.DataFrame(
        index=df.index, columns=["close_squared", "high_cubed", "open_sqrt"]
      )
      dfr["close_squared"] = df["close"].shift(1).pow(2)
      dfr["high_cubed"] = df["high"].shift(1).pow(3)
      dfr["open_sqrt"] = df["open"].pow(0.5)
      return dfr

    df = _get_test_df()
    group_size_minutes = 3
    bsr = u_s.compare_batch_and_stream(
      df,
      lambda df_: fc_utils.calculate_features_with_complete_coverage(
        df=df_,
        f_calc=_f_calc_with_initial_nans_2,
        group_size_minutes=group_size_minutes,
      ),
    )
    assert bsr.dfs_match
    # Although streaming matches batch in this test case,
    # STREAMING CAN MISMATCH BATCH IN CASES LIKE nan,2,nan,3,4,5
    # IN THIS CASE STREAMING WILL GIVE nan,2,nan,3,4,5
    # BATCH WILL GIVE nan,nan,nan,3,4,5
    # THIS WILL HAPPEN RARELY SO WE ALLOW IT.
    # See test_results_have_bad_values_at_start_only which shows a case where streaming mismatches batch

  def test_results_have_bad_values_at_start_only(self):
    def _f_calc_with_initial_nans_3(df: pd.DataFrame):
      dfr = pd.DataFrame(
        index=df.index, columns=["close_rolling_sum", "high_rolling_sum"]
      )
      dfr["close_rolling_sum"] = df["close"].shift(1).rolling(window=3).sum()
      dfr["high_rolling_sum"] = df["high"].shift(1).rolling(window=3).sum()
      return dfr

    timestamps = pd.to_datetime(
      [
        # BAR GROUPINGS WITH OFFSET 0
        "2023-01-02 03:46:59.999000+00:00",
        "2023-01-02 03:47:59.999000+00:00",
        # "2023-01-02 03:48:59.999000+00:00",
        ####################################
        # "2023-01-02 03:49:59.999000+00:00",
        # "2023-01-02 03:50:59.999000+00:00",
        "2023-01-02 03:51:59.999000+00:00",
        ####################################
        "2023-01-02 03:52:59.999000+00:00",
        "2023-01-02 03:53:59.999000+00:00",
        "2023-01-02 03:54:59.999000+00:00",
        ####################################
        "2023-01-02 03:55:59.999000+00:00",
        "2023-01-02 03:56:59.999000+00:00",
        "2023-01-02 03:57:59.999000+00:00",
        ####################################
        "2023-01-02 03:58:59.999000+00:00",
      ]
    )

    np.random.seed(42)
    df = pd.DataFrame(
      {
        "open": np.random.uniform(100, 110, len(timestamps)),
        "high": np.random.uniform(111, 115, len(timestamps)),
        "low": np.random.uniform(80, 95, len(timestamps)),
        "close": np.random.uniform(100, 110, len(timestamps)),
        "volume": np.random.randint(1000, 2000, len(timestamps)),
      },
      index=timestamps,
    )
    df.index.name = "bar_close_timestamp"

    with u_t.expect_no_mutation(df):
      result_df = fc_utils.calculate_features_with_complete_coverage(
        df=df, f_calc=_f_calc_with_initial_nans_3, group_size_minutes=3
      )

    # check that bad values are at the start only
    qual = u_df.analyse_numeric_columns_quality(df=result_df)
    for col, qual in qual.items():
      assert qual.n_bad_values == qual.n_bad_values_at_start

    bsr = u_s.compare_batch_and_stream(
      df,
      lambda df_: fc_utils.calculate_features_with_complete_coverage(
        df=df_,
        f_calc=_f_calc_with_initial_nans_3,
        group_size_minutes=3,
      ),
    )
    # THIS TEST CASE IS ENGINEERED TO BE STREAM UNSAFE
    assert not bsr.dfs_match

  def test_two_f_calcs_matches_individual_results(self):
    """When two f_calcs are supplied together, their combined output should be
    identical to the column-wise concat of running each f_calc separately."""

    def _f_calc_1(df: pd.DataFrame) -> pd.DataFrame:
      dfr = pd.DataFrame(index=df.index, columns=["close_squared"])
      dfr["close_squared"] = df["close"].shift(1).pow(2)
      return dfr

    def _f_calc_2(df: pd.DataFrame) -> pd.DataFrame:
      dfr = pd.DataFrame(index=df.index, columns=["open_plus_close"])
      dfr["open_plus_close"] = df["open"] + df["close"]
      return dfr

    df = _get_test_df()
    group_size_minutes = 3

    # Combined run (both f_calcs at once)
    with u_t.expect_no_mutation(df):
      result_combined = fc_utils.calculate_features_with_complete_coverage(
        df=df,
        f_calc=[_f_calc_1, _f_calc_2],
        group_size_minutes=group_size_minutes,
      )

    # Individual runs concatenated
    with u_t.expect_no_mutation(df):
      result_1 = fc_utils.calculate_features_with_complete_coverage(
        df=df, f_calc=_f_calc_1, group_size_minutes=group_size_minutes
      )
    with u_t.expect_no_mutation(df):
      result_2 = fc_utils.calculate_features_with_complete_coverage(
        df=df, f_calc=_f_calc_2, group_size_minutes=group_size_minutes
      )
    expected = pd.concat([result_1, result_2], axis=1)

    pd.testing.assert_frame_equal(result_combined, expected)


class TestCalculateFeaturesWithLastValueGuaranteed:
  def test_last_value_matches_complete_coverage(self):
    df = _get_test_df()
    group_size_minutes = 3

    def _f_calc(df_: pd.DataFrame):
      dfr = pd.DataFrame(index=df_.index, columns=["close_squared"])
      dfr["close_squared"] = df_["close"].pow(2)
      return dfr

    with u_t.expect_no_mutation(df):
      res_last = fc_utils.calculate_features_with_last_value_guaranteed(
        df=df, f_calc=_f_calc, group_size_minutes=group_size_minutes
      )
      res_full = fc_utils.calculate_features_with_complete_coverage(
        df=df, f_calc=_f_calc, group_size_minutes=group_size_minutes
      )

    # res_full has all values
    assert np.isfinite(res_full.values).all()
    # res_last should not have all values when this test data is used
    assert not np.isfinite(res_last.values).all()
    # Verify the last row values match and are finite.
    pd.testing.assert_series_equal(res_last.iloc[-1], res_full.iloc[-1])
    assert np.isfinite(res_last.iloc[-1]).all()

  def test_streaming_lvg_matches_batch_cc(self):
    # while strategy development, we will calculate features with calculate_features_with_complete_coverage.
    # but that function is too slow for streaming (i.e. while trading live).
    # so for that, we will use calculate_features_with_last_value_guaranteed.
    # this function guarantees that just the last value is a good value.
    # this test checks that creating a result by calculating one by one using calculate_features_with_last_value_guaranteed
    # matches the result created by calculating all at once using calculate_features_with_complete_coverage.
    # this is necessary to ensure that feature values during strategy development and live trading are the same.

    # we cannot use u_s.compare_batch_and_stream here as we want to compare
    # results from different functions.
    # so we are using DfStreamer directly to test that streaming last_value_guaranteed (lvg)
    # matches batch complete_coverage (cc).
    def f_calc_lvg(df):
      # faster as only the values needed to compute the last value are calculated
      return fc_utils.calculate_features_with_last_value_guaranteed(
        df=df,
        f_calc=lambda df: fc_drsi.detrended_rsi(
          df=df, short_length=2, long_length=20, length=30
        ),
        group_size_minutes=5,
      )

    def f_calc_cc(df):
      # slower as all values are calculated
      return fc_utils.calculate_features_with_complete_coverage(
        df=df,
        f_calc=lambda df: fc_drsi.detrended_rsi(
          df=df, short_length=2, long_length=20, length=30
        ),
        group_size_minutes=5,
      )

    np.random.seed(42)
    df = u_t.get_test_df(start_date=dt.date(2023, 1, 1), end_date=dt.date(2023, 1, 2))
    streamer = u_s.DfStreamer(df)
    result_batch = f_calc_cc(df)

    # just check that there are enough good values to make the test meaningful
    assert (
      u_df.analyse_numeric_series_quality(
        result_batch["detrended_rsi_2_20_30"]
      ).n_good_values
      > 100
    )
    for i in range(len(df)):
      df_ = streamer.next()
      result_stream = f_calc_lvg(df_)
      assert result_batch.iloc[i].equals(result_stream.iloc[-1])


class TestFindMinDataPointsNeededForStableGoodLastValue:
  def test_direct_dependency_returns_one(self):
    df = _get_test_df()

    def _f_calc(df_: pd.DataFrame):
      return pd.DataFrame({"close": df_["close"]})

    assert (
      fc_utils.find_min_data_points_needed_for_stable_good_last_value(
        df=df, f_calc=_f_calc
      )
      == 1
    )

  def test_lagged_rolling_three_requires_four_rows(self):
    df = _get_test_df()

    def _f_calc(df_: pd.DataFrame):
      return pd.DataFrame({"close_roll3": df_["close"].shift(1).rolling(3).mean()})

    assert (
      fc_utils.find_min_data_points_needed_for_stable_good_last_value(
        df=df, f_calc=_f_calc
      )
      == 4
    )

  def test_an_exponential_moving_average_feature(self):
    np.random.seed(42)
    df = u_t.get_test_df(start_date=dt.date(2023, 1, 1), end_date=dt.date(2023, 2, 1))
    # just checking that the test data is large enough
    assert len(df) > 1000

    def f_calc_lvg(df):
      # faster as only the values needed to compute the last value are calculated
      return fc_utils.calculate_features_with_last_value_guaranteed(
        df=df,
        f_calc=lambda df: fc_drsi.detrended_rsi(
          df=df, short_length=2, long_length=20, length=252
        ),
        group_size_minutes=5,
      )

    def f_calc_cc(df):
      # slower as all values are calculated
      return fc_utils.calculate_features_with_complete_coverage(
        df=df,
        f_calc=lambda df: fc_drsi.detrended_rsi(
          df=df, short_length=2, long_length=20, length=252
        ),
        group_size_minutes=5,
      )

    min1 = fc_utils.find_min_data_points_needed_for_stable_good_last_value(
      df=df, f_calc=f_calc_lvg
    )

    # just checking that the test data is such that min1 is substantial
    assert min1 > 1000

    # for this usecase, using the faster version should make no difference as only the last value is needed
    # this assert verifies that
    assert min1 == fc_utils.find_min_data_points_needed_for_stable_good_last_value(
      df=df, f_calc=f_calc_cc
    )

    val_partial = f_calc_lvg(df.iloc[-min1:])["detrended_rsi_2_20_252"].values[-1]

    # just checking that the test data is such that the last value is finite and comfortably above 0
    assert np.isfinite(val_partial)
    assert np.abs(val_partial) > 1

    # this assert verifies that the min rows needed are correct
    # i.e. the last value computed using just them is the same as the last value computed using all rows
    assert np.isclose(
      val_partial,
      f_calc_lvg(df)["detrended_rsi_2_20_252"].values[-1],
    )

    # this assert verifies that the min1 is indeed the min rows needed
    # it computes the last value using min1-1 rows and verifies that it differs from the last value computed using all rows
    # => min1-1 is not sufficient to match the last value computed using all rows
    # => min1 is indeed the min
    assert not np.isclose(
      f_calc_lvg(df.iloc[-(min1 - 1) :])["detrended_rsi_2_20_252"].values[-1],
      f_calc_lvg(df)["detrended_rsi_2_20_252"].values[-1],
    )
