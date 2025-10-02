import datetime as dt

import numpy as np
import pandas as pd
import pytest

import algotrading_v40.constants as ctnts
import algotrading_v40.labellers.triple_barrier as l_tb
import algotrading_v40.sample_weighers.concurrency_return_age_adjusted as sw_craa
import algotrading_v40.utils.df as u_df
import algotrading_v40.utils.testing as u_t


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _market_ts(
  minutes_after_open: int, day: dt.date = dt.date(2025, 9, 15)
) -> pd.Timestamp:
  """
  Convenience: return a Timestamp <minutes_after_open> minutes after the
  first (minute-bar-close) timestamp of the trading session, in UTC.
  """
  base = pd.Timestamp.combine(
    day, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
  ).tz_localize("UTC")
  return base + pd.Timedelta(minutes=minutes_after_open)


def _create_test_df(
  *,
  timestamps: list[pd.Timestamp],
  prices: list[float],
  selected: list[int],
  label_last_indices: list[float | int],
) -> pd.DataFrame:
  """Helper to create a test DataFrame with required columns."""
  return pd.DataFrame(
    {
      "close": prices,
      "selected": selected,
      "label_last_index": label_last_indices,
    },
    index=pd.DatetimeIndex(timestamps, name="bar_close_timestamp"),
  )


class TestConcurrencyReturnAgeAdjustedWeights:
  def test_no_time_decay(self):
    timestamps = [_market_ts(i) for i in range(9)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[103.0, 106.0, 100.0, 104.0, 101.0, 105.0, 102.0, 97.0, 90.0],
      selected=[1, 0, 1, 0, 0, 0, 1, 0, 1],
      label_last_indices=[4, np.nan, 5, np.nan, np.nan, np.nan, 8, np.nan, np.nan],
    )

    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #   print("df\n", df)

    #                                    close  selected  label_last_index
    # bar_close_timestamp
    # 2025-09-15 03:45:59.999000+00:00  103.0         1               4.0
    # 2025-09-15 03:46:59.999000+00:00  106.0         0               NaN
    # 2025-09-15 03:47:59.999000+00:00  100.0         1               5.0
    # 2025-09-15 03:48:59.999000+00:00  104.0         0               NaN
    # 2025-09-15 03:49:59.999000+00:00  101.0         0               NaN
    # 2025-09-15 03:50:59.999000+00:00  105.0         0               NaN
    # 2025-09-15 03:51:59.999000+00:00  102.0         1               8.0
    # 2025-09-15 03:52:59.999000+00:00   97.0         0               NaN
    # 2025-09-15 03:53:59.999000+00:00   90.0         1               NaN

    with u_t.expect_no_mutation(df):
      result = sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #   print("result\n", result)

    pd.testing.assert_index_equal(result.index, df.index)

    np.testing.assert_array_equal(
      result["concurrency"].values,
      [
        0,
        1,  # from label at 3:45:59
        1,  # from label at 3:45:59
        2,  # from label at 3:45:59 AND 3:47:59
        2,  # from label at 3:45:59 AND 3:47:59
        1,  # from label at             3:47:59
        0,
        1,  # from label at                         3:51:59
        1,  # from label at                         3:51:59
      ],
    )

    np.testing.assert_array_equal(
      result["avg_uniqueness"].values,
      [
        0.75,
        np.nan,
        2 / 3,
        np.nan,
        np.nan,
        np.nan,
        1,
        np.nan,
        np.nan,
      ],
    )

    np.testing.assert_allclose(
      result["attribution_weight_raw"].values,
      [
        # Expected attribution weight for label at index 0:
        # Uses returns from prices[0→4]: [103→106, 106→100, 100→104 (50%), 104→101 (50%)]
        # Concurrency: [0, 1, 1, 2, 2] at indices [0, 1, 2, 3, 4]
        np.abs(
          np.log(106 / 103)  # full return at index 1, concurrency=1
          + np.log(100 / 106)  # full return at index 2, concurrency=1
          + 0.5 * np.log(104 / 100)  # half return at index 3, concurrency=2
          + 0.5 * np.log(101 / 104)  # half return at index 4, concurrency=2
        ),
        np.nan,
        np.abs(0.5 * np.log(104 / 100) + 0.5 * np.log(101 / 104) + np.log(105 / 101)),
        np.nan,
        np.nan,
        np.nan,
        np.abs(np.log(97 / 102) + np.log(90 / 97)),
        np.nan,
        np.nan,
      ],
    )

    assert (
      (result["time_decay_factor"] == 1.0) | (result["time_decay_factor"].isna())
    ).all()

    np.testing.assert_array_equal(
      result["attribution_weight_raw"].values, result["sample_weight"].values
    )

  @pytest.mark.parametrize("time_decay_c", [0, 0.5, 0.9])
  def test_with_time_decay(self, time_decay_c):
    timestamps = [_market_ts(i) for i in range(9)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[103.0, 106.0, 100.0, 104.0, 101.0, 105.0, 102.0, 97.0, 90.0],
      selected=[1, 0, 1, 0, 0, 0, 1, 0, 1],
      label_last_indices=[4, np.nan, 5, np.nan, np.nan, np.nan, 8, np.nan, np.nan],
    )

    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #   print("df\n", df)

    #                                    close  selected  label_last_index
    # bar_close_timestamp
    # 2025-09-15 03:45:59.999000+00:00  103.0         1               4.0
    # 2025-09-15 03:46:59.999000+00:00  106.0         0               NaN
    # 2025-09-15 03:47:59.999000+00:00  100.0         1               5.0
    # 2025-09-15 03:48:59.999000+00:00  104.0         0               NaN
    # 2025-09-15 03:49:59.999000+00:00  101.0         0               NaN
    # 2025-09-15 03:50:59.999000+00:00  105.0         0               NaN
    # 2025-09-15 03:51:59.999000+00:00  102.0         1               8.0
    # 2025-09-15 03:52:59.999000+00:00   97.0         0               NaN
    # 2025-09-15 03:53:59.999000+00:00   90.0         1               NaN

    with u_t.expect_no_mutation(df):
      result = sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=time_decay_c,
      )

    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #   print("result\n", result)
    pd.testing.assert_index_equal(result.index, df.index)

    good_td = result[result["time_decay_factor"].notna()]["time_decay_factor"].values
    assert len(good_td) == 3
    # good_td[0] is the oldest data point so its time decay factor should be the lowest (but still > 0)
    # good_td[2] is the newest data point so its time decay factor should be the highest (but still <= 1)
    assert 0 < good_td[0] < good_td[1] < good_td[2] <= 1

    # check that (avg_uniqueness[0],good_td[0] | avg_uniqueness[0]+avg_uniqueness[1],good_td[1] |
    # avg_uniqueness[0]+avg_uniqueness[1]+avg_uniqueness[2],good_td[2])
    # lie on a straight line
    good_au = result[result["avg_uniqueness"].notna()]["avg_uniqueness"].values
    cumulative_uniqueness = [
      good_au[0],
      good_au[0] + good_au[1],
      good_au[0] + good_au[1] + good_au[2],
    ]

    expected_td_1 = np.interp(
      cumulative_uniqueness[1],
      [cumulative_uniqueness[0], cumulative_uniqueness[2]],
      [good_td[0], good_td[2]],
    )
    np.testing.assert_allclose(good_td[1], expected_td_1, rtol=1e-10)

    np.testing.assert_allclose(
      result["sample_weight"].values,
      np.multiply(
        result["attribution_weight_raw"].values,
        result["time_decay_factor"].values,
      ),
    )

  def test_all_selected_with_nan_lli(self):
    """Test when all selected samples have NaN label_last_index."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 1, 1],
      label_last_indices=[np.nan, np.nan, np.nan],
    )
    with pytest.raises(ValueError, match="No non-NaN label_last_index found"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_invalid_selected_values(self):
    """Test that selected must only contain 0 or 1."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 2, 0],  # Invalid: contains 2
      label_last_indices=[2, 2, np.nan],
    )

    with pytest.raises(ValueError, match="selected must only contain values 0 or 1"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_invalid_prices(self):
    """Test that prices must be positive."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 0.0, 102.0],  # Invalid: contains 0
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )

    with pytest.raises(ValueError, match="prices must be greater than 0"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, -101.0, 102.0],  # Invalid: negative
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )

    with pytest.raises(ValueError, match="prices must be greater than 0"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, np.nan, 102.0],  # Invalid: negative
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )

    with pytest.raises(ValueError, match="prices must not have bad values"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_invalid_time_decay_c(self):
    """Test that time_decay_c must be in [0,1]."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )

    with pytest.raises(ValueError, match="time_decay_c must be a finite number in"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=-0.1,
      )

    with pytest.raises(ValueError, match="time_decay_c must be a finite number in"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.1,
      )

  def test_invalid_label_last_index(self):
    """Test that negative label_last_index raises error."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 0, 0],
      label_last_indices=[-1, np.nan, np.nan],  # Invalid: negative
    )

    with pytest.raises(RuntimeError, match="Invalid values for index"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 0, 0],
      label_last_indices=[3, np.nan, np.nan],  # Invalid: >= len(df)
    )

    with pytest.raises(RuntimeError, match="Invalid values for index"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    timestamps = [_market_ts(i) for i in range(4)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0, 103.0],
      selected=[0, 1, 0, 0],
      label_last_indices=[np.nan, 1, np.nan, np.nan],  # Invalid: lli[1]=1 but needs > 1
    )

    with pytest.raises(RuntimeError, match="Invalid values for index"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_no_selected_samples_raises_error(self):
    """Test that all selected=0 raises error."""
    timestamps = [_market_ts(i) for i in range(3)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[0, 0, 0],
      label_last_indices=[np.nan, np.nan, np.nan],
    )

    with pytest.raises(ValueError, match="No significant samples found"):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_selected_0_with_lli_nan_fails(self):
    """Test that selected=0 must have NaN label_last_index."""
    timestamps = [_market_ts(i) for i in range(4)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0, 103.0],
      selected=[1, 0, 0, 0],
      label_last_indices=[2, 3, np.nan, np.nan],  # Invalid: selected=0 but lli=3
    )

    with pytest.raises(
      ValueError,
      match="selected=0 cases must have all values bad for label_last_indices",
    ):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_index_not_monotonic_increasing_or_unique(self):
    """Test that index mismatch raises error."""
    timestamps = [_market_ts(1), _market_ts(0), _market_ts(2)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )
    with pytest.raises(
      ValueError, match="index must be monotonic increasing and unique"
    ):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

    timestamps = [_market_ts(0), _market_ts(0), _market_ts(2)]
    df = _create_test_df(
      timestamps=timestamps,
      prices=[100.0, 101.0, 102.0],
      selected=[1, 0, 0],
      label_last_indices=[2, np.nan, np.nan],
    )
    with pytest.raises(
      ValueError, match="index must be monotonic increasing and unique"
    ):
      sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df["label_last_index"],
        prices=df["close"],
        time_decay_c=1.0,
      )

  def test_integration_with_triple_barrier_labelling(self):
    df = (
      u_t.get_test_df(
        start_date=dt.date(2025, 9, 15),
        end_date=dt.date(2025, 9, 19),
      )
      * 4
    )
    df.drop(columns=["open", "high", "low", "volume"], inplace=True)
    df["vol"] = np.log(df["close"].shift(1)).diff().ewm(span=30).std()

    df["tpb"] = df["vol"] * 10
    df["slb"] = df["vol"] * -10

    df["selected"] = np.random.randint(0, 2, len(df))
    df["side"] = 1
    df["vb"] = np.arange(len(df)) + 45

    df = df.iloc[20:]

    with u_t.expect_no_mutation(df):
      df_tb = l_tb.triple_barrier(
        prices=df["close"],
        selected=df["selected"],
        tpb=df["tpb"],
        slb=df["slb"],
        vb=df["vb"],
        side=df["side"],
      )
    # print(df_tb["first_touch_type"].value_counts(dropna=False))

    with u_t.expect_no_mutation(df, df_tb):
      df_craa = sw_craa.concurrency_return_age_adjusted_weights(
        selected=df["selected"],
        label_last_indices=df_tb["first_touch_at"],
        prices=df["close"],
        time_decay_c=0.5,
      )

    swg = df_craa["sample_weight"].loc[
      u_df.analyse_numeric_series_quality(df_craa["sample_weight"]).good_values_mask
    ]
    ssw = swg.sum()
    n = len(swg)

    df_craa["sample_weight"] = n * df_craa["sample_weight"] / ssw
