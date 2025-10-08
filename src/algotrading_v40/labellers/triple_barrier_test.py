import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.constants as ctnts
import algotrading_v40.labellers.triple_barrier as ltb
import algotrading_v40.trading_time_elapsed_calculators.with_overnight_gaps_only as ttec_wogo


def calc_ret(v0: float, v1: float) -> float:
  return v1 / v0 - 1


class TestValidateAndRunTripleBarrier:
  def _create_datetime_index(self, n_periods: int) -> pd.DatetimeIndex:
    """Create a DatetimeIndex in the format specified by the user."""
    return pd.date_range(
      start="2023-01-02 03:45:59.999000+00:00",
      periods=n_periods,
      freq="1min",
      tz="UTC",
      name="date",
    )

  def test_long_take_profit_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 1.01, 0.99, 0.98, 1.05, 1.02],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.03] * 6, index=index)
    # can't be the length of the series (i.e. 5) as that would make the last
    # vertical barrier the same as the index which is not allowed
    vb_tte = pd.Series([6] * 6, index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [4, np.nan, 4, 4, np.nan, np.nan],
        "first_touch_type": [1, np.nan, 1, 1, np.nan, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    # print(result.to_string())
    # print(expected.to_string())
    pd.testing.assert_frame_equal(result, expected)

  def test_long_stop_loss_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 1.01, 0.99, 0.98, 1.05, 1.02],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb_tte = pd.Series([6] * 6, index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4     2  <NA>               2                -1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2  <NA>               2                -1
    # 2023-01-02 03:47:59.999000+00:00     4     3  <NA>               3                -1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5  <NA>               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [2, 2, 3, np.nan, 5, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [2, 2, 3, 4, 5, np.nan],
        "first_touch_type": [-1, -1, -1, 1, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_take_profit_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 0.99, 1.01, 1.02, 0.95, 0.98],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.03] * 6, index=index)
    vb_tte = pd.Series([6] * 6, index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     3  <NA>               3                -1
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5  <NA>               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [np.nan, 3, np.nan, np.nan, 5, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [4, 3, 4, 4, 5, np.nan],
        "first_touch_type": [1, -1, 1, 1, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_stop_loss_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 0.99, 1.01, 1.02, 0.95, 0.98],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb_tte = pd.Series([6] * 6, index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4     2  <NA>               2                -1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2  <NA>               2                -1
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5  <NA>               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [2, 2, np.nan, np.nan, 5, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [2, 2, 4, 4, 5, np.nan],
        "first_touch_type": [-1, -1, 1, 1, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_long_vertical_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 1.01, 1, 1.01, 1.05, 1.02],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb_tte = pd.Series([3, 2, 7, 2, 3, 2], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4  <NA>     3               3                 0
    # 2023-01-02 03:46:59.999000+00:00  <NA>  <NA>     3               3                 0
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>  <NA>               4                 1
    # 2023-01-02 03:48:59.999000+00:00  <NA>  <NA>     5               5                 0
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5  <NA>               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, np.nan, np.nan, np.nan],
        "slha": [np.nan, np.nan, np.nan, np.nan, 5, np.nan],
        "vbha": [3, 3, np.nan, 5, np.nan, np.nan],
        "first_touch_at": [3, 3, 4, 5, 5, np.nan],
        "first_touch_type": [0, 0, 1, 0, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_vertical_barrier(self):
    index = self._create_datetime_index(6)
    prices = pd.Series(
      [1, 0.99, 1, 0.99, 0.95, 0.98],
      index=index,
    )
    selected = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.01] * 6, index=index)
    slb = pd.Series([-0.05] * 6, index=index)
    vb_tte = pd.Series([3, 2, 1, 2, 3, 1], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1  <NA>     3               1                 1
    # 2023-01-02 03:46:59.999000+00:00     4  <NA>     3               3                 0
    # 2023-01-02 03:47:59.999000+00:00     3  <NA>     3               3                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, 4, 3, 4, np.nan, np.nan],
        "slha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "vbha": [3, 3, 3, 5, np.nan, np.nan],
        "first_touch_at": [1, 3, 3, 4, np.nan, np.nan],
        "first_touch_type": [1, 0, 1, 1, np.nan, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed(self):
    index = self._create_datetime_index(5)
    prices = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    selected = pd.Series([1, 1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb_tte = pd.Series([5, 4, 3, 2, 1], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2  <NA>               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2  <NA>               2                -1
    # 2023-01-02 03:47:59.999000+00:00     3     4  <NA>               3                 1
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4  <NA>               4                -1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, 3, np.nan, np.nan],
        "slha": [2, 2, 4, 4, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [1, 2, 3, 4, np.nan],
        "first_touch_type": [1, -1, 1, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed_with_inc_0(self):
    index = self._create_datetime_index(5)
    prices = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    selected = pd.Series([1, 1, 0, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb_tte = pd.Series([5, 4, 3, 2, 1], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2  <NA>               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2  <NA>               2                -1
    # 2023-01-02 03:47:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4  <NA>               4                -1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, np.nan, np.nan, np.nan],
        "slha": [2, 2, np.nan, 4, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [1, 2, np.nan, 4, np.nan],
        "first_touch_type": [1, -1, np.nan, -1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed_with_vb_same_as_index(self):
    index = self._create_datetime_index(5)
    prices = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    selected = pd.Series([1, 1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb_tte = pd.Series([4, 3, 0, 1, 0], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                    tpha  slha  vbha  first_touch_at  first_touch_type  first_touch_raw_return
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2     4               1                 1           -0.060000
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     4               2                -1            0.106383
    # 2023-01-02 03:47:59.999000+00:00     3     4     2               2                 0            0.000000
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4     4               4                -1           -0.181818
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>     4               4                 0            0.000000
    assert result["vbha"].to_list() == [4, 4, 2, 4, 4]

  def test_variable_barriers(self):
    index = self._create_datetime_index(4)

    prices = pd.Series(
      [1.00, 1.05, 0.95, 1.08],
      index=index,
    )
    selected = pd.Series([1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.04, 0.03, 0.06, 0.05], index=index)
    slb = pd.Series([-0.02, -0.01, -0.03, -0.05], index=index)
    vb_tte = pd.Series([4, 3, 2, 1], index=index)
    tte = ttec_wogo.with_overnight_gaps_only(index, 0)
    side = pd.Series([1] * 4, index=index, dtype="Int64")

    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2  <NA>               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2  <NA>               2                -1
    # 2023-01-02 03:47:59.999000+00:00     3  <NA>  <NA>               3                 1
    # 2023-01-02 03:48:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, 3, np.nan],
        "slha": [2, 2, np.nan, np.nan],
        "vbha": [np.nan, np.nan, np.nan, np.nan],
        "first_touch_at": [1, 2, 3, np.nan],
        "first_touch_type": [1, -1, 1, np.nan],
      },
      index=index,
    ).astype(
      {
        "tpha": "Int32",
        "slha": "Int32",
        "vbha": "Int32",
        "first_touch_at": "Int32",
        "first_touch_type": "Int32",
      }
    )
    expected["first_touch_raw_return"] = expected.apply(
      lambda row: calc_ret(prices.loc[row.name], prices.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_vertical_barrier_with_overnight_gap(self):
    CURR_DAY = dt.date(2025, 9, 15)
    NEXT_DAY = dt.date(2025, 9, 18)

    def _market_ts(minutes_after_open: int, day: dt.date) -> pd.Timestamp:
      """
      Convenience: return a Timestamp <minutes_after_open> minutes after the
      first (minute-bar-close) timestamp of the trading session, in UTC.
      """
      base = pd.Timestamp.combine(
        day, ctnts.INDIAN_MARKET_FIRST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
      ).tz_localize("UTC")
      border = pd.Timestamp.combine(
        day, ctnts.INDIAN_MARKET_LAST_MINUTE_BAR_CLOSE_TIMESTAMP_UTC
      ).tz_localize("UTC")
      result = base + pd.Timedelta(minutes=minutes_after_open)
      if result > border:
        raise ValueError(f"minutes_after_open {minutes_after_open} is too large")
      return result

    index = pd.DatetimeIndex(
      [
        _market_ts(2, CURR_DAY),
        _market_ts(373, CURR_DAY),
        _market_ts(374, CURR_DAY),
        _market_ts(4, NEXT_DAY),
        _market_ts(5, NEXT_DAY),
        _market_ts(14, NEXT_DAY),
      ]
    )
    prices = pd.Series(
      [100, 103, 95, 97, 101, 102],
      index=index,
    )
    selected = pd.Series([1, 1, 1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.04, 0.03, 0.06, 0.05, 0.02, 0.01], index=index)
    slb = pd.Series([-0.02, -0.01, -0.03, -0.05, -0.01, -0.02], index=index)
    vb_tte = pd.Series([372, 6, 6, 11, 8, 0], index=index)
    # overnight_gap_minutes is 1065 minutes for standard Indian-market timings
    tte = ttec_wogo.with_overnight_gaps_only(index, overnight_gap_minutes=1065)
    side = pd.Series([1] * 6, index=index, dtype="Int64")

    result = ltb.triple_barrier(
      prices=prices,
      selected=selected,
      tpb=tpb,
      slb=slb,
      vb_tte=vb_tte,
      tte=tte,
      side=side,
    )
    #                                    tpha  slha  vbha  first_touch_at  first_touch_type  first_touch_raw_return
    # 2025-09-15 03:47:59.999000+00:00  <NA>     2     2               2                -1           -0.050000
    # 2025-09-15 09:58:59.999000+00:00  <NA>     2     3               2                -1           -0.077670
    # 2025-09-15 09:59:59.999000+00:00     4  <NA>     4               4                 1            0.063158
    # 2025-09-18 03:49:59.999000+00:00     5  <NA>  <NA>               5                 1            0.051546
    # 2025-09-18 03:50:59.999000+00:00  <NA>  <NA>     5               5                 0            0.009901
    # 2025-09-18 03:59:59.999000+00:00  <NA>  <NA>     5               5                 0            0.000000
    np.testing.assert_array_equal(result["vbha"].to_numpy(), [2, 3, 4, np.nan, 5, 5])
