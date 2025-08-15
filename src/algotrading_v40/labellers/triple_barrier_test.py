import numpy as np
import pandas as pd

import algotrading_v40.labellers.triple_barrier as ltb


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
    s = pd.Series(
      [1, 1.01, 0.99, 0.98, 1.05, 1.02],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.03] * 6, index=index)
    vb = pd.Series([5] * 6, index=index)
    side = pd.Series([1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>  <NA>     5               5                 0
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>     5               5                 0
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "vbha": [5, 5, 5, 5, 5, np.nan],
        "first_touch_at": [4, 5, 4, 4, 5, np.nan],
        "first_touch_type": [1, 0, 1, 1, 0, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    # print(result.to_string())
    # print(expected.to_string())
    pd.testing.assert_frame_equal(result, expected)

  def test_long_stop_loss_barrier(self):
    index = self._create_datetime_index(6)
    s = pd.Series(
      [1, 1.01, 0.99, 0.98, 1.05, 1.02],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb = pd.Series([5] * 6, index=index)
    side = pd.Series([1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4     2     5               2                -1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     5               2                -1
    # 2023-01-02 03:47:59.999000+00:00     4     3     5               3                -1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5     5               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [2, 2, 3, np.nan, 5, np.nan],
        "vbha": [5, 5, 5, 5, 5, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_take_profit_barrier(self):
    index = self._create_datetime_index(6)
    s = pd.Series(
      [1, 0.99, 1.01, 1.02, 0.95, 0.98],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.03] * 6, index=index)
    vb = pd.Series([5] * 6, index=index)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     3     5               3                -1
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5     5               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [np.nan, 3, np.nan, np.nan, 5, np.nan],
        "vbha": [5, 5, 5, 5, 5, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_stop_loss_barrier(self):
    index = self._create_datetime_index(6)
    s = pd.Series(
      [1, 0.99, 1.01, 1.02, 0.95, 0.98],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb = pd.Series([5] * 6, index=index)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     4     2     5               2                -1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     5               2                -1
    # 2023-01-02 03:47:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:48:59.999000+00:00     4  <NA>     5               4                 1
    # 2023-01-02 03:49:59.999000+00:00  <NA>     5     5               5                -1
    # 2023-01-02 03:50:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [4, np.nan, 4, 4, np.nan, np.nan],
        "slha": [2, 2, np.nan, np.nan, 5, np.nan],
        "vbha": [5, 5, 5, 5, 5, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_long_vertical_barrier(self):
    index = self._create_datetime_index(6)
    s = pd.Series(
      [1, 1.01, 1, 1.01, 1.05, 1.02],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.05] * 6, index=index)
    slb = pd.Series([-0.01] * 6, index=index)
    vb = pd.Series([3, 3, 9, 5, 7, 7], index=index)
    side = pd.Series([1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_short_vertical_barrier(self):
    index = self._create_datetime_index(6)
    s = pd.Series(
      [1, 0.99, 1, 0.99, 0.95, 0.98],
      index=index,
    )
    inc = pd.Series([1] * 6, index=index)
    tpb = pd.Series([0.01] * 6, index=index)
    slb = pd.Series([-0.05] * 6, index=index)
    vb = pd.Series([3, 3, 3, 5, 7, 5], index=index)
    side = pd.Series([-1] * 6, index=index)
    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed(self):
    index = self._create_datetime_index(5)
    s = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    inc = pd.Series([1, 1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb = pd.Series([4] * 5, index=index)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2     4               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     4               2                -1
    # 2023-01-02 03:47:59.999000+00:00     3     4     4               3                 1
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4     4               4                -1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, 3, np.nan, np.nan],
        "slha": [2, 2, 4, 4, np.nan],
        "vbha": [4, 4, 4, 4, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed_with_inc_0(self):
    index = self._create_datetime_index(5)
    s = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    inc = pd.Series([1, 1, 0, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb = pd.Series([4] * 5, index=index)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2     4               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     4               2                -1
    # 2023-01-02 03:47:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4     4               4                -1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, np.nan, np.nan, np.nan],
        "slha": [2, 2, np.nan, 4, np.nan],
        "vbha": [4, 4, np.nan, 4, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_mixed_with_vb_same_as_index(self):
    index = self._create_datetime_index(5)
    s = pd.Series(
      [1.00, 0.94, 1.04, 1.10, 0.90],
      index=index,
    )
    inc = pd.Series([1, 1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.05] * 5, index=index)
    slb = pd.Series([-0.03] * 5, index=index)
    vb = pd.Series([4, 4, 2, 4, 4], index=index)
    side = pd.Series([-1, -1, 1, 1, -1], index=index, dtype="Int64")

    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2     4               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     4               2                -1
    # 2023-01-02 03:47:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    # 2023-01-02 03:48:59.999000+00:00  <NA>     4     4               4                -1
    # 2023-01-02 03:49:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, np.nan, np.nan, np.nan],
        "slha": [2, 2, np.nan, 4, np.nan],
        "vbha": [4, 4, np.nan, 4, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)

  def test_variable_barriers(self):
    index = self._create_datetime_index(4)

    s = pd.Series(
      [1.00, 1.05, 0.95, 1.08],
      index=index,
    )
    inc = pd.Series([1, 1, 1, 1], index=index, dtype="Int64")
    tpb = pd.Series([0.04, 0.03, 0.06, 0.05], index=index)
    slb = pd.Series([-0.02, -0.01, -0.03, -0.05], index=index)
    vb = pd.Series([3] * 4, index=index)
    side = pd.Series([1] * 4, index=index, dtype="Int64")

    result = ltb.validate_and_run_triple_barrier(s, inc, tpb, slb, vb, side)
    #                                   tpha  slha  vbha  first_touch_at  first_touch_type
    # date
    # 2023-01-02 03:45:59.999000+00:00     1     2     3               1                 1
    # 2023-01-02 03:46:59.999000+00:00  <NA>     2     3               2                -1
    # 2023-01-02 03:47:59.999000+00:00     3  <NA>     3               3                 1
    # 2023-01-02 03:48:59.999000+00:00  <NA>  <NA>  <NA>            <NA>              <NA>
    expected = pd.DataFrame(
      {
        "tpha": [1, np.nan, 3, np.nan],
        "slha": [2, 2, np.nan, np.nan],
        "vbha": [3, 3, 3, np.nan],
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
    expected["first_touch_return"] = expected.apply(
      lambda row: calc_ret(s.loc[row.name], s.iloc[row["first_touch_at"]])
      if pd.notna(row["first_touch_at"])
      else np.nan,
      axis=1,
    ).astype("float32")
    pd.testing.assert_frame_equal(result, expected)
