import numpy as np
import pandas as pd
import pytest

import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


def _make_df(n: int = 5) -> pd.DataFrame:
  """Utility to create a tiny OHLCV DataFrame with UTC‐tz index."""
  idx = pd.date_range("2023-01-01 09:15", periods=n, freq="1min", tz="UTC")
  df = pd.DataFrame(
    {
      "open": np.arange(n, dtype=float),
      "high": np.arange(n, dtype=float) + 0.5,
      "low": np.arange(n, dtype=float) - 0.5,
      "close": np.arange(n, dtype=float) + 0.2,
      "volume": np.arange(n, dtype=float) * 100,
    },
    index=idx,
  )
  return df


class TestDfStreamer:
  def test_invalid_columns_raises(self) -> None:
    df_bad = _make_df().drop(columns="volume")
    with pytest.raises(ValueError, match="must have columns"):
      us.DfStreamer(df_bad)

  def test_next_returns_growing_frame_and_nan_mask(self) -> None:
    df = _make_df(4)
    streamer = us.DfStreamer(df)

    for i in range(len(df)):
      out = streamer.next()
      # Length grows by 1 each iteration
      assert len(out) == i + 1
      # All rows except last are identical to original slice
      pd.testing.assert_frame_equal(out.iloc[:-1], df.iloc[:i], check_dtype=False)
      # In last row, every column except "open" must be NaN
      last_row = out.iloc[-1]
      assert pd.isna(last_row[["high", "low", "close", "volume"]]).all()
      assert last_row["open"] == df.iloc[i]["open"]


class TestCompareBatchAndStreamResult:
  @pytest.mark.parametrize(
    "batch_mm, stream_mm, dfs_match",
    [
      (None, pd.DataFrame(), True),
      (pd.DataFrame(), None, True),
      (None, None, False),
    ],
  )
  def test_invalid_combinations_raise(self, batch_mm, stream_mm, dfs_match) -> None:
    with pytest.raises(ValueError):
      us.CompareBatchAndStreamResult(
        df_batch=pd.DataFrame(),
        df_batch_mismatch=batch_mm,
        df_stream_mismatch=stream_mm,
        dfs_match=dfs_match,
      )


class TestCompareBatchAndStream:
  def test_perfect_match(self) -> None:
    df = _make_df(6)

    # A simple, stream-safe function: double the open price
    def func(d: pd.DataFrame) -> pd.DataFrame:
      return pd.DataFrame({"open_x2": d["open"] * 2}, index=d.index)

    with ut.expect_no_mutation(df):
      result = us.compare_batch_and_stream(df, func)

    assert result.dfs_match is True
    assert result.df_batch_mismatch is None
    assert result.df_stream_mismatch is None

    # df_batch should equal applying func once to full df
    pd.testing.assert_frame_equal(result.df_batch, func(df))

  def test_first_mismatch_detected(self) -> None:
    df = _make_df(4)

    # This function is *not* stream-safe because it relies on the
    # latest `close` values which become NaN in the streaming case.
    def func(d: pd.DataFrame) -> pd.DataFrame:
      return d[["open", "close"]]

    result = us.compare_batch_and_stream(df, func)

    assert result.dfs_match is False
    # The mismatch should be detected on the first streaming step
    assert len(result.df_batch_mismatch) == 1
    # Batch slice equals original function output
    pd.testing.assert_frame_equal(result.df_batch_mismatch, func(df).iloc[:1])
    # Stream mismatch should have NaN in 'close'
    assert pd.isna(result.df_stream_mismatch.iloc[-1]["close"])
