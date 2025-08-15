import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.feature_calculators.rsi as fc_rsi
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us


class TestRsiStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    """
    The RSI calculated on the full price series should be identical to the
    values obtained when the same data arrive one point at a time.
    """
    np.random.seed(42)
    lookback = 20
    id = sid.InstrumentDesc(
      market=sid.Market.INDIAN_MARKET,
      symbol="ABCD",
    )
    data = das.get_synthetic_data(
      instrument_descs=[id],
      date_range=sdr.DateRange(
        start_date=dt.date(2023, 1, 2),
        end_date=dt.date(2023, 1, 3),
      ),
    )

    df = data.get_full_df_for_instrument_desc(id).copy()
    df["volume"] = 1.0  # dummy volume column
    dfb = df.copy()
    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_rsi.rsi(df_, lookback),
    )
    # input data should not be modified
    pd.testing.assert_frame_equal(df, dfb)
    assert result.df_batch.columns == [f"rsi_{lookback}"]
    # result should have the same index as the input data
    assert result.df_batch.index.equals(df.index)
    result_quality = udf.analyse_numeric_series_quality(
      result.df_batch[f"rsi_{lookback}"]
    )
    # at least 1 full session's data should be good and compared with streaming
    assert result_quality.n_good_values >= 375
    # there should be no bad values at the end
    assert result_quality.n_bad_values_at_end == 0
    # all bad values should be at the start
    assert result_quality.n_bad_values_at_start == result_quality.n_bad_values
    print("result_quality.n_bad_values_at_start", result_quality.n_bad_values_at_start)
    # batch and streaming results should match
    assert result.dfs_match, (
      "Batch and streaming RSI results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
