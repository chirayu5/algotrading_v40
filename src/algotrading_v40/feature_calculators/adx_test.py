import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.feature_calculators.adx as fc_adx
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us


class TestAdxStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    np.random.seed(42)
    lookback = 14

    inst = sid.InstrumentDesc(
      market=sid.Market.INDIAN_MARKET,
      symbol="ABCD",
    )
    data = das.get_synthetic_data(
      instrument_descs=[inst],
      date_range=sdr.DateRange(
        start_date=dt.date(2023, 1, 2),
        end_date=dt.date(2023, 1, 3),
      ),
    )
    df = data.get_full_df_for_instrument_desc(inst).copy()
    df["volume"] = 1.0  # dummy volume column
    dfb = df.copy()
    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_adx.adx(
        df_,
        lookback=lookback,
      ),
    )
    pd.testing.assert_frame_equal(df, dfb)
    expected_col = f"adx_{lookback}"
    assert result.df_batch.columns == [expected_col]
    assert result.df_batch.index.equals(df.index)

    q = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert q.n_good_values >= 375
    assert q.n_bad_values_at_end == 0
    assert q.n_bad_values_at_start == q.n_bad_values
    print("q.n_bad_values_at_start", q.n_bad_values_at_start)
    assert result.dfs_match, (
      "Batch and streaming ADX results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
