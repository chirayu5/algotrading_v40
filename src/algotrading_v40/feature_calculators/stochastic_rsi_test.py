import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.feature_calculators.stochastic_rsi as fc_srsi
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us


class TestStochasticRsiStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    """
    Stochastic RSI computed on the full series must match the
    values obtained when data arrive incrementally.
    """
    np.random.seed(42)
    rsi_lookback = 14
    stoch_lookback = 14
    n_to_smooth = 10

    inst_desc = sid.InstrumentDesc(
      market=sid.Market.INDIAN_MARKET,
      symbol="ABCD",
    )
    data = das.get_synthetic_data(
      instrument_descs=[inst_desc],
      date_range=sdr.DateRange(
        start_date=dt.date(2023, 1, 2),
        end_date=dt.date(2023, 1, 3),
      ),
    )

    df = data.get_full_df_for_instrument_desc(inst_desc).copy()
    df["volume"] = 1.0  # dummy volume column
    dfb = df.copy()

    result = us.compare_batch_and_stream(
      df,
      lambda d: fc_srsi.stochastic_rsi(
        d,
        rsi_lookback=rsi_lookback,
        stoch_lookback=stoch_lookback,
        n_to_smooth=n_to_smooth,
      ),
    )

    pd.testing.assert_frame_equal(df, dfb)  # input not modified
    expected_col = f"stochastic_rsi_{rsi_lookback}_{stoch_lookback}_{n_to_smooth}"
    assert result.df_batch.columns == [expected_col]
    assert result.df_batch.index.equals(df.index)

    quality = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert quality.n_good_values >= 375
    assert quality.n_bad_values_at_end == 0
    assert quality.n_bad_values_at_start == quality.n_bad_values
    print("quality.n_bad_values_at_start", quality.n_bad_values_at_start)

    assert result.dfs_match, (
      "Batch and streaming Stochastic RSI diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
