import datetime as dt

import algotrading_v40.feature_calculators.stochastic_rsi as fc_srsi
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


class TestStochasticRsiStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    """
    Stochastic RSI computed on the full series must match the
    values obtained when data arrive incrementally.
    """
    rsi_lookback = 14
    stoch_lookback = 14
    n_to_smooth = 10

    df = ut.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )

    with ut.expect_no_mutation(df):
      result = us.compare_batch_and_stream(
        df,
        lambda d: fc_srsi.stochastic_rsi(
          d,
          rsi_lookback=rsi_lookback,
          stoch_lookback=stoch_lookback,
          n_to_smooth=n_to_smooth,
        ),
      )

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
