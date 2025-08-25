import datetime as dt

import algotrading_v40.feature_calculators.macd as fc_macd
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


class TestMacdStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    short_length = 10
    long_length = 100
    n_to_smooth = 5

    df = ut.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )

    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_macd.macd(
        df_,
        short_length=short_length,
        long_length=long_length,
        n_to_smooth=n_to_smooth,
      ),
    )
    expected_col = f"macd_{short_length}_{long_length}_{n_to_smooth}"
    assert result.df_batch.columns == [expected_col]
    assert result.df_batch.index.equals(df.index)

    q = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert q.n_good_values >= 375
    assert q.n_bad_values_at_end == 0
    assert q.n_bad_values_at_start == q.n_bad_values
    print("q.n_bad_values_at_start", q.n_bad_values_at_start)
    assert result.dfs_match, (
      "Batch and streaming MACD results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
