import datetime as dt

import algotrading_v40.feature_calculators.price_intensity as fc_price_intensity
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


class TestPriceIntensityStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    """
    The price intensity calculated on the full price series should be identical
    to the values obtained when the same data arrive one point at a time.
    """
    n_to_smooth = 20

    df = ut.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )

    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_price_intensity.price_intensity(
        df_,
        n_to_smooth=n_to_smooth,
      ),
    )

    expected_col = f"price_intensity_{n_to_smooth}"
    assert result.df_batch.columns == [expected_col]

    # result should have the same index as the input data
    assert result.df_batch.index.equals(df.index)

    result_quality = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    # at least one full session's data should be good and compared with streaming
    assert result_quality.n_good_values >= 375
    # there should be no bad values at the end
    assert result_quality.n_bad_values_at_end == 0
    # all bad values should be at the start
    assert result_quality.n_bad_values_at_start == result_quality.n_bad_values
    print("result_quality.n_bad_values_at_start", result_quality.n_bad_values_at_start)
    # batch and streaming results should match
    assert result.dfs_match, (
      "Batch and streaming price intensity results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
