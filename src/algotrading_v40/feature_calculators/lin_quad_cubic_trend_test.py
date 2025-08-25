import datetime as dt

import pytest

import algotrading_v40.feature_calculators.lin_quad_cubic_trend as fc_lin_quad_cubic_trend
import algotrading_v40.utils.df as udf
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut


class TestLinQuadCubicTrendStreamingVsBatch:
  @pytest.mark.parametrize("poly_degree", [1, 2])
  def test_streaming_matches_batch(self, poly_degree: int) -> None:
    lookback = 10
    atr_length = 252

    df = ut.get_test_df(
      start_date=dt.date(2023, 1, 2),
      end_date=dt.date(2023, 1, 3),
    )

    result = us.compare_batch_and_stream(
      df,
      lambda df_: fc_lin_quad_cubic_trend.lin_quad_cubic_trend(
        df_,
        poly_degree=poly_degree,
        lookback=lookback,
        atr_length=atr_length,
      ),
    )

    expected_col = f"lin_quad_cubic_trend_{poly_degree}_{lookback}_{atr_length}"
    assert result.df_batch.columns == [expected_col]
    assert result.df_batch.index.equals(df.index)

    q = udf.analyse_numeric_series_quality(result.df_batch[expected_col])
    assert q.n_good_values >= 375
    assert q.n_bad_values_at_end == 0
    assert q.n_bad_values_at_start == q.n_bad_values
    print("q.n_bad_values_at_start", q.n_bad_values_at_start)
    assert result.dfs_match, (
      "Batch and streaming lin_quad_cubic_trend results diverged.\n"
      f"Batch (up to mismatch):\n{result.df_batch_mismatch}\n\n"
      f"Streaming:\n{result.df_stream_mismatch}"
    )
