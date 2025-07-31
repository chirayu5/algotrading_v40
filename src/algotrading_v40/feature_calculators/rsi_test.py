import datetime as dt

import numpy as np
import pandas as pd

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.feature_calculators.rsi as fc_rsi
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid


class TestRsiStreamingVsBatch:
  def test_streaming_matches_batch(self) -> None:
    """
    The RSI calculated on the full price series should be identical to the
    values obtained when the same data arrive one point at a time.
    """
    np.random.seed(42)
    lookback = 14
    id = sid.InstrumentDesc(
      market=sid.Market.INDIAN_MARKET,
      symbol="ABCD",
    )
    data = das.get_synthetic_data(
      instrument_descs=[id],
      date_range=sdr.DateRange(
        start_date=dt.date(2023, 1, 1),
        end_date=dt.date(2023, 1, 20),
      ),
    )

    prices = data.get_full_df_for_instrument_desc(id)["close"]
    n_points = len(prices)
    batch_rsi = fc_rsi.rsi(prices, lookback)
    for k in range(n_points + 1):
      partial_prices = prices.iloc[:k]
      streaming_rsi = fc_rsi.rsi(partial_prices, lookback)
      expected = batch_rsi.iloc[:k]
      pd.testing.assert_series_equal(streaming_rsi, expected)
