import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.data_nav as udn


class TestRoundTrip:
  """Test that converting instrument desc to path and back yields the same result."""

  def _get_test_cases(self):
    """Helper method to get test cases."""
    return [
      (
        sid.EquityDesc(market=sid.Market.INDIAN_MARKET, symbol="APOLLOHOSP"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/indian_market/equity/APOLLOHOSP.parquet",
      ),
      (
        sid.IndexDesc(market=sid.Market.INDIAN_MARKET, symbol="NIFTY FIN SERVICE"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/indian_market/index/NIFTY FIN SERVICE.parquet",
      ),
      (
        sid.EquityDesc(market=sid.Market.US_MARKET, symbol="AAPL"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/us_market/equity/AAPL.parquet",
      ),
      (
        sid.IndexDesc(market=sid.Market.US_MARKET, symbol="SPY"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/us_market/index/SPY.parquet",
      ),
      (
        sid.EquityDesc(market=sid.Market.US_MARKET, symbol="BRK-A"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/us_market/equity/BRK-A.parquet",
      ),
      (
        sid.IndexDesc(market=sid.Market.INDIAN_MARKET, symbol="NIFTY BANK"),
        "/Users/chirayuagrawal/algotrading_v40/data/raw_or_cleaned/indian_market/index/NIFTY BANK.parquet",
      ),
    ]

  def test_raw_path_round_trip(self):
    """Test round trip conversion for raw paths."""
    test_cases = self._get_test_cases()
    for instrument_desc, expected_path in test_cases:
      raw_path = udn.get_raw_path_from_instrument_desc(instrument_desc)
      assert raw_path == expected_path.replace("/raw_or_cleaned/", "/raw/")
      parsed_desc = udn.get_instrument_desc_from_path(raw_path)
      assert parsed_desc == instrument_desc

  def test_cleaned_path_round_trip(self):
    """Test round trip conversion for cleaned paths."""
    test_cases = self._get_test_cases()
    for instrument_desc, expected_path in test_cases:
      cleaned_path = udn.get_cleaned_path_from_instrument_desc(instrument_desc)
      assert cleaned_path == expected_path.replace("/raw_or_cleaned/", "/cleaned/")
      parsed_desc = udn.get_instrument_desc_from_path(cleaned_path)
      assert parsed_desc == instrument_desc
