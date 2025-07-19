import pytest

import algotrading_v40.structures.instrument_desc as sid


class TestEquityDesc:
  def test_valid_indian_market(self):
    equity_desc = sid.EquityDesc(symbol="RELIANCE", market=sid.Market.INDIAN_MARKET)
    assert equity_desc.symbol == "RELIANCE"
    assert equity_desc.market == sid.Market.INDIAN_MARKET

  def test_valid_us_market(self):
    equity_desc = sid.EquityDesc(symbol="AAPL", market=sid.Market.US_MARKET)
    assert equity_desc.symbol == "AAPL"
    assert equity_desc.market == sid.Market.US_MARKET

  def test_crypto_market_raises_error(self):
    with pytest.raises(
      ValueError,
      match="EquityDesc can not be used for crypto market, got Market.CRYPTO",
    ):
      sid.EquityDesc(symbol="BTC", market=sid.Market.CRYPTO)


class TestIndexDesc:
  def test_valid_indian_market(self):
    index_desc = sid.IndexDesc(symbol="NIFTY 50", market=sid.Market.INDIAN_MARKET)
    assert index_desc.symbol == "NIFTY 50"
    assert index_desc.market == sid.Market.INDIAN_MARKET

  def test_valid_us_market(self):
    index_desc = sid.IndexDesc(symbol="S&P 500", market=sid.Market.US_MARKET)
    assert index_desc.symbol == "S&P 500"
    assert index_desc.market == sid.Market.US_MARKET

  def test_crypto_market_raises_error(self):
    with pytest.raises(
      ValueError, match="IndexDesc can not be used for crypto market, got Market.CRYPTO"
    ):
      sid.IndexDesc(symbol="CRYPTO_INDEX", market=sid.Market.CRYPTO)
