import datetime
from unittest.mock import Mock, call

import pandas as pd
import pytest
from dateutil.tz import tzoffset

import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.zerodha as zerodha


class TestInstrumentDescIsValid:
  def test_valid_indian_spot_desc(self):
    spot_desc = sid.SpotDesc(symbol="RELIANCE", market=sid.Market.INDIAN_MARKET)
    assert zerodha.instrument_desc_is_valid(spot_desc) is True

  def test_valid_indian_index_desc(self):
    index_desc = sid.IndexDesc(symbol="NIFTY 50", market=sid.Market.INDIAN_MARKET)
    assert zerodha.instrument_desc_is_valid(index_desc) is True

  def test_invalid_non_indian_market(self):
    spot_desc = sid.SpotDesc(symbol="BTC", market=sid.Market.CRYPTO)
    assert zerodha.instrument_desc_is_valid(spot_desc) is False

  def test_invalid_option_desc(self):
    future_desc = sid.FutureDesc(
      symbol="BANKNIFTY", market=sid.Market.INDIAN_MARKET, expiry=None
    )
    assert zerodha.instrument_desc_is_valid(future_desc) is False


class TestGetKiteDataForRange:
  def test_invalid_instrument_desc(self):
    invalid_desc = sid.SpotDesc(symbol="BTC", market=sid.Market.CRYPTO)
    date_range = sdr.DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    mock_kite = Mock()

    with pytest.raises(ValueError, match="Invalid instrument description"):
      zerodha.get_kite_data_for_range(invalid_desc, date_range, mock_kite, "minute")


class TestGetFullKiteDataForRange:
  def test_short_date_range(self):
    spot_desc = sid.SpotDesc(symbol="HDFCBANK", market=sid.Market.INDIAN_MARKET)
    date_range = sdr.DateRange(datetime.date(2025, 1, 1), datetime.date(2025, 1, 1))
    mock_kite = Mock()
    mock_kite.ltp.return_value = {"NSE:HDFCBANK": {"instrument_token": 123456}}
    rvs = [
      {
        "date": datetime.datetime(2025, 1, 1, 9, 15, tzinfo=tzoffset(None, 19800)),
        "open": 1773.45,
        "high": 1781.9,
        "low": 1771.05,
        "close": 1777.15,
        "volume": 43320,
      },
      {
        "date": datetime.datetime(2025, 1, 1, 9, 16, tzinfo=tzoffset(None, 19800)),
        "open": 1777.15,
        "high": 1777.35,
        "low": 1771.7,
        "close": 1775,
        "volume": 27943,
      },
      {
        "date": datetime.datetime(2025, 1, 1, 9, 17, tzinfo=tzoffset(None, 19800)),
        "open": 1775,
        "high": 1776.5,
        "low": 1774.05,
        "close": 1774.05,
        "volume": 9919,
      },
      {
        "date": datetime.datetime(2025, 1, 1, 9, 18, tzinfo=tzoffset(None, 19800)),
        "open": 1774,
        "high": 1774,
        "low": 1772.2,
        "close": 1773.75,
        "volume": 7644,
      },
    ]
    mock_kite.historical_data.return_value = rvs

    result = zerodha.get_full_kite_data_for_range(spot_desc, date_range, mock_kite)

    mock_kite.ltp.assert_called_once_with("NSE:HDFCBANK")
    mock_kite.historical_data.assert_called_once_with(
      instrument_token=123456,
      from_date=datetime.date(2025, 1, 1),
      to_date=datetime.date(2025, 1, 1),
      interval="minute",
    )
    pd.testing.assert_frame_equal(
      result,
      pd.DataFrame(rvs),
    )

  def test_long_date_range_partial(self):
    spot_desc = sid.SpotDesc(symbol="HDFCBANK", market=sid.Market.INDIAN_MARKET)
    date_range = sdr.DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 5, 5))
    mock_kite = Mock()
    mock_kite.ltp.return_value = {"NSE:HDFCBANK": {"instrument_token": 123456}}
    ses = [
      [
        {
          "date": pd.Timestamp("2023-01-03 09:15:00+0530"),
          "open": 100,
          "high": 110,
          "low": 90,
          "close": 105,
          "volume": 1000,
        }
      ],
      [
        {
          "date": pd.Timestamp("2023-03-03 09:15:00+0530"),
          "open": 200,
          "high": 210,
          "low": 190,
          "close": 205,
          "volume": 2000,
        }
      ],
      [
        {
          "date": pd.Timestamp("2023-05-03 09:15:00+0530"),
          "open": 300,
          "high": 310,
          "low": 290,
          "close": 305,
          "volume": 3000,
        }
      ],
    ]
    mock_kite.historical_data.side_effect = ses
    result = zerodha.get_full_kite_data_for_range(spot_desc, date_range, mock_kite)
    pd.testing.assert_frame_equal(
      result, pd.DataFrame([y for x in ses for y in x]).reset_index(drop=True)
    )
    assert mock_kite.historical_data.call_args_list == [
      call(
        instrument_token=123456,
        from_date=datetime.date(2023, 1, 1),
        to_date=datetime.date(2023, 3, 1),
        interval="minute",
      ),
      call(
        instrument_token=123456,
        from_date=datetime.date(2023, 3, 2),
        to_date=datetime.date(2023, 4, 30),
        interval="minute",
      ),
      call(
        instrument_token=123456,
        from_date=datetime.date(2023, 5, 1),
        to_date=datetime.date(2023, 5, 5),
        interval="minute",
      ),
    ]

  def test_long_date_range_no_partial(self):
    spot_desc = sid.SpotDesc(symbol="HDFCBANK", market=sid.Market.INDIAN_MARKET)
    date_range = sdr.DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 4, 30))
    mock_kite = Mock()
    mock_kite.ltp.return_value = {"NSE:HDFCBANK": {"instrument_token": 123456}}
    mock_kite.historical_data.return_value = [
      {
        "date": pd.Timestamp("2023-01-03 09:15:00+0530"),
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 1000,
      }
    ]

    _ = zerodha.get_full_kite_data_for_range(spot_desc, date_range, mock_kite)
    assert mock_kite.historical_data.call_args_list == [
      call(
        instrument_token=123456,
        from_date=datetime.date(2023, 1, 1),
        to_date=datetime.date(2023, 3, 1),
        interval="minute",
      ),
      call(
        instrument_token=123456,
        from_date=datetime.date(2023, 3, 2),
        to_date=datetime.date(2023, 4, 30),
        interval="minute",
      ),
    ]

  def test_invalid_instrument_desc(self):
    invalid_desc = sid.SpotDesc(symbol="BTC", market=sid.Market.CRYPTO)
    date_range = sdr.DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    mock_kite = Mock()

    with pytest.raises(ValueError, match="Invalid instrument description"):
      zerodha.get_full_kite_data_for_range(invalid_desc, date_range, mock_kite)
