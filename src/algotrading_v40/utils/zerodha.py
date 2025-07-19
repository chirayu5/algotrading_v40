import datetime

import kiteconnect
import pandas as pd

import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid


def instrument_desc_is_valid(instrument_desc: sid.InstrumentDesc) -> bool:
  """
  Check if the instrument description is valid.

  Zerodha Kite is only for Indian market.
  Only spot and index instruments are supported for now.
  Options are not supported.
  """
  if not (
    isinstance(instrument_desc, sid.SpotDesc)
    or isinstance(instrument_desc, sid.IndexDesc)
  ):
    return False
  if instrument_desc.market != sid.Market.INDIAN_MARKET:
    return False
  return True


# All instruments on Zerodha available here:
# https://api.kite.trade/instruments
def get_kite_data_for_range(
  instrument_desc: sid.InstrumentDesc,
  date_range: sdr.DateRange,
  kite: kiteconnect.KiteConnect,
  interval: str,
) -> pd.DataFrame:
  if not instrument_desc_is_valid(instrument_desc):
    raise ValueError(f"Invalid instrument description: {instrument_desc}")
  nse_symbol = f"NSE:{instrument_desc.symbol}"
  instrument_token = kite.ltp(nse_symbol)[nse_symbol]["instrument_token"]  # type: ignore
  return pd.DataFrame(
    # the time stamps here are candle opening times
    # reference: https://kite.trade/forum/discussion/14764/timestamps-on-1-minute-candles-are-wrong
    kite.historical_data(
      instrument_token=instrument_token,
      from_date=date_range.start_date,
      to_date=date_range.end_date,
      interval=interval,
    )
  )


def get_full_kite_data_for_range(
  instrument_desc: sid.InstrumentDesc,
  date_range: sdr.DateRange,
  kite: kiteconnect.KiteConnect,
) -> pd.DataFrame:
  if not instrument_desc_is_valid(instrument_desc):
    raise ValueError(f"Invalid instrument description: {instrument_desc}")
  curr_date_range = sdr.DateRange(
    date_range.start_date,
    date_range.start_date + datetime.timedelta(days=60 - 1),
  )
  all_dfs = []
  while curr_date_range.start_date <= date_range.end_date:
    if curr_date_range.end_date > date_range.end_date:
      curr_date_range = sdr.DateRange(
        curr_date_range.start_date,
        date_range.end_date,
      )
    df_symbol = get_kite_data_for_range(
      instrument_desc, curr_date_range, kite, "minute"
    )
    all_dfs.append(df_symbol)
    curr_date_range = curr_date_range.slide(days=60)
  df = pd.concat(all_dfs).reset_index(drop=True)
  return df
