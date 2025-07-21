import os

import algotrading_v40.structures.instrument_desc as sid


def get_data_dir() -> str:
  current_file_path = __file__
  return os.path.join(
    os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    ),
    "data",
  )


def _get_path_from_instrument_desc(
  instrument_desc: sid.InstrumentDesc, data_type: str
) -> str:
  data_dir = get_data_dir()
  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"data directory is not at {data_dir}")
  prefix = os.path.join(
    data_dir,
    data_type,
    instrument_desc.market.value,
  )
  ans = None
  if isinstance(instrument_desc, sid.EquityDesc):
    ans = os.path.join(prefix, "equity", f"{instrument_desc.symbol}.parquet")
  elif isinstance(instrument_desc, sid.IndexDesc):
    ans = os.path.join(prefix, "index", f"{instrument_desc.symbol}.parquet")
  else:
    raise ValueError(f"Unsupported instrument description: {instrument_desc}")
  return ans


def get_raw_path_from_instrument_desc(instrument_desc: sid.InstrumentDesc) -> str:
  return _get_path_from_instrument_desc(instrument_desc, "raw")


def get_cleaned_path_from_instrument_desc(instrument_desc: sid.InstrumentDesc) -> str:
  return _get_path_from_instrument_desc(instrument_desc, "cleaned")


def get_instrument_desc_from_path(path: str) -> sid.InstrumentDesc:
  if not path.endswith(".parquet"):
    raise ValueError(f"Path does not end with .parquet: {path}")
  ps = path.split("/")
  if ps[-5] != "data":
    raise ValueError(f"Path does not contain data: {path}")
  if ps[-4] not in ("raw", "cleaned"):
    raise ValueError(f"Path does not contain raw or cleaned: {path}")
  market = None
  if ps[-3] == sid.Market.INDIAN_MARKET.value:
    market = sid.Market.INDIAN_MARKET
  elif ps[-3] == sid.Market.US_MARKET.value:
    market = sid.Market.US_MARKET
  else:
    raise ValueError(f"Market not found in path: {path}")
  if "equity" == ps[-2]:
    ans = sid.EquityDesc(symbol=ps[-1].split(".")[0], market=market)
  elif "index" == ps[-2]:
    ans = sid.IndexDesc(symbol=ps[-1].split(".")[0], market=market)
  else:
    raise ValueError(f"Path does not contain equity or index: {path}")
  return ans
