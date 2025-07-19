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


def get_raw_path_from_instrument_desc(instrument_desc: sid.InstrumentDesc) -> str:
  data_dir = get_data_dir()
  if not os.path.exists(data_dir):
    raise FileNotFoundError(f"data directory is not at {data_dir}")
  prefix = os.path.join(
    data_dir,
    "raw",
    instrument_desc.market.value,
  )
  if isinstance(instrument_desc, sid.EquityDesc):
    return os.path.join(prefix, "equity", f"{instrument_desc.symbol}.parquet")
  elif isinstance(instrument_desc, sid.IndexDesc):
    return os.path.join(prefix, "index", f"{instrument_desc.symbol}.parquet")
  raise ValueError(f"Unsupported instrument description: {instrument_desc}")
