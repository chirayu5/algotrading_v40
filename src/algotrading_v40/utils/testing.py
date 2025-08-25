import copy
import datetime as dt
from contextlib import contextmanager
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid


def _default_equal(a: Any, b: Any) -> bool:
  try:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
      return np.array_equal(a, b)
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
      return a.equals(b)
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
      return a.equals(b)
    if isinstance(a, pd.Index) and isinstance(b, pd.Index):
      return a.equals(b) and a.name == b.name
  except ImportError:
    pass
  return a == b


@contextmanager
def expect_no_mutation(
  *objs: Any,
) -> Iterable[None]:
  snapshots: List[Any] = [copy.deepcopy(o) for o in objs]
  # we have captured a snapshot of the objects before the with block
  # the yield is the point where the original function is run
  yield
  # the original function is run; now we check if the objects have been mutated
  for i, (before, after) in enumerate(zip(snapshots, objs)):
    if not _default_equal(before, after):
      raise AssertionError(
        f"Object number {i} was mutated: before={before!r}, after={after!r}"
      )


def get_test_df(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
  """
  Get a test DataFrame with a dummy volume column set to 1.
  """
  inst_desc = sid.InstrumentDesc(
    market=sid.Market.INDIAN_MARKET,
    symbol="ABCD",
  )
  data = das.get_synthetic_data(
    instrument_descs=[inst_desc],
    date_range=sdr.DateRange(
      start_date=start_date,
      end_date=end_date,
    ),
  )
  df = data.get_full_df_for_instrument_desc(inst_desc).copy()
  df["volume"] = 1  # dummy volume column
  return df
