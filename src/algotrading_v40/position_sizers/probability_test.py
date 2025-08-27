import numpy as np
import pandas as pd
import pytest

import algotrading_v40.position_sizers.probability as psp

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
TZ = "UTC"
START = pd.Timestamp("2021-01-01 03:45:59.999000", tz=TZ)
ONE_MIN = pd.Timedelta(minutes=1)


def ns(ts: pd.Timestamp) -> int:
  """Convert timestamp => int nanoseconds (same as .astype(int))."""
  return int(ts.value)


def make_df(index: list[pd.Timestamp]) -> pd.DataFrame:
  """Return a minimally-filled dataframe with constant, valid defaults."""
  n = len(index)
  return pd.DataFrame(
    {
      # these will be overwritten by individual tests where needed
      "prob": 0.55,
      "side": 1,
      "open": 100.0,
      "high": 100.5,
      "low": 99.5,
      "selected": 0,
      "tpb": 0.03,
      "slb": -0.03,
      "vb_timestamp_exec": [ns(index[-1] + ONE_MIN)] * n,
    },
    index=index,
  )


# a configuration that avoids extra rounding inside the sizing routine
QA_STEP = 1e-4
QA_MAX = 100.0
BA_STEP_NONE = None  # no BA rounding


# --------------------------------------------------------------------------- #
# 1. all long, different sizes => average long                                #
# --------------------------------------------------------------------------- #
def test_average_of_multiple_long_bets():
  idx = [START + i * ONE_MIN for i in range(3)]
  df = make_df(idx)

  # row-0 long, prob=0.60
  df.loc[idx[0], ["prob", "selected"]] = (0.60, 1)

  # row-1 long, prob=0.70
  # selected=0 => this bet is not opened
  df.loc[idx[1], ["prob", "selected"]] = (0.70, 0)

  df.loc[idx[2], ["prob", "selected"]] = (0.85, 1)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_average_of_multiple_long_bets:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_average_of_multiple_long_bets:")
  print(out.to_string())
  size_a = psp.get_size(0.60)
  size_c = psp.get_size(0.85)
  expected = (size_a + size_c) / 2

  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], expected)


# --------------------------------------------------------------------------- #
# 2. long + short => average cancels out                                      #
# --------------------------------------------------------------------------- #
def test_average_long_and_short_bets():
  idx = [START + i * ONE_MIN for i in range(3)]
  df = make_df(idx)

  # open long on row-0
  df.loc[idx[0], ["prob", "selected", "side"]] = (0.65, 1, 1)
  # open short on row-1
  df.loc[idx[1], ["prob", "selected", "side"]] = (0.80, 1, -1)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_average_long_and_short_bets:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_average_long_and_short_bets:")
  print(out.to_string())
  size_long = psp.get_size(0.65)
  size_short = psp.get_size(0.80)
  expected = (size_long * 1 + size_short * -1) / 2

  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], expected)


# --------------------------------------------------------------------------- #
# 3. new long bet opened                                                      #
# --------------------------------------------------------------------------- #
def test_open_new_long_bet():
  idx = [START, START + ONE_MIN]
  df = make_df(idx)

  # nothing on first row, open long on second
  df.loc[idx[1], ["prob", "selected"]] = (0.75, 1)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_open_new_long_bet:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_open_new_long_bet:")
  print(out.to_string())
  expected = psp.get_size(0.75)  # only one active bet
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], expected)


# --------------------------------------------------------------------------- #
# 4. new short bet opened                                                     #
# --------------------------------------------------------------------------- #
def test_open_new_short_bet():
  idx = [START, START + ONE_MIN]
  df = make_df(idx)

  # row-1 short
  df.loc[idx[1], ["prob", "selected", "side"]] = (0.80, 1, -1)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_open_new_short_bet:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_open_new_short_bet:")
  print(out.to_string())
  expected = psp.get_size(0.80) * -1
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], expected)


# --------------------------------------------------------------------------- #
# 5. long closed via stop-loss                                                #
# --------------------------------------------------------------------------- #
def test_long_bet_closed_at_stop_loss():
  # three rows in the stream
  idx = [START + i * ONE_MIN for i in range(3)]
  df = make_df(idx)

  # Bet #1: open long on row-0 – will remain open
  df.loc[idx[0], ["prob", "selected", "slb"]] = (0.60, 1, -0.2)
  # set a very low (-20%) stop-loss so that bet #1 remains open

  # Bet #2: open long on row-1 – will be closed via its stop-loss
  df.loc[idx[1], ["prob", "selected"]] = (0.80, 1)
  # drive price sharply lower in row-1 so that stop-loss is detected on row-2
  df.loc[idx[1], "low"] = 95.0  # ~5 % drop ⇒ return ≈ -5 % < slb (-3 %)

  # simulate streaming – for the last (incomplete) bar we don't yet know high/low
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_long_bet_closed_at_stop_loss:")
  print(df.to_string())

  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_long_bet_closed_at_stop_loss:")
  print(out.to_string())
  expected = psp.get_size(0.60)  # only Bet #1 is still active
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], expected)


# --------------------------------------------------------------------------- #
# 6. short closed via take-profit                                             #
# --------------------------------------------------------------------------- #
def test_short_bet_closed_at_take_profit():
  idx = [START, START + ONE_MIN]
  df = make_df(idx)

  # open short on row-0
  df.loc[idx[0], ["prob", "selected", "side"]] = (0.75, 1, -1)

  # drive price down in row-0 to hit TP in next step
  df.loc[idx[0], "low"] = 95.0  # ~5% drop gives positive return for short

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_short_bet_closed_at_take_profit:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_short_bet_closed_at_take_profit:")
  print(out.to_string())
  assert np.isclose(out.loc[idx[0], "raw_qa_position"], -psp.get_size(0.75))
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], 0.0)


# --------------------------------------------------------------------------- #
# 7. bets closed by vertical barrier                                          #
# --------------------------------------------------------------------------- #
def test_bets_closed_by_vertical_barrier():
  idx = [START + i * ONE_MIN for i in range(3)]
  df = make_df(idx)

  vb_hit_time = ns(idx[2])  # barrier on the last row

  # two bets with the SAME vertical barrier
  df.loc[idx[0], ["prob", "selected", "vb_timestamp_exec"]] = (0.65, 1, vb_hit_time)
  df.loc[idx[1], ["prob", "selected", "vb_timestamp_exec"]] = (0.70, 1, vb_hit_time)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_bets_closed_by_vertical_barrier:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_bets_closed_by_vertical_barrier:")
  print(out.to_string())
  # one bet open
  assert np.isclose(out.loc[idx[0], "raw_qa_position"], psp.get_size(0.65))
  # two bets open
  assert np.isclose(
    out.loc[idx[1], "raw_qa_position"], (psp.get_size(0.7) + psp.get_size(0.65)) / 2
  )
  # vertical barrier hit...both bets closed
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], 0.0)


# --------------------------------------------------------------------------- #
# 8. discretised position is within [-qa_max, qa_max]                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.55, 0.80, 1])
def test_discretised_position_bounds(p: float):
  idx = [START, START + ONE_MIN]
  df = make_df(idx)
  df.loc[idx[1], ["prob", "selected"]] = (p, 1)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  print()
  print("--------------------------------")
  print("data for test_discretised_position_bounds:")
  print(df.to_string())
  out = psp.probability_position_sizer(
    df=df,
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  print("output for test_discretised_position_bounds:")
  print(out.to_string())
  dpos = out.loc[idx[-1], "discretised_qa_position"]
  assert -QA_MAX <= dpos <= QA_MAX
