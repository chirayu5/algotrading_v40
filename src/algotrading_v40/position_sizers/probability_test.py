import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import algotrading_v40.position_sizers.probability as psp
import algotrading_v40.utils.streaming as us
import algotrading_v40.utils.testing as ut

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
  df = pd.DataFrame(
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
      "vb_timestamp_exec_int": [ns(index[-1] + ONE_MIN)] * n,
    },
    index=index,
  )
  df["vb_timestamp_exec_int"] = df["vb_timestamp_exec_int"].astype(int)
  return df


def get_size(prob: float) -> float:
  # clip the probability to avoid division by 0
  prob = np.clip(prob, 1e-10, 1 - 1e-10)
  v = (prob - 0.5) / np.sqrt((prob * (1 - prob)))
  return 2 * norm.cdf(v) - 1


def discretise_position(
  position: float,
  step_size: float,
) -> float:
  return np.sign(position) * np.round(np.abs(position) / step_size) * step_size


# a configuration that avoids extra rounding inside the sizing routine
QA_STEP = 1e-1
QA_MAX = 1e5
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
  # print()
  # print("--------------------------------")
  # print("data for test_average_of_multiple_long_bets:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_average_of_multiple_long_bets:")
  # print(out.to_string())
  size_a = get_size(0.60)
  size_c = get_size(0.85)
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
  # print()
  # print("--------------------------------")
  # print("data for test_average_long_and_short_bets:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_average_long_and_short_bets:")
  # print(out.to_string())
  size_long = get_size(0.65)
  size_short = get_size(0.80)
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
  # print()
  # print("--------------------------------")
  # print("data for test_open_new_long_bet:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_open_new_long_bet:")
  # print(out.to_string())
  expected = get_size(0.75)  # only one active bet
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
  # print()
  # print("--------------------------------")
  # print("data for test_open_new_short_bet:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_open_new_short_bet:")
  # print(out.to_string())
  expected = get_size(0.80) * -1
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
  # print()
  # print("--------------------------------")
  # print("data for test_long_bet_closed_at_stop_loss:")
  # print(df.to_string())

  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_long_bet_closed_at_stop_loss:")
  # print(out.to_string())
  expected = get_size(0.60)  # only Bet #1 is still active
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
  # print()
  # print("--------------------------------")
  # print("data for test_short_bet_closed_at_take_profit:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_short_bet_closed_at_take_profit:")
  # print(out.to_string())
  assert np.isclose(out.loc[idx[0], "raw_qa_position"], -get_size(0.75))
  assert np.isclose(out.loc[idx[-1], "raw_qa_position"], 0.0)


# --------------------------------------------------------------------------- #
# 7. bets closed by vertical barrier                                          #
# --------------------------------------------------------------------------- #
def test_bets_closed_by_vertical_barrier():
  idx = [START + i * ONE_MIN for i in range(3)]
  df = make_df(idx)

  vb_hit_time = ns(idx[2])  # barrier on the last row

  # two bets with the SAME vertical barrier
  df.loc[idx[0], ["prob", "selected", "vb_timestamp_exec_int"]] = (0.65, 1, vb_hit_time)
  df.loc[idx[1], ["prob", "selected", "vb_timestamp_exec_int"]] = (0.70, 1, vb_hit_time)

  # set the last row's OHLC values to NaN to simulate a stream
  df.loc[idx[-1], ["high", "low"]] = np.nan
  # print()
  # print("--------------------------------")
  # print("data for test_bets_closed_by_vertical_barrier:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_bets_closed_by_vertical_barrier:")
  # print(out.to_string())
  # one bet open
  assert np.isclose(out.loc[idx[0], "raw_qa_position"], get_size(0.65))
  # two bets open
  assert np.isclose(
    out.loc[idx[1], "raw_qa_position"], (get_size(0.7) + get_size(0.65)) / 2
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
  # print()
  # print("--------------------------------")
  # print("data for test_discretised_position_bounds:")
  # print(df.to_string())
  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_discretised_position_bounds:")
  # print(out.to_string())
  dpos = out.loc[idx[-1], "discretised_qa_position"]
  assert -QA_MAX <= dpos <= QA_MAX


# --------------------------------------------------------------------------- #
# 9. unchanged QA position ⇒ unchanged BA position                            #
# --------------------------------------------------------------------------- #
def test_final_ba_position_unchanged_when_qa_position_same():
  idx = [START, START + ONE_MIN]
  df = make_df(idx)

  # open one long bet on the first row
  df.loc[idx[0], ["prob", "selected"]] = (0.70, 1)

  # keep QA position the same on row-1 (selected = 0 → no new bet)
  # but change the open price to a different value to check that BA
  # position stays constant even though price moved.
  # choose a price change < +3 % so the original bet is NOT closed by TP
  df.loc[idx[1], ["open", "selected"]] = (101.0, 0)

  # simulate streaming – last bar’s high/low are still unknown
  df.loc[idx[-1], ["high", "low"]] = np.nan
  # print()
  # print("--------------------------------")
  # print("data for test_final_ba_position_unchanged_when_qa_position_same:")
  # print(df.to_string())

  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  # print("output for test_final_ba_position_unchanged_when_qa_position_same:")
  # print(out.to_string())

  # BA position on the second row should equal BA position on the first row
  assert np.isclose(
    out.loc[idx[1], "final_ba_position"],
    out.loc[idx[0], "final_ba_position"],
  )
  # Additionally, ensure it is *not* equal to what it would be if
  # recomputed using the new price (would trigger an unnecessary trade)
  recomputed = out.loc[idx[0], "discretised_qa_position"] / df.loc[idx[1], "open"]
  assert not np.isclose(out.loc[idx[1], "final_ba_position"], recomputed)


# --------------------------------------------------------------------------- #
# 10. BA position is rounded down to the nearest step                         #
# --------------------------------------------------------------------------- #
def test_final_ba_position_is_rounded_down_to_step():
  idx = [START]  # single bar is enough
  df = make_df(idx)

  # open a long bet with probability chosen to give a non-integer BA position
  prob_val = 0.75
  open_price = 123.45
  df.loc[idx[0], ["prob", "selected", "open", "high", "low"]] = (
    prob_val,
    1,
    open_price,
    open_price,
    open_price,
  )

  # BA step size for rounding
  BA_STEP = 15  # choose a convenient step
  # print()
  # print("--------------------------------")
  # print("data for test_final_ba_position_is_rounded_down_to_step:")
  # print(df.to_string())

  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP,
    qa_max=QA_MAX,
  )

  # print("output for test_final_ba_position_is_rounded_down_to_step:")
  # print(out.to_string())

  # --- expected value -------------------------------------------------------
  size = get_size(prob_val)  # raw QA size (between 0-1)
  rdqa = discretise_position(size, QA_STEP)  # after QA discretisation
  discretised_qa_position = rdqa * QA_MAX  # quote-asset position
  unrounded_ba = discretised_qa_position / open_price  # before BA rounding
  expected_ba = (
    np.sign(unrounded_ba) * np.floor(np.abs(unrounded_ba) / BA_STEP) * BA_STEP
  )
  # --------------------------------------------------------------------------
  # without the flooring logic, this value would be 324.017821
  # with rounding instead of flooring, this value would be 330.0
  # correct value should be 315.0
  assert np.isclose(expected_ba, 315.0)
  assert np.isclose(out.loc[idx[0], "final_ba_position"], expected_ba)


# --------------------------------------------------------------------------- #
# 11. new bet is opened and immediately closed by its vertical barrier        #
# --------------------------------------------------------------------------- #
def test_new_bet_immediately_closed_by_vertical_barrier():
  idx = [START]
  df = make_df(idx)

  # configure the bar so that:
  # • a new long bet is opened (selected = 1)
  # • the vertical barrier is the bar's own close-timestamp ⇒ immediate close
  prob_val = 0.80
  df.loc[idx[0], ["prob", "selected", "vb_timestamp_exec_int"]] = (
    prob_val,
    1,
    ns(idx[0]),  # barrier hit NOW
  )

  # print()
  # print("--------------------------------")
  # print("data for test_new_bet_immediately_closed_by_vertical_barrier:")
  # print(df.to_string())

  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )

  # print("output for test_new_bet_immediately_closed_by_vertical_barrier:")
  # print(out.to_string())

  # bet must be closed in the same bar ⇒ all positions zero
  assert np.isclose(out.loc[idx[0], "raw_qa_position"], 0.0)
  assert np.isclose(out.loc[idx[0], "discretised_qa_position"], 0.0)
  assert np.isclose(out.loc[idx[0], "final_ba_position"], 0.0)


# --------------------------------------------------------------------------- #
# 11. stream vs batch                                                         #
# --------------------------------------------------------------------------- #
def test_stream_vs_batch():
  np.random.seed(42)
  df = ut.get_test_df(
    start_date=dt.date(2023, 1, 2),
    end_date=dt.date(2023, 1, 4),
  )
  n = len(df)
  prob = np.random.uniform(0.5, 1, n)
  side = np.random.choice([-1, 1], n)
  selected = np.random.choice([0, 1], n)
  tpb = np.random.uniform(0.01, 0.1, n)
  slb = np.random.uniform(-0.1, -0.01, n)
  vb_timestamp_exec_int = (
    df.index + pd.to_timedelta(np.random.randint(1, 81, n), unit="min")
  ).astype(int)

  def inner_(df_: pd.DataFrame) -> pd.DataFrame:
    n_ = len(df_)
    df_["prob"] = prob[:n_]
    df_["side"] = side[:n_]
    df_["selected"] = selected[:n_]
    df_["tpb"] = tpb[:n_]
    df_["slb"] = slb[:n_]
    df_["vb_timestamp_exec_int"] = vb_timestamp_exec_int[:n_]
    return psp.probability_position_sizer(
      prob=df_["prob"],
      side=df_["side"],
      open=df_["open"],
      high=df_["high"],
      low=df_["low"],
      selected=df_["selected"],
      tpb=df_["tpb"],
      slb=df_["slb"],
      vb_timestamp_exec_int=df_["vb_timestamp_exec_int"],
      qa_step_size=QA_STEP,
      ba_step_size=1,
      qa_max=QA_MAX,
    )

  result = us.compare_batch_and_stream(
    df,
    inner_,
  )
  assert result.dfs_match


# --------------------------------------------------------------------------- #
# 12. position allowed
# --------------------------------------------------------------------------- #
def test_position_allowed():
  df = pd.DataFrame(
    {
      "open": [268.70, 269.05, 269.30, 269.28, 268.70],
      "high": [269.07, 269.25, 269.34, 269.40, 268.77],
      "low": [268.62, 269.02, 269.15, 268.71, 268.62],
      "close": [269.07, 269.25, 269.30, 268.71, 268.68],
      "volume": [1, 1, 1, 1, 1],
      "prob": [0.533824, 0.736792, 0.544903, 0.848821, 0.926738],
      "side": [-1, -1, 1, 1, 1],
      "selected": [1, 1, 0, 1, 1],
      "tpb": [0.093419, 0.049842, 0.090408, 0.071795, 0.081796],
      "slb": [-0.013168, -0.067938, -0.036831, -0.046839, -0.055281],
      "vb_timestamp_exec_int": [
        1672633259999000000,
        1672634159999000000,
        1672631699999000000,
        1672634339999000000,
        1672634819999000000,
      ],
      "position_allowed": [0, 1, 0, 1, 0],
    },
    index=pd.to_datetime(
      [
        "2023-01-02 03:45:59.999000+00:00",
        "2023-01-02 03:46:59.999000+00:00",
        "2023-01-02 03:47:59.999000+00:00",
        "2023-01-02 03:48:59.999000+00:00",
        "2023-01-02 03:49:59.999000+00:00",
      ]
    ),
  )
  df.index.name = "bar_close_timestamp"

  # with pd.option_context("display.max_rows", None, "display.max_columns", None):
  #   print(df.to_string())

  out_pa = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
    position_allowed=df["position_allowed"],
  )

  out = psp.probability_position_sizer(
    prob=df["prob"],
    side=df["side"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    selected=df["selected"],
    tpb=df["tpb"],
    slb=df["slb"],
    vb_timestamp_exec_int=df["vb_timestamp_exec_int"],
    qa_step_size=QA_STEP,
    ba_step_size=BA_STEP_NONE,
    qa_max=QA_MAX,
  )
  assert not out_pa.equals(out)
  assert out_pa.equals(out.mul(df["position_allowed"], axis=0))
