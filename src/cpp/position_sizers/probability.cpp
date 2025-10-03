
#include "../utils/io.cpp"
#include <algorithm>
#include <cmath>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace {

inline double get_size(double prob) {
  prob = std::clamp(prob, 1e-10, 1. - 1e-10);
  const double v = (prob - 0.5) / std::sqrt(prob * (1. - prob));
  return 2. * 0.5 * (1. + std::erf(v / std::sqrt(2.))) - 1.; // 2*N(v)-1
}

struct Bet {
  double size{};
  int side{};
  int64_t entry_ts{};
  double entry_price{};
  double tpb{}, slb{};
  int64_t vb_ts{};

  bool should_close(int64_t now_ts, double highest_price,
                    double lowest_price) const {
    if (now_ts >= vb_ts)
      return true;

    const auto get_ret = [&](double p1) {
      return side * ((p1 / entry_price) - 1.);
    };
    const double ret_h = get_ret(highest_price);
    const double ret_l = get_ret(lowest_price);
    return std::max(ret_h, ret_l) >= tpb || std::min(ret_h, ret_l) <= slb;
  }
};

inline double average_bets(const std::vector<Bet> &bets) {
  if (bets.empty())
    return 0.;
  double acc = 0.;
  for (auto &b : bets)
    acc += b.size * b.side;
  return acc / static_cast<double>(bets.size());
}

inline double discretise(double pos, double step) {
  return std::copysign(std::round(std::abs(pos) / step), pos) * step;
}

pybind11::dict probability_position_sizer_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &prob,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &side,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &open,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &selected,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &tpb,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &slb,
    const pybind11::array_t<int64_t, pybind11::array::c_style |
                                         pybind11::array::forcecast>
        &close_ts_int,
    const pybind11::array_t<int64_t, pybind11::array::c_style |
                                         pybind11::array::forcecast>
        &vb_ts_exec_int,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast>
        &position_allowed,
    /* --- scalars --- */
    double qa_step_size, std::optional<double> ba_step_size, double qa_max) {

  /* ----------  basic shape / argument validation ---------- */
  const auto [prob_, n1] = atv40::io::get_input_ptr<double>(prob, "prob");
  const auto [side_, n2] = atv40::io::get_input_ptr<int32_t>(side, "side");
  const auto [open_, n3] = atv40::io::get_input_ptr<double>(open, "open");
  const auto [high_, n4] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n5] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [selected_, n6] =
      atv40::io::get_input_ptr<int32_t>(selected, "selected");
  const auto [tpb_, n7] = atv40::io::get_input_ptr<double>(tpb, "tpb");
  const auto [slb_, n8] = atv40::io::get_input_ptr<double>(slb, "slb");
  const auto [close_ts_int_, n9] =
      atv40::io::get_input_ptr<int64_t>(close_ts_int, "close_ts_int");
  const auto [vb_ts_exec_, n10] =
      atv40::io::get_input_ptr<int64_t>(vb_ts_exec_int, "vb_ts_exec_int");
  const auto [position_allowed_, n11] =
      atv40::io::get_input_ptr<int32_t>(position_allowed, "position_allowed");

  for (auto nii : {n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11}) {
    if (nii != n1)
      throw std::runtime_error("Array length mismatch.");
  }

  if (qa_step_size <= 0.0 || qa_step_size > 1.0) {
    throw std::runtime_error("qa_step_size out of range");
  }
  if (qa_max <= 0.0) {
    throw std::runtime_error("qa_max must be positive");
  }

  /* ----------  output buffers ---------- */
  auto [raw_out_arr, raw_out] = atv40::io::make_output_array<double>(n1);
  auto [disc_out_arr, disc_out] = atv40::io::make_output_array<double>(n1);
  auto [ba_out_arr, ba_out] = atv40::io::make_output_array<double>(n1);

  /* ----------  main loop ---------- */
  std::vector<Bet> active;
  active.reserve(16); // heuristic

  for (std::size_t i = 0; i < n1; ++i) {
    const double prob_i = prob_[i];
    const int side_i = side_[i];
    const double open_i = open_[i];
    const double high_prev = (i ? high_[i - 1] : open_i);
    const double low_prev = (i ? low_[i - 1] : open_i);

    const int sel_i = selected_[i];
    const double tpb_i = tpb_[i];
    const double slb_i = slb_[i];
    const int64_t close_ts = close_ts_int_[i];
    const int64_t vb_exec = vb_ts_exec_[i];

    // open a new bet if the index is selected
    if (sel_i == 1) {
      active.push_back(Bet{get_size(prob_i), side_i, close_ts, open_i, tpb_i,
                           slb_i, vb_exec});
    }

    // close bets that hit any of the barriers
    active.erase(std::remove_if(active.begin(), active.end(),
                                [&](const Bet &b) {
                                  return b.should_close(
                                      close_ts, std::max(high_prev, open_i),
                                      std::min(low_prev, open_i));
                                }),
                 active.end());

    /* ---- position calculations ---- */
    const double raw_pos = average_bets(active);
    raw_out[i] = raw_pos;

    double rdqa = discretise(raw_pos, qa_step_size);
    double disc = std::clamp(qa_max * rdqa, -qa_max, qa_max);
    disc_out[i] = disc;

    if (i > 0 && disc_out[i - 1] == disc_out[i])
      ba_out[i] = ba_out[i - 1];
    else {
      double ba = disc / open_i;
      if (ba_step_size.has_value()) {
        const double step = *ba_step_size;
        ba = std::copysign(std::floor(std::abs(ba) / step), ba) * step;
      }
      ba_out[i] = ba;
    }
  }

  // apply position allowed
  for (std::size_t i = 0; i < n1; ++i) {
    raw_out[i] *= position_allowed_[i];
    disc_out[i] *= position_allowed_[i];
    ba_out[i] *= position_allowed_[i];
  }

  /* ----------  return dict with three arrays ---------- */
  pybind11::dict d;
  d["raw_qa_position"] = raw_out_arr;
  d["discretised_qa_position"] = disc_out_arr;
  d["final_ba_position"] = ba_out_arr;
  return d;
}

} // anonymous namespace

/* -----------------  Python module registration  ----------------- */
void register_probability_sizer(pybind11::module_ &m) {
  m.def("probability_position_sizer_cpp", &probability_position_sizer_cpp,
        pybind11::arg("prob"), pybind11::arg("side"), pybind11::arg("open"),
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("selected"),
        pybind11::arg("tpb"), pybind11::arg("slb"),
        pybind11::arg("close_timestamp_int"),
        pybind11::arg("vb_timestamp_exec_int"),
        pybind11::arg("position_allowed"), pybind11::arg("qa_step_size"),
        pybind11::arg("ba_step_size"), pybind11::arg("qa_max"),
        "Probability-based position sizer (C++)");
}

/*
def probability_position_sizer_(
  prob: pd.Series,  # probability of the prediction being correct
  side: pd.Series,  # 1 or -1
  open: pd.Series,  # open price series from OHLCV bars
  high: pd.Series,  # high price series from OHLCV bars
  low: pd.Series,  # low price series from OHLCV bars
  selected: pd.Series,  # whether a new bet can be opened on this index.
existing bets can be closed at all indices though. tpb: pd.Series,  # take
profit barriers; a take profit of 3% will be 0.03 slb: pd.Series,  # stop loss
barriers; signed; so a stop loss of 3% will be -0.03 close_timestamp_int:
pd.Series,  # integer representation of the index timestamp # example:
2021-01-01 03:45:59.999000+00:00 -> 1609472759999000000 # (can be found by doing
`df.index.astype(int)`) vb_timestamp_exec_int: pd.Series,  # integer timestamp
of vertical barriers # any bet opened here needs to be closed on or before this
timestamp. # the _exec (execution) suffix is needed as vertical barriers during
execution can # be different from vertical barriers during labelling. # example
- while labelling, we might always set the vertical barrier to be 1 hour away
from current time. # but during execution, we might need to always close all
positions before the end of # a session. #
------------------------------------------------------------------------------------------------
  # qa_step_size: step size for discretising the dimensionless position size
(i.e. between -1 and 1) # should be in (0,1] qa_step_size: float, # step size
for discretising position sizes in base asset (stock units, BTCUSDT units, etc.)
  ba_step_size: float | None,
  # maximum absolute position size in quote asset (INR, USD, USDT, etc.)
  qa_max: float,
) -> pd.DataFrame:
  if not prob.index.is_monotonic_increasing:
    raise ValueError("prob index must be monotonic increasing")
  if not prob.index.is_unique:
    raise ValueError("prob index must be unique")
  if not all(
    prob.index.equals(other.index)
    for other in [
      side,
      open,
      high,
      low,
      selected,
      tpb,
      slb,
      close_timestamp_int,
      vb_timestamp_exec_int,
    ]
  ):
    raise ValueError("All series must have the same index")

  if qa_step_size <= 0:
    raise ValueError("qa_step_size must be positive")
  if qa_step_size > 1:
    # qa_step_size acts on average bet size which is [0, 1]
    # so having it > 1 does not make sense
    raise ValueError("qa_step_size must be less than or equal to 1")
  if ba_step_size is not None and ba_step_size <= 0:
    raise ValueError("ba_step_size must be positive if provided")

  if qa_max <= 0:
    raise ValueError("qa_max must be positive")

  if not pd.api.types.is_integer_dtype(side):
    raise ValueError("side must be of integer dtype")
  if not side.isin([1, -1]).all():
    # only the binary case is supported for now
    raise ValueError("side values must be only 1 or -1")
  if not pd.api.types.is_integer_dtype(selected):
    raise ValueError("selected must be of integer dtype")
  if not selected.isin([0, 1]).all():
    raise ValueError("selected values must be only 0 or 1")
  if not ((prob >= 0.5) & (prob <= 1)).all():
    # if prob < 0.5, the user should send -side with probability (1 - prob)
    raise ValueError("prob values must be between 0.5 and 1 (inclusive)")
  if not pd.api.types.is_integer_dtype(close_timestamp_int):
    raise ValueError("close_timestamp_int must be of integer dtype")
  if not pd.api.types.is_integer_dtype(vb_timestamp_exec_int):
    raise ValueError("vb_timestamp_exec_int must be of integer dtype")

  n = len(prob)

  raw_qa_positions = np.empty(n, dtype=float)
  discretised_qa_positions = np.empty(n, dtype=float)
  final_ba_position = np.empty(n, dtype=float)
  active_bets: list[Bet] = []

  for i in range(n):
    prob_i = prob.iloc[i]
    side_i = side.iloc[i]
    open_price_i = open.iloc[i]
    # logic should work when data is coming as a stream
    # while streaming, this step will run when the bar has just started forming
    # say we are receiving the [t,t+1) bar right now.
    # this step will run as soon as we receive the first tick AFTER time t (that
will be the open price) # the previous [t-1,t) bar is fully formed # so to
evaluate the barrier hits, we have access to the high and low prices of the
[t-1,t) # we DO NOT have access to the high and low prices of the [t,t+1)
bar...we only have the open price # that's why we use the previous bar's high
and low prices high_price_prev = high.iloc[i - 1] if i - 1 >= 0 else
open_price_i low_price_prev = low.iloc[i - 1] if i - 1 >= 0 else open_price_i if
(open_price_i <= 0) or (high_price_prev <= 0) or (low_price_prev <= 0): raise
ValueError( "open_price, high_price_prev, and low_price_prev must be positive"
      )
    if not (
      np.isfinite(open_price_i)
      and np.isfinite(high_price_prev)
      and np.isfinite(low_price_prev)
    ):
      raise ValueError("open_price, high_price_prev, and low_price_prev must be
finite") sel_i = selected.iloc[i] tpb_i = tpb.iloc[i] slb_i = slb.iloc[i]
    close_ts_int_i = close_timestamp_int.iloc[i]
    vb_ts_int_exec_i = vb_timestamp_exec_int.iloc[i]

    if tpb_i <= 0 or slb_i >= 0:
      raise ValueError("tpb and slb must be positive and negative respectively")

    # open new bet if timestamp is selected by the data selector upstream
    if sel_i == 1:
      # prob_i is enforced to be >= 0.5
      # so size would be >= 0
      size = get_size(prob=prob_i)
      active_bets.append(
        Bet(
          size=size,
          side=side_i,
          bet_open_timestamp_int=close_ts_int_i,
          bet_entry_price=open_price_i,
          tpb=tpb_i,
          slb=slb_i,
          vb_timestamp_exec_int=vb_ts_int_exec_i,
        )
      )

    # close bets that hit any of the barriers
    # this is done after opening a new bet to handle the case where the
    # new bet needs to be closed as well
    active_bets = [
      bet
      for bet in active_bets
      if not bet.should_close(
        current_timestamp_int=close_ts_int_i,
        highest_price=max(high_price_prev, open_price_i),
        lowest_price=min(low_price_prev, open_price_i),
      )
    ]

    raw_qa_positions[i] = average_bets(bets=active_bets)
    # discretise_position does bankers rounding and does not always round down
    # this is fine since we clip the position to be between -qa_max and qa_max
    # on the next line
    rdqa_pos_i = discretise_position(
      position=raw_qa_positions[i], step_size=qa_step_size
    )
    discretised_qa_positions[i] = np.clip(
      a=qa_max * rdqa_pos_i,
      a_min=-qa_max,
      a_max=qa_max,
    )

    # Prevent unnecessary base asset position changes when quote asset position
is unchanged. # This avoids spurious trades caused by price movements when the
underlying position # sizing decision (discretised_qa_position) hasn't actually
changed. if i > 0 and discretised_qa_positions[i - 1] ==
discretised_qa_positions[i]: final_ba_position[i] = final_ba_position[i - 1]
    else:
      final_ba_position[i] = discretised_qa_positions[i] / open_price_i
      if ba_step_size is not None:
        # always round down in magnitude to the nearest step size
        final_ba_position[i] = (
          np.sign(final_ba_position[i])
          * np.floor(np.abs(final_ba_position[i]) / ba_step_size)
          * ba_step_size
        )

  return pd.DataFrame(
    {
      # this is the raw dimensionless position size between -1 and 1
      "raw_qa_position": raw_qa_positions,
      # this is the quote asset (INR, USD, USDT etc.) position size between
-qa_max and qa_max "discretised_qa_position": discretised_qa_positions, # this
is in # of stock units, # of BTCUSDT units, etc. "final_ba_position":
final_ba_position,
    },
    index=prob.index,
  )

*/