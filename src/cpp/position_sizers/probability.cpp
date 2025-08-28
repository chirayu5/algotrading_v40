
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
        &vb_ts_exec,
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
      atv40::io::get_input_ptr<int64_t>(vb_ts_exec, "vb_ts_exec");

  for (auto nii : {n1, n2, n3, n4, n5, n6, n7, n8, n9, n10}) {
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

    // open new bet
    if (sel_i == 1) {
      active.push_back(Bet{get_size(prob_i), side_i, close_ts, open_i, tpb_i,
                           slb_i, vb_exec});
    }

    // close triggered bets
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
        pybind11::arg("vb_timestamp_int_exec"), pybind11::arg("qa_step_size"),
        pybind11::arg("ba_step_size"), pybind11::arg("qa_max"),
        "Probability-based position sizer (C++)");
}
