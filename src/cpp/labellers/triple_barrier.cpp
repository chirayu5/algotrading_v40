#include "../utils/io.cpp"
#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

pybind11::dict triple_barrier_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &s,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &selected,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &tpb,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &slb,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &vb,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &side) {

  /* --------------------  borrow raw pointers / sizes -------------------- */
  const auto [s_, n1] = atv40::io::get_input_ptr<double>(s, "s");
  const auto [selected_, n2] =
      atv40::io::get_input_ptr<int32_t>(selected, "selected");
  const auto [tpb_, n3] = atv40::io::get_input_ptr<double>(tpb, "tpb");
  const auto [slb_, n4] = atv40::io::get_input_ptr<double>(slb, "slb");
  const auto [vb_, n5] = atv40::io::get_input_ptr<int32_t>(vb, "vb");
  const auto [side_, n6] = atv40::io::get_input_ptr<int32_t>(side, "side");

  for (auto nn : {n2, n3, n4, n5, n6})
    if (nn != n1)
      throw std::runtime_error("Array length mismatch");

  const double NaN = std::numeric_limits<double>::quiet_NaN();

  /* --------------------  allocate output buffers  ----------------------- */
  auto [tpha_arr, tpha] = atv40::io::make_output_array<double>(n1);
  auto [slha_arr, slha] = atv40::io::make_output_array<double>(n1);
  auto [vbha_arr, vbha] = atv40::io::make_output_array<double>(n1);
  auto [fta_arr, fta] = atv40::io::make_output_array<double>(n1);
  auto [fttype_arr, fttype] = atv40::io::make_output_array<double>(n1);
  auto [ftret_arr, ftret] = atv40::io::make_output_array<double>(n1);

  /* ----------------------------- main loop ------------------------------ */
  for (std::size_t i = 0; i < n1; ++i) {
    /* ----- early exit when no search is needed ----- */
    if (selected_[i] == 0) {
      tpha[i] = slha[i] = vbha[i] = NaN;
      continue;
    }
    if (vb_[i] == static_cast<int32_t>(i)) {
      throw std::runtime_error(
          "Vertical barrier must be greater than its index");
    }

    /* ----- scan forward to detect barrier hits ----- */
    double tp_idx = NaN;
    double sl_idx = NaN;

    // find the first index where take profit barrier is hit
    for (std::size_t j = i; j < n1; ++j) {
      const double ret_ij =
          ((s_[j] / s_[i]) - 1.0) * static_cast<double>(side_[i]);
      if (ret_ij >= tpb_[i]) {
        tp_idx = static_cast<double>(j);
        break;
      }
    }

    // find the first index where stop loss barrier is hit
    for (std::size_t j = i; j < n1; ++j) {
      const double ret_ij =
          ((s_[j] / s_[i]) - 1.0) * static_cast<double>(side_[i]);
      if (ret_ij <= slb_[i]) {
        sl_idx = static_cast<double>(j);
        break;
      }
    }

    tpha[i] = tp_idx;
    slha[i] = sl_idx;

    // first index where vertical barrier is hit
    vbha[i] =
        (vb_[i] < static_cast<int32_t>(n1)) ? static_cast<double>(vb_[i]) : NaN;
  }

  /* -------------------  first-touch calculations  ----------------------- */
  for (std::size_t i = 0; i < n1; ++i) {
    const double tp = tpha[i];
    const double sl = slha[i];
    const double vbv = vbha[i];

    // nan-aware minimum
    double first_at = NaN;
    for (double v : {tp, sl, vbv})
      if ((!std::isnan(v)) && (std::isnan(first_at) || (v < first_at)))
        first_at = v;

    fta[i] = first_at;

    // first_touch_type (fttype):
    //   1: take profit barrier hit
    //   -1: stop loss barrier hit
    //   0: vertical barrier hit
    //   np.nan: no barrier hit

    if (std::isnan(first_at)) {
      fttype[i] = ftret[i] = NaN;
      continue;
    }

    if (tp == first_at) {
      fttype[i] = 1.0;
      ftret[i] = (s_[static_cast<std::size_t>(tp)] / s_[i]) - 1.0;
    } else if (sl == first_at) {
      fttype[i] = -1.0;
      ftret[i] = (s_[static_cast<std::size_t>(sl)] / s_[i]) - 1.0;
    } else { // vertical barrier
      fttype[i] = 0.0;
      ftret[i] = (s_[static_cast<std::size_t>(vbv)] / s_[i]) - 1.0;
    }
  }

  /* ---------------------------   return dict  --------------------------- */
  pybind11::dict out;
  out["tpha"] = tpha_arr;
  out["slha"] = slha_arr;
  out["vbha"] = vbha_arr;
  out["first_touch_at"] = fta_arr;
  out["first_touch_type"] = fttype_arr;
  out["first_touch_return"] = ftret_arr;
  return out;
}

} // anonymous namespace

/* -------------------   Python registration helper   -------------------- */
void register_triple_barrier(pybind11::module_ &m) {
  m.def("triple_barrier_cpp", &triple_barrier_cpp, pybind11::arg("s"),
        pybind11::arg("selected"), pybind11::arg("tpb"), pybind11::arg("slb"),
        pybind11::arg("vb"), pybind11::arg("side"),
        "Fast C++ implementation of the triple-barrier labeling algorithm");
}