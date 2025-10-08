#include "../utils/io.cpp"
#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();
const int32_t FIRST_TOUCH_TYPE_TP = 1;
const int32_t FIRST_TOUCH_TYPE_SL = -1;
const int32_t FIRST_TOUCH_TYPE_VB = 0;

pybind11::dict triple_barrier_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &prices,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &selected,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &tpb,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &slb,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &vb_tte,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &tte,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast> &side) {

  /* --------------------  borrow raw pointers / sizes -------------------- */
  const auto [prices_, n1] = atv40::io::get_input_ptr<double>(prices, "prices");
  const auto [selected_, n2] =
      atv40::io::get_input_ptr<int32_t>(selected, "selected");
  const auto [tpb_, n3] = atv40::io::get_input_ptr<double>(tpb, "tpb");
  const auto [slb_, n4] = atv40::io::get_input_ptr<double>(slb, "slb");
  const auto [vb_tte_, n5] =
      atv40::io::get_input_ptr<int32_t>(vb_tte, "vb_tte");
  const auto [tte_, n6] = atv40::io::get_input_ptr<int32_t>(tte, "tte");
  const auto [side_, n7] = atv40::io::get_input_ptr<int32_t>(side, "side");

  /* --------------------  allocate output buffers  ----------------------- */
  // tpha: take profit hit at
  auto [tpha_arr, tpha] = atv40::io::make_output_array<double>(n1);
  // slha: stop loss hit at
  auto [slha_arr, slha] = atv40::io::make_output_array<double>(n1);
  // vbha: vertical barrier hit at
  auto [vbha_arr, vbha] = atv40::io::make_output_array<double>(n1);
  // fta: first touch at
  auto [fta_arr, fta] = atv40::io::make_output_array<double>(n1);
  // fttype: first touch type
  auto [fttype_arr, fttype] = atv40::io::make_output_array<double>(n1);
  // ftret: first touch return
  auto [ftret_arr, ftret] = atv40::io::make_output_array<double>(n1);

  /* ----------------------------- main loop ------------------------------ */
  for (std::size_t i = 0; i < n1; ++i) {
    /* ----- early exit when no search is needed ----- */
    if (selected_[i] == 0) {
      tpha[i] = slha[i] = vbha[i] = NaN;
      fta[i] = fttype[i] = ftret[i] = NaN;
      continue;
    }

    /* ----- scan forward to detect barrier hits ----- */
    double tp_idx = NaN;
    double sl_idx = NaN;
    double v_idx = NaN;

    // find the first index where take profit barrier is hit
    if (!std::isnan(tpb_[i])) {
      for (std::size_t j = i; j < n1; ++j) {
        const double ret_ij =
            ((prices_[j] / prices_[i]) - 1.0) * static_cast<double>(side_[i]);
        if (ret_ij >= tpb_[i]) {
          tp_idx = static_cast<double>(j);
          break;
        }
      }
    }

    // find the first index where stop loss barrier is hit
    if (!std::isnan(slb_[i])) {
      for (std::size_t j = i; j < n1; ++j) {
        const double ret_ij =
            ((prices_[j] / prices_[i]) - 1.0) * static_cast<double>(side_[i]);
        if (ret_ij <= slb_[i]) {
          sl_idx = static_cast<double>(j);
          break;
        }
      }
    }

    // find the first index where vertical barrier is hit
    if (!std::isnan(vb_tte_[i])) {
      for (std::size_t j = i; j < n1; ++j) {
        const int dtte = tte_[j] - tte_[i];
        if (dtte >= vb_tte_[i]) {
          v_idx = static_cast<double>(j);
          break;
        }
      }
    }

    tpha[i] = tp_idx;
    slha[i] = sl_idx;
    vbha[i] = v_idx;
  }

  /* -------------------  first-touch calculations  ----------------------- */
  for (std::size_t i = 0; i < n1; ++i) {
    const double tpha_i = tpha[i];
    const double slha_i = slha[i];
    const double vbha_i = vbha[i];

    // nan-aware minimum
    double first_at = NaN;
    for (double v : {tpha_i, slha_i, vbha_i})
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

    if (tpha_i == first_at) {
      fttype[i] = FIRST_TOUCH_TYPE_TP;
      ftret[i] = (prices_[static_cast<std::size_t>(tpha_i)] / prices_[i]) - 1.0;
    } else if (slha_i == first_at) {
      fttype[i] = FIRST_TOUCH_TYPE_SL;
      ftret[i] = (prices_[static_cast<std::size_t>(slha_i)] / prices_[i]) - 1.0;
    } else { // vertical barrier
      fttype[i] = FIRST_TOUCH_TYPE_VB;
      ftret[i] = (prices_[static_cast<std::size_t>(vbha_i)] / prices_[i]) - 1.0;
    }
  }

  /* ---------------------------   return dict  --------------------------- */
  pybind11::dict out;
  out["tpha"] = tpha_arr;
  out["slha"] = slha_arr;
  out["vbha"] = vbha_arr;
  out["first_touch_at"] = fta_arr;
  out["first_touch_type"] = fttype_arr;
  out["first_touch_raw_return"] = ftret_arr;
  return out;
}

} // anonymous namespace

/* -------------------   Python registration helper   -------------------- */
void register_triple_barrier(pybind11::module_ &m) {
  m.def("triple_barrier_cpp", &triple_barrier_cpp, pybind11::arg("prices"),
        pybind11::arg("selected"), pybind11::arg("tpb"), pybind11::arg("slb"),
        pybind11::arg("vb_tte"), pybind11::arg("tte"), pybind11::arg("side"),
        "Fast C++ implementation of the triple-barrier labeling algorithm");
}