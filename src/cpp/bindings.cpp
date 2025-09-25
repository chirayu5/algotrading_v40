#include <pybind11/pybind11.h>

// Forward declarations of the helpers implemented

// bar groupers
void register_time_based_uniform(pybind11::module_ &);

// data selectors
void register_cusum(pybind11::module_ &);

// feature calculators
void register_rsi(pybind11::module_ &);
void register_detrended_rsi(pybind11::module_ &);
void register_stochastic(pybind11::module_ &);
void register_stochastic_rsi(pybind11::module_ &);
void register_ma_diff(pybind11::module_ &);
void register_macd(pybind11::module_ &);
void register_lin_quad_cubic_trend(pybind11::module_ &);
void register_price_intensity(pybind11::module_ &);
void register_adx(pybind11::module_ &);
void register_close_minus_ma(pybind11::module_ &);

// labellers
void register_triple_barrier(pybind11::module_ &);

// position sizers
void register_probability_sizer(pybind11::module_ &);

// utils
void register_features(pybind11::module_ &);

PYBIND11_MODULE(algotrading_v40_cpp, m) {
  m.doc() = "algotrading_v40 consolidated C++ extension";

  // submodule: bar groupers
  auto bg = m.def_submodule("bar_groupers", "bar groupers");
  register_time_based_uniform(bg);

  // submodule: data selectors
  auto ds = m.def_submodule("data_selectors", "data selectors");
  register_cusum(ds);

  // submodule: feature calculators
  auto fc = m.def_submodule("feature_calculators", "feature calculators");
  register_rsi(fc);
  register_detrended_rsi(fc);
  register_stochastic(fc);
  register_stochastic_rsi(fc);
  register_ma_diff(fc);
  register_macd(fc);
  register_lin_quad_cubic_trend(fc);
  register_price_intensity(fc);
  register_adx(fc);
  register_close_minus_ma(fc);

  // submodule: labellers
  auto lab = m.def_submodule("labellers", "labellers");
  register_triple_barrier(lab);

  // submodule: position sizers
  auto ps = m.def_submodule("position_sizers", "position sizers");
  register_probability_sizer(ps);

  // submodule: utils
  auto utils = m.def_submodule("utils", "utils");
  register_features(utils);
}
