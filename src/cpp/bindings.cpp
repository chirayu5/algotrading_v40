#include <pybind11/pybind11.h>

// Forward declarations of the helpers implemented
// feature calculators
void register_rsi(pybind11::module_ &);
void register_detrended_rsi(pybind11::module_ &);
void register_stochastic(pybind11::module_ &);
void register_stochastic_rsi(pybind11::module_ &);
void register_ma_diff(pybind11::module_ &);
void register_macd(pybind11::module_ &);

// utils
void register_features(pybind11::module_ &);

PYBIND11_MODULE(algotrading_v40_cpp, m) {
  m.doc() = "algotrading_v40 consolidated C++ extension";

  // submodule: feature calculators
  auto fc = m.def_submodule("feature_calculators", "feature calculators");
  register_rsi(fc);
  register_detrended_rsi(fc);
  register_stochastic(fc);
  register_stochastic_rsi(fc);
  register_ma_diff(fc);
  register_macd(fc);

  // submodule: utils
  auto utils = m.def_submodule("utils", "utils");
  register_features(utils);
}
