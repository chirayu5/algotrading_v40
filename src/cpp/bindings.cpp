#include <pybind11/pybind11.h>

// Forward declarations of the helpers implemented in the feature files
void register_rsi(pybind11::module_ &);
void register_detrended_rsi(pybind11::module_ &);
void register_stochastic(pybind11::module_ &);
// add more as you create new .cpp files

PYBIND11_MODULE(algotrading_v40_cpp, m) {
  m.doc() = "algotrading_v40 consolidated C++ extension";

  // create a sub-package for logical grouping
  auto fc = m.def_submodule("feature_calculators", "feature calculators");

  register_rsi(fc);
  register_detrended_rsi(fc);
  register_stochastic(fc);
  // call additional register_* helpers here
}