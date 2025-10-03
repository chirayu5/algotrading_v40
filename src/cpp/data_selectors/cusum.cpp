#include "../utils/io.cpp"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<int32_t> cusum_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &s_diff,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &thresholds,
    const pybind11::array_t<int32_t, pybind11::array::c_style |
                                         pybind11::array::forcecast>
        &position_allowed) {

  const auto [s_diff_, n] = atv40::io::get_input_ptr<double>(s_diff, "s_diff");
  const auto [thresholds_, n2] =
      atv40::io::get_input_ptr<double>(thresholds, "thresholds");
  const auto [position_allowed_, n3] =
      atv40::io::get_input_ptr<int32_t>(position_allowed, "position_allowed");
  if (n != n2 || n != n3) {
    throw std::runtime_error("s_diff and thresholds must have the same length");
  }

  // Create output array
  auto [out, output] = atv40::io::make_output_array<int32_t>(n);
  if (n == 0) {
    return out;
  }

  double s_pos = 0.0;
  double s_neg = 0.0;

  // First element is always 0
  output[0] = 0;

  // Main CUSUM algorithm loop
  for (std::size_t i = 1; i < n; ++i) {
    s_pos = std::max(0.0, s_pos + s_diff_[i]);
    s_neg = std::min(0.0, s_neg + s_diff_[i]);

    if ((s_neg < -thresholds_[i]) && (position_allowed_[i] == 1)) {
      s_neg = 0.0;
      output[i] = -1;
    } else if ((s_pos > thresholds_[i]) && (position_allowed_[i] == 1)) {
      s_pos = 0.0;
      output[i] = 1;
    } else {
      output[i] = 0;
    }
  }

  return out;
}

void register_cusum(pybind11::module_ &m) {
  m.def("cusum_cpp", &cusum_cpp, pybind11::arg("s_diff"),
        pybind11::arg("thresholds"), pybind11::arg("position_allowed"),
        "CUSUM filter for event detection");
}