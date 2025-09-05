#include "../utils/io.cpp"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

static pybind11::array_t<int32_t> cusum_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &s_diff,
    double h) {

  const auto [s_diff_, n] = atv40::io::get_input_ptr<double>(s_diff, "s_diff");

  if (h <= 0.0) {
    throw std::runtime_error("cusum threshold must be greater than 0");
  }

  // Create output array
  auto [out, output] = atv40::io::make_output_array<int32_t>(n);

  double s_pos = 0.0;
  double s_neg = 0.0;

  // First element is always 0
  if (n > 0) {
    output[0] = 0;
  }

  // Main CUSUM algorithm loop
  for (std::size_t i = 1; i < n; ++i) {
    s_pos = std::max(0.0, s_pos + s_diff_[i]);
    s_neg = std::min(0.0, s_neg + s_diff_[i]);

    if (s_neg < -h) {
      s_neg = 0.0;
      output[i] = -1;
    } else if (s_pos > h) {
      s_pos = 0.0;
      output[i] = 1;
    } else {
      output[i] = 0;
    }
  }

  return out;
}

void register_cusum(pybind11::module_ &m) {
  m.def("cusum_cpp", &cusum_cpp, pybind11::arg("s_diff"), pybind11::arg("h"),
        "CUSUM filter for event detection");
}