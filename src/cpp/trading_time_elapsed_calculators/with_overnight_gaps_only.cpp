#include "../utils/io.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

static pybind11::array_t<int> with_overnight_gaps_only_cpp(
    const pybind11::array_t<int, pybind11::array::c_style |
                                     pybind11::array::forcecast>
        &minutes_from_first_data_point,
    const pybind11::array_t<int, pybind11::array::c_style |
                                     pybind11::array::forcecast>
        &days_from_first_day,
    int overnight_gap_minutes) {
  if (overnight_gap_minutes <= 0)
    throw std::runtime_error("overnight_gap_minutes must be positive");

  const auto [mins, n] = atv40::io::get_input_ptr<int>(
      minutes_from_first_data_point, "minutes_from_first_data_point");
  const auto [days, n2] =
      atv40::io::get_input_ptr<int>(days_from_first_day, "days_from_first_day");
  if (n != n2)
    throw std::runtime_error("minutes_from_first_data_point and "
                             "days_from_first_day must have equal length");

  /* output arrays */
  auto [tte_arr, tte] = atv40::io::make_output_array<int>(n);

  if (n == 0) {
    return tte_arr;
  }
  tte[0] = 0;
  for (std::size_t i = 1; i < n; ++i) {
    tte[i] = tte[i - 1] + (mins[i] - mins[i - 1]);
    if (days[i] != days[i - 1]) {
      int date_diff = days[i] - days[i - 1]; // >= 1
      tte[i] -= overnight_gap_minutes + static_cast<int>(date_diff - 1) * 1440;
    }
  }
  return tte_arr;
}

void register_with_overnight_gaps_only(pybind11::module_ &m) {
  m.def("with_overnight_gaps_only_cpp", &with_overnight_gaps_only_cpp,
        pybind11::arg("minutes_from_first_data_point"),
        pybind11::arg("days_from_first_day"),
        pybind11::arg("overnight_gap_minutes"),
        "Compute trading time elapsed accounting for overnight gaps "
        "only");
}