#include "../utils/io.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

static pybind11::dict time_based_uniform_cpp(
    const pybind11::array_t<int, pybind11::array::c_style |
                                     pybind11::array::forcecast>
        &minutes_from_first_row,
    const pybind11::array_t<int, pybind11::array::c_style |
                                     pybind11::array::forcecast> &day_offsets,
    int group_size_minutes, int offset_minutes, int overnight_gap_minutes) {
  if (group_size_minutes <= 0)
    throw std::runtime_error("group_size_minutes must be positive");
  if (offset_minutes < 0 || offset_minutes >= group_size_minutes)
    throw std::runtime_error("offset_minutes out of range");

  const auto [mins, n] = atv40::io::get_input_ptr<int>(
      minutes_from_first_row, "minutes_from_first_row");
  const auto [days, n2] =
      atv40::io::get_input_ptr<int>(day_offsets, "day_offsets");
  if (n != n2)
    throw std::runtime_error(
        "minutes_from_first_row and day_offsets must have equal length");

  /* output arrays */
  auto [gsp_arr, gsp] = atv40::io::make_output_array<int>(n);
  auto [offs_arr, off] = atv40::io::make_output_array<int>(n);

  if (n == 0) {
    pybind11::dict out;
    out["group_start_positions"] = gsp_arr;
    out["offsets"] = offs_arr;
    return out;
  }

  std::vector<int> cumulative(n);
  int grouped_elapsed_prev;

  /* -- initialise first row ------------------------------------------------ */
  cumulative[0] = group_size_minutes - offset_minutes;
  grouped_elapsed_prev = static_cast<int>(group_size_minutes) *
                         (cumulative[0] / group_size_minutes);

  gsp[0] = 0;
  off[0] = static_cast<int>(cumulative[0] % group_size_minutes);

  /* -- main loop ----------------------------------------------------------- */
  for (std::size_t i = 1; i < n; ++i) {
    int cm = cumulative[i - 1] + (mins[i] - mins[i - 1]);

    /* adjust for overnight / multi-day gaps */
    if (days[i] != days[i - 1]) {
      int date_diff = days[i] - days[i - 1]; // ≥ 1
      cm -= overnight_gap_minutes + static_cast<int>(date_diff - 1) * 1440;
    }

    cumulative[i] = cm;

    int grouped_elapsed =
        static_cast<int>(group_size_minutes) * (cm / group_size_minutes);

    gsp[i] = (grouped_elapsed != grouped_elapsed_prev) ? static_cast<int>(i)
                                                       : gsp[i - 1];

    off[i] = static_cast<int>(cm % group_size_minutes);
    grouped_elapsed_prev = grouped_elapsed;
  }

  /* pack result ------------------------------------------------------------ */
  pybind11::dict out;
  out["group_start_positions"] = gsp_arr;
  out["offsets"] = offs_arr;
  return out;
}

/* registration helper -------------------------------------------------------
 */
void register_time_based_uniform(pybind11::module_ &m) {
  m.def("time_based_uniform_cpp", &time_based_uniform_cpp,
        pybind11::arg("minutes_from_first_row"), pybind11::arg("day_offsets"),
        pybind11::arg("group_size_minutes"), pybind11::arg("offset_minutes"),
        pybind11::arg("overnight_gap_minutes"),
        "Helper to compute time-based uniform bar grouping for the Indian "
        "market");
}