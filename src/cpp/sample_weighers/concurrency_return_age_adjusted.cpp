#include "../utils/io.cpp"
#include <cmath>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();

void _get_concurrency(const double *label_last_indices, std::size_t n,
                      double *concurrency) {
  // concurrency[i] is the number of labels that used the prices[i-1] to
  // prices[i] return (concurrency[0] is 0 since no label uses the prices[-1]
  // to prices[0] return)

  for (std::size_t i = 0; i < n; ++i) {
    concurrency[i] = 0.0;
  }
  if (n == 0) {
    return;
  }

  std::vector<double> diff(n, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(label_last_indices[i])) {
      continue;
    }

    std::size_t t_in = i + 1;
    // label_last_indices contains positional indices (0-based) into the
    // arrays, NOT the actual DataFrame index values
    std::size_t t_out =
        static_cast<std::size_t>(std::round(label_last_indices[i]));
    // to generate label[i], we used prices[i] to prices[lli[i]] returns
    // => we used prices[i],prices[i+1] | prices[i+1],prices[i+2] | ... |
    // prices[lli[i]-1],prices[lli[i]] returns
    // => lli[i]-i returns were used
    // (lli means label_last_indices)
    if ((t_out < t_in) || (t_out >= n)) {
      throw std::runtime_error("Invalid values for index " + std::to_string(i) +
                               "; got t_in=" + std::to_string(t_in) +
                               ", t_out=" + std::to_string(t_out));
    }

    diff[t_in] += 1.0;
    if (t_out + 1 < n) { // guard: diff has size n
      diff[t_out + 1] -= 1.0;
    }
  }

  // Prefix sum to convert the difference array to the actual counts.
  double running = 0.0;
  for (std::size_t t = 1; t < n; ++t) { // concurrency[0] stays 0.0
    running += diff[t];
    concurrency[t] = running;
  }
}

void _get_log_returns(const double *prices, std::size_t n,
                      double *log_returns) {
  log_returns[0] = NaN;
  for (std::size_t t = 1; t < n; ++t) {
    log_returns[t] = std::log(prices[t]) - std::log(prices[t - 1]);
  }
}

void _get_weights_raw(const double *label_last_indices,
                      const double *log_returns, const double *concurrency,
                      std::size_t n, double *attribution_weights_raw,
                      double *avg_uniqueness) {

  for (std::size_t i = 0; i < n; ++i) {
    attribution_weights_raw[i] = NaN;
    avg_uniqueness[i] = NaN;
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(label_last_indices[i])) {
      continue;
    }

    std::size_t t_in = i + 1;
    std::size_t t_out =
        static_cast<std::size_t>(std::round(label_last_indices[i]));
    // to generate label[i], we used prices[i] to prices[lli[i]] returns
    // => we used prices[i],prices[i+1] | prices[i+1],prices[i+2] | ... |
    // prices[lli[i]-1],prices[lli[i]] returns
    // => lli[i]-i returns were used
    // (lli means label_last_indices)
    std::size_t lifespan = t_out - t_in + 1;
    // => lifespan is also t_out - t_in + 1 = lli[i]-i

    double sum_uniqueness = 0.0;
    double sum_attributed_return = 0.0;

    for (std::size_t t = t_in; t <= t_out; ++t) {
      if (concurrency[t] == 0) {
        throw std::runtime_error("Concurrency at index " + std::to_string(t) +
                                 " is zero, which should never happen");
      }
      double inv_c = 1.0 / concurrency[t];
      sum_uniqueness += inv_c;
      sum_attributed_return += log_returns[t] / concurrency[t];
    }

    avg_uniqueness[i] = sum_uniqueness / static_cast<double>(lifespan);
    attribution_weights_raw[i] = std::abs(sum_attributed_return);
  }
}

void _get_time_decay_factors(const double *label_last_indices,
                             const double *avg_uniqueness, std::size_t n,
                             double time_decay_c, double *decay_factors) {

  if ((!std::isfinite(time_decay_c)) || time_decay_c < 0 || time_decay_c > 1) {
    // time_decay_c < 0 is disabled. This mode is to discard the last
    // time_decay_c fraction of events. If you want to do such a thing, just
    // drop those data points explicitly. Refer to page 70 of Advances in
    // Financial Machine Learning for the logic in case we want to enable this
    // mode in the future.
    throw std::runtime_error("time_decay_c must be a finite number in [0,1]");
  }

  // Decay is based on cumulative uniqueness, not chronological time.
  std::vector<double> cumulative_uniqueness(n, NaN);
  double running_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(label_last_indices[i])) {
      continue;
    }

    running_sum += avg_uniqueness[i];
    cumulative_uniqueness[i] = running_sum;
  }

  double total_uniqueness = running_sum;

  // df(x) = intercept + slope * x
  // Boundary conditions: df(total_uniqueness)=1, df(0)=c
  // => intercept = c
  // => slope = (1 - c) / total_uniqueness
  if (total_uniqueness == 0) {
    throw std::runtime_error("total_uniqueness must be greater than 0");
  }
  double slope = (1.0 - time_decay_c) / total_uniqueness;
  double intercept = time_decay_c;

  for (std::size_t i = 0; i < n; ++i) {
    decay_factors[i] = NaN;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(label_last_indices[i])) {
      continue;
    }
    double decay_factor = intercept + slope * cumulative_uniqueness[i];

    decay_factors[i] = decay_factor;
  }
}

pybind11::dict concurrency_return_age_adjusted_weights_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast>
        &label_last_indices,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &prices,
    double time_decay_c) {

  /* --------------------  borrow raw pointers / sizes -------------------- */
  const auto [lli_, n1] = atv40::io::get_input_ptr<double>(
      label_last_indices, "label_last_indices");
  const auto [prices_, n2] = atv40::io::get_input_ptr<double>(prices, "prices");

  if (n2 != n1) {
    throw std::runtime_error("Array length mismatch");
  }

  const std::size_t n = n1;

  /* --------------------  allocate output buffers  ----------------------- */
  auto [concurrency_arr, concurrency] = atv40::io::make_output_array<double>(n);
  auto [log_returns_arr, log_returns] = atv40::io::make_output_array<double>(n);
  auto [attribution_weights_raw_arr, attribution_weights_raw] =
      atv40::io::make_output_array<double>(n);
  auto [avg_uniqueness_arr, avg_uniqueness] =
      atv40::io::make_output_array<double>(n);
  auto [decay_factors_arr, decay_factors] =
      atv40::io::make_output_array<double>(n);
  auto [sample_weight_arr, sample_weight] =
      atv40::io::make_output_array<double>(n);

  /* -------------------------- main computation -------------------------- */
  _get_concurrency(lli_, n, concurrency);

  _get_log_returns(prices_, n, log_returns);

  _get_weights_raw(lli_, log_returns, concurrency, n, attribution_weights_raw,
                   avg_uniqueness);

  _get_time_decay_factors(lli_, avg_uniqueness, n, time_decay_c, decay_factors);

  // Compute final sample weights
  for (std::size_t i = 0; i < n; ++i) {
    sample_weight[i] = attribution_weights_raw[i] * decay_factors[i];
  }

  // The weights are not normalised here to sum to the number of samples that
  // go in the ML model. This should be done right before training the ML model
  // downstream.

  /* ---------------------------   return dict  --------------------------- */
  pybind11::dict out;
  out["concurrency"] = concurrency_arr;
  out["attribution_weight_raw"] = attribution_weights_raw_arr;
  out["avg_uniqueness"] = avg_uniqueness_arr;
  out["time_decay_factor"] = decay_factors_arr;
  out["sample_weight"] = sample_weight_arr;
  return out;
}

} // anonymous namespace

/* -------------------   Python registration helper   --------------------- */
void register_concurrency_return_age_adjusted_weights(pybind11::module_ &m) {
  m.def("concurrency_return_age_adjusted_weights_cpp",
        &concurrency_return_age_adjusted_weights_cpp,
        pybind11::arg("label_last_indices"), pybind11::arg("prices"),
        pybind11::arg("time_decay_c"),
        "C++ implementation of the concurrency return age adjusted "
        "sample weighting algorithm");
}