#include "../utils/features.hpp"
#include "../utils/io.cpp"
#include "../utils/stats.hpp"
#include <cassert>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double>
macd_cpp(const pybind11::array_t<double, pybind11::array::c_style |
                                             pybind11::array::forcecast> &high,
         const pybind11::array_t<double, pybind11::array::c_style |
                                             pybind11::array::forcecast> &low,
         const pybind11::array_t<double, pybind11::array::c_style |
                                             pybind11::array::forcecast> &close,
         int short_length, int long_length, int n_to_smooth) {
  const auto [high_, n1] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n2] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n3] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3) {
    throw std::runtime_error("high, low, and close must have the same length");
  }
  const int n = n1;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int front_bad =
      long_length +
      n_to_smooth; // Somewhat arbitrary because exponential smoothing
  if (front_bad > n)
    front_bad = n;

  double long_alpha, short_alpha, long_sum, short_sum, diff, denom, k, smoothed,
      alpha;
  int icase;

  long_alpha = 2.0 / (long_length + 1.0);
  short_alpha = 2.0 / (short_length + 1.0);

  long_sum = short_sum = close_[0];
  output[0] = 0.0; // This would be poorly defined
  for (icase = 1; icase < n; icase++) {

    // Compute long-term and short-term exponential smoothing
    long_sum = long_alpha * close_[icase] + (1.0 - long_alpha) * long_sum;
    short_sum = short_alpha * close_[icase] + (1.0 - short_alpha) * short_sum;

    // Compute the normalizing factor, then multiply it by atr to get scaling
    // factor

    diff = 0.5 * (long_length - 1.0); // Center of long block
    diff -= 0.5 * (short_length -
                   1.0); // Minus center of short block for random walk variance
    denom = sqrt(
        fabs(diff)); // Absolute value should never be needed if careful caller
    k = long_length + n_to_smooth;
    if (k > icase) // ATR requires case at least equal to length
      k = icase;   // Which will not be true at the beginning
    denom *= atr_cpp(0, icase, k, high, low, close);

    // These are the two scalings.  To skip scaling, just use short_sum -
    // long_sum.
    output[icase] = (short_sum - long_sum) / (denom + 1.e-15);
    output[icase] = 100.0 * normal_cdf_cpp(1.0 * output[icase]) - 50.0;
  } // For all cases

  // Smooth and compute differences if requested
  if (n_to_smooth > 1) {
    alpha = 2.0 / (n_to_smooth + 1.0);
    smoothed = output[0];
    for (icase = 1; icase < n; icase++) {
      smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed;
      output[icase] -= smoothed;
    } // For all cases
  } // If n_to_smooth > 1

  for (int icase = 0; icase < front_bad; ++icase) {
    output[icase] = std::numeric_limits<double>::quiet_NaN();
  }
  output[0] = std::numeric_limits<double>::quiet_NaN();

  return out;
}

void register_macd(pybind11::module_ &m) {
  m.def("macd_cpp", &macd_cpp, pybind11::arg("high"), pybind11::arg("low"),
        pybind11::arg("close"), pybind11::arg("short_length"),
        pybind11::arg("long_length"), pybind11::arg("n_to_smooth"),
        "Compute the MACD indicator");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  MACD (Moving Average Convergence Divergence)
//*************************************************************
            
   else if (var_num == VAR_MACD) {
      short_length = (int) (param1 + 0.5) ;
      long_length = (int) (param2 + 0.5) ;
      n_to_smooth = (int) (param3 + 0.5) ;
      front_bad = long_length + n_to_smooth ;  // Somewhat arbitrary because exponential smoothing
      if (front_bad > n)
         front_bad = n ;
      back_bad = 0 ;

      long_alpha = 2.0 / (long_length + 1.0) ;
      short_alpha = 2.0 / (short_length + 1.0) ;

      long_sum = short_sum = close[0] ;
      output[0] = 0.0 ;   // This would be poorly defined
      for (icase=1 ; icase<n ; icase++) {

         // Compute long-term and short-term exponential smoothing
         long_sum = long_alpha * close[icase] + (1.0 - long_alpha) * long_sum ;
         short_sum = short_alpha * close[icase] + (1.0 - short_alpha) * short_sum ;

         // Compute the normalizing factor, then multiply it by atr to get scaling factor

         diff = 0.5 * (long_length - 1.0) ;     // Center of long block
         diff -= 0.5 * (short_length - 1.0) ;   // Minus center of short block for random walk variance
         denom = sqrt ( fabs(diff) ) ;          // Absolute value should never be needed if careful caller
         k = long_length + n_to_smooth ;
         if (k > icase)                         // ATR requires case at least equal to length
            k = icase ;                         // Which will not be true at the beginning
         denom *= atr ( 0 , icase , k , open , high , low , close ) ;

         // These are the two scalings.  To skip scaling, just use short_sum - long_sum.
         output[icase] = (short_sum - long_sum) / (denom + 1.e-15) ;
         output[icase] = 100.0 * normal_cdf ( 1.0 * output[icase] ) - 50.0 ;
         } // For all cases

      // Smooth and compute differences if requested
      if (n_to_smooth > 1) {
         alpha = 2.0 / (n_to_smooth + 1.0) ;
         smoothed = output[0] ;
         for (icase=1 ; icase<n ; icase++) {
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed ;
            output[icase] -= smoothed ;
            } // For all cases
         } // If n_to_smooth > 1
      } // VAR_MACD


//*************************************************************

*/