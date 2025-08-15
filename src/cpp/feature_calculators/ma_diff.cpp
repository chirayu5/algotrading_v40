#include "../utils/features.hpp"
#include "../utils/io.cpp"
#include "../utils/stats.hpp"
#include <cassert>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double> ma_diff_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &open,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close,
    int short_length, int long_length, int lag) {

  const auto [open_, n1] = atv40::io::get_input_ptr<double>(open, "open");
  const auto [high_, n2] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n3] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n4] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3 || n1 != n4) {
    throw std::runtime_error(
        "open, close, high, and low must have the same length");
  }
  const int n = n1;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int front_bad =
      long_length + lag; // ATR will need one extra case for prior close
  if (front_bad > n)
    front_bad = n;

  // Set undefined initial values to NaN
  for (int icase = 0; icase < front_bad; ++icase) {
    output[icase] = std::numeric_limits<double>::quiet_NaN();
  }

  int back_bad = 0;
  double long_sum, short_sum, diff, denom;
  int k, icase;

  for (icase = front_bad; icase < n; icase++) {
    long_sum = short_sum = 0.0;

    // Compute long-term and short-term moving averages
    for (k = icase - long_length + 1; k <= icase; k++)
      long_sum += close_[k - lag];
    long_sum /= long_length;

    for (k = icase - short_length + 1; k <= icase; k++)
      short_sum += close_[k];
    short_sum /= short_length;

    // Compute the normalizing factor, then multiply it by atr to get scaling
    // factor

    diff = 0.5 * (long_length - 1.0) + lag; // Center of long block
    diff -= 0.5 * (short_length -
                   1.0); // Minus center of short block for random walk variance
    denom = sqrt(
        fabs(diff)); // Absolute value should never be needed if careful caller
    denom *= atr_cpp(0, icase, long_length + lag, high, low, close);

    output[icase] = (short_sum - long_sum) / (denom + 1.e-60);
    output[icase] = 100.0 * normal_cdf_cpp(1.5 * output[icase]) - 50.0;
  } // For all cases

  return out;
}

void register_ma_diff(pybind11::module_ &m) {
  m.def("ma_diff_cpp", &ma_diff_cpp, pybind11::arg("open"),
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("close"),
        pybind11::arg("short_length"), pybind11::arg("long_length"),
        pybind11::arg("lag"),
        "Compute the Moving-Average Difference indicator");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  Moving Average Difference
//*************************************************************
            
   else if (var_num == VAR_MA_DIFF) {
      short_length = (int) (param1 + 0.5) ;
      long_length = (int) (param2 + 0.5) ;
      lag = (int) (param3 + 0.5) ;
      front_bad = long_length + lag ;  // ATR will need one extra case for prior close
      if (front_bad > n)
         front_bad = n ;
      back_bad = 0 ;

      for (icase=front_bad ; icase<n ; icase++) {
         long_sum = short_sum = 0.0 ;

         // Compute long-term and short-term moving averages
         for (k=icase-long_length+1 ; k<=icase ; k++)
            long_sum += close[k-lag] ;
         long_sum /= long_length ;

         for (k=icase-short_length+1 ; k<=icase ; k++)
            short_sum += close[k] ;
         short_sum /= short_length ;

         // Compute the normalizing factor, then multiply it by atr to get scaling factor

         diff = 0.5 * (long_length - 1.0) + lag ; // Center of long block
         diff -= 0.5 * (short_length - 1.0) ;     // Minus center of short block for random walk variance
         denom = sqrt ( fabs(diff) ) ;            // Absolute value should never be needed if careful caller
         denom *= atr ( 0 , icase , long_length + lag , open , high , low , close ) ;

         output[icase] = (short_sum - long_sum) / (denom + 1.e-60) ;
         output[icase] = 100.0 * normal_cdf ( 1.5 * output[icase] ) - 50.0 ;
         } // For all cases
      } // VAR_MA_DIFF


//*************************************************************

*/
