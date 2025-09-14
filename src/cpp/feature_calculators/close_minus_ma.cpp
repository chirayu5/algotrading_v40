#include "../utils/features.hpp"
#include "../utils/io.cpp"
#include "../utils/stats.hpp"
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double> close_minus_ma_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close,
    int lookback, int atr_length) {

  const auto [high_, n2] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n3] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n4] = atv40::io::get_input_ptr<double>(close, "close");
  if (n2 != n3 || n2 != n4) {
    throw std::runtime_error("high, low, and close must have the same length");
  }
  const int n = n2;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int icase, k;
  double sum, denom;

  int front_bad = (lookback > atr_length)
                      ? lookback
                      : atr_length; // Number of undefined values at start
  if (front_bad >= n) {
    for (int icase = 0; icase < n; icase++) {
      output[icase] = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
  }

  for (icase = 0; icase < front_bad; icase++) {
    output[icase] = std::numeric_limits<double>::quiet_NaN();
  }

  for (icase = front_bad; icase < n; icase++) {
    sum = 0.0;
    for (k = icase - lookback; k < icase; k++)
      sum += log(close_[k]);
    sum /= lookback;
    denom = atr_cpp(1, icase, atr_length, high, low, close); // use_log=1 (true)
    if (denom > 0.0) {
      denom *= sqrt(lookback + 1.0);
      output[icase] = (log(close_[icase]) - sum) / denom;
      output[icase] =
          100.0 * normal_cdf_cpp(1.0 * output[icase]) -
          50.0; // Increase 1.0 for more compression, decrease for less
    } else
      output[icase] = 0.0;
  }

  return out;
}

void register_close_minus_ma(pybind11::module_ &m) {
  m.def("close_minus_ma_cpp", &close_minus_ma_cpp, pybind11::arg("high"),
        pybind11::arg("low"), pybind11::arg("close"), pybind11::arg("lookback"),
        pybind11::arg("atr_length"),
        "Compute the close minus moving average indicator.");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  CLOSE_MINUS_MA
//*************************************************************

   else if (var_num == VAR_CLOSE_MINUS_MA) {
      lookback = (int) (param1 + 0.5 ) ;
      atr_length = (int) (param2 + 0.5 ) ;
      front_bad = (lookback > atr_length) ? lookback : atr_length ; // Number of undefined values at start
      back_bad = 0 ;                                                // Number of undefined values at end
      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 0.0 ;   // Set undefined values to neutral value

      for (icase=front_bad ; icase<n ; icase++) {
         sum = 0.0 ;
         for (k=icase-lookback ; k<icase ; k++)
            sum += log ( close[k] ) ;
         sum /= lookback ;
         denom = atr ( 1 , icase , atr_length , open , high , low , close ) ;
         if (denom > 0.0) {
            denom *= sqrt ( lookback + 1.0 ) ;
            output[icase] = (log ( close[icase] ) - sum)  / denom ;
            output[icase] = 100.0 * normal_cdf ( 1.0 * output[icase] ) - 50.0 ; // Increase 1.0 for more compression, decrease for less
            }
         else
            output[icase] = 0.0 ;
         }
      }

//*************************************************************

*/