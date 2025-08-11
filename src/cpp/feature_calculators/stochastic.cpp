#include "../utils/io.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double> stochastic_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    int lookback, int n_to_smooth) {

  const auto [close_, n1] = atv40::io::get_input_ptr<double>(close, "close");
  const auto [high_, n2] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n3] = atv40::io::get_input_ptr<double>(low, "low");
  if (n1 != n2 || n1 != n3) {
    throw std::runtime_error("close, high, and low must have the same length");
  }
  const std::size_t n = n1;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int front_bad, icase, j;
  double min_val, max_val, sto_0, sto_1, sto_2;

  front_bad = lookback - 1; // Number of undefined values at start

  for (icase = 0; icase < std::min(front_bad, static_cast<int>(n)); icase++)
    output[icase] =
        std::numeric_limits<double>::quiet_NaN(); // Set undefined values to NaN

  for (icase = front_bad; icase < n; icase++) {
    min_val = 1.e60;
    max_val = -1.e60;
    for (j = 0; j < lookback; j++) {
      if (high_[icase - j] > max_val)
        max_val = high_[icase - j];
      if (low_[icase - j] < min_val)
        min_val = low_[icase - j];
    }

    sto_0 = (close_[icase] - min_val) / (max_val - min_val + 1.e-60);

    // n_to_smooth will be 0 for raw (rarely used), 1 for K, 2 for D

    if (n_to_smooth == 0)
      output[icase] = 100.0 * sto_0;

    else {
      if (icase == front_bad) {
        sto_1 = sto_0;
        output[icase] = 100.0 * sto_0;
      } else {
        sto_1 = 0.33333333 * sto_0 + 0.66666667 * sto_1;
        if (n_to_smooth == 1)
          output[icase] = 100.0 * sto_1;
        else {
          if (icase == front_bad + 1) {
            sto_2 = sto_1;
            output[icase] = 100.0 * sto_1;
          } else {
            sto_2 = 0.33333333 * sto_1 + 0.66666667 * sto_2;
            output[icase] = 100.0 * sto_2;
          }
        }
      }
    }
  } // For all cases being computed
  return out;
}

void register_stochastic(pybind11::module_ &m) {
  m.def("stochastic_cpp", &stochastic_cpp, pybind11::arg("close"),
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("lookback"),
        pybind11::arg("n_to_smooth"),
        "Compute the stochastic indicator for a sequence of prices");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL
//*************************************************************
//  STOCHASTIC Raw, K, and D
//*************************************************************

   else if (var_num == VAR_STOCHASTIC) {
      lookback = (int) (param1 + 0.5) ;    // Lookback includes current bar
      n_to_smooth = (int) (param2 + 0.5) ; // Times to smooth; 1 for K, 2 for D
      front_bad = lookback - 1 ;           // Number of undefined values at start
      back_bad = 0 ;                       // Number of undefined values at end

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 50.0 ;   // Set undefined values to neutral value

      for (icase=front_bad ; icase<n ; icase++) {
         min_val = 1.e60 ;
         max_val = -1.e60 ;
         for (j=0 ; j<lookback ; j++) {
            if (high[icase-j] > max_val)
               max_val = high[icase-j] ;
            if (low[icase-j] < min_val)
               min_val = low[icase-j] ;
            }

         sto_0 = (close[icase] - min_val) / (max_val - min_val + 1.e-60) ;

         // n_to_smooth will be 0 for raw (rarely used), 1 for K, 2 for D

         if (n_to_smooth == 0)
            output[icase] = 100.0 * sto_0 ;

         else {
            if (icase == front_bad) {
               sto_1 = sto_0 ;
               output[icase] = 100.0 * sto_0 ;
               }
            else {
               sto_1 = 0.33333333 * sto_0 + 0.66666667 * sto_1 ;
               if (n_to_smooth == 1)
                  output[icase] = 100.0 * sto_1 ;
               else {
                  if (icase == front_bad + 1) {
                     sto_2 = sto_1 ;
                     output[icase] = 100.0 * sto_1 ;
                     }
                  else {
                     sto_2 = 0.33333333 * sto_1 + 0.66666667 * sto_2 ;
                     output[icase] = 100.0 * sto_2 ;
                     }
                  }
               }
            }
         } // For all cases being computed
      } // VAR_STOCHASTIC

//*************************************************************
*/