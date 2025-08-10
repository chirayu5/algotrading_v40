#include "../utils/io.cpp"
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double>
rsi_cpp(const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &prices,
        int lookback) {
  if (lookback < 1) {
    throw std::runtime_error("lookback must be positive");
  }

  const auto [prices_, n] = atv40::io::get_input_ptr<double>(prices, "prices");
  auto [out, output] = atv40::io::make_output_array<double>(n);

  // Implement RSI computation logic (unaltered from reference)
  int front_bad = lookback; // Number of undefined values at start
  int back_bad = 0;         // Number of undefined values at end (unused)
  double upsum, dnsum, diff;
  std::size_t icase;

  // Edge-case: series shorter than lookback => all NaN
  if (n <= static_cast<std::size_t>(lookback)) {
    for (icase = 0; icase < n; ++icase) {
      output[icase] = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
  }

  // Set undefined initial values to NaN
  for (icase = 0; icase < static_cast<std::size_t>(front_bad); ++icase) {
    output[icase] = std::numeric_limits<double>::quiet_NaN();
  }

  // Initialize
  upsum = dnsum = 1.e-60;
  for (icase = 1; icase < static_cast<std::size_t>(front_bad); ++icase) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0) {
      upsum += diff;
    } else {
      dnsum -= diff;
    }
  }
  upsum /= (lookback - 1);
  dnsum /= (lookback - 1);

  // Compute RSI for remaining points
  for (icase = front_bad; icase < n; ++icase) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0) {
      upsum = ((lookback - 1) * upsum + diff) / lookback;
      dnsum *= (lookback - 1.0) / lookback;
    } else {
      dnsum = ((lookback - 1) * dnsum - diff) / lookback;
      upsum *= (lookback - 1.0) / lookback;
    }
    output[icase] = 100.0 * upsum / (upsum + dnsum);
  }

  return out;
}

void register_rsi(pybind11::module_ &m) {
  m.def("rsi_cpp", &rsi_cpp, pybind11::arg("prices"), pybind11::arg("lookback"),
        "Compute the RSI indicator for a sequence of prices");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  RSI
//*************************************************************

   if (var_num == VAR_RSI) {
      lookback = (int) (param1 + 0.5) ;
      front_bad = lookback ;             // Number of undefined values at start
      back_bad = 0 ;                     // Number of undefined values at end

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 50.0 ;   // Set undefined values to neutral value

      // Initialize
      upsum = dnsum = 1.e-60 ;
      for (icase=1 ; icase<front_bad ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0)
            upsum += diff ;
         else
            dnsum -= diff ;
         }
      upsum /= (lookback - 1) ;
      dnsum /= (lookback - 1) ;

      // Initialization is done.  Start computing.

      for (icase=front_bad ; icase<n ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0) {
            upsum = ((lookback - 1) * upsum + diff) / lookback ;
            dnsum *= (lookback - 1.0) / lookback ;
            }
         else {
            dnsum = ((lookback - 1) * dnsum - diff) / lookback ;
            upsum *= (lookback - 1.0) / lookback ;
            }
         output[icase] = 100.0 * upsum / (upsum + dnsum) ;
         } // For all cases being computed
      } // VAR_RSI

//*************************************************************

*/