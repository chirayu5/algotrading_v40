#include "../utils/io.cpp"
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

static pybind11::array_t<double> stochastic_rsi_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &prices,
    int rsi_lookback, int stoch_lookback, int n_to_smooth) {
  if (rsi_lookback < 1)
    throw std::runtime_error("rsi_lookback must be positive");
  if (stoch_lookback < 1)
    throw std::runtime_error("stoch_lookback must be positive");

  const auto [prices_, n] = atv40::io::get_input_ptr<double>(prices, "prices");
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int lookback = rsi_lookback;    // RSI lookback
  int lookback2 = stoch_lookback; // Stochastic lookback
  // n_to_smooth  // Lookback for final exponential smoothing

  int front_bad =
      lookback + lookback2 - 1; // Number of undefined values at start
  int back_bad = 0;             // Number of undefined values at end
  int icase, j;
  double upsum, dnsum, diff, min_val, max_val, alpha, smoothed;
  std::vector<double> work1(n, 0.0);
  std::vector<double> work2(n, 0.0);

  for (icase = 0; icase < std::min(front_bad, static_cast<int>(n)); icase++)
    output[icase] =
        std::numeric_limits<double>::quiet_NaN(); // Set undefined values to NaN

  // Compute RSI and save it in work1

  upsum = dnsum = 1.e-60; // Initialize
  for (icase = 1; icase < std::min(lookback, static_cast<int>(n)); icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0)
      upsum += diff;
    else
      dnsum -= diff;
  }
  upsum /= (lookback - 1);
  dnsum /= (lookback - 1);

  // Initialization is done.  Compute RSI.

  for (icase = lookback; icase < n; icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0) {
      upsum = ((lookback - 1) * upsum + diff) / lookback;
      dnsum *= (lookback - 1.0) / lookback;
    } else {
      dnsum = ((lookback - 1) * dnsum - diff) / lookback;
      upsum *= (lookback - 1.0) / lookback;
    }
    work1[icase] = 100.0 * upsum / (upsum + dnsum);
  } // For all cases being computed

  // RSI is computed in work1.  Now do stochastic.

  for (icase = front_bad; icase < n; icase++) {
    min_val = 1.e60;
    max_val = -1.e60;
    for (j = 0; j < lookback2; j++) {
      if (work1[icase - j] > max_val)
        max_val = work1[icase - j];
      if (work1[icase - j] < min_val)
        min_val = work1[icase - j];
    }

    output[icase] =
        100.0 * (work1[icase] - min_val) / (max_val - min_val + 1.e-60);
  } // For icase, computing stochastic

  // Smooth if requested
  if (n_to_smooth > 1) {
    alpha = 2.0 / (n_to_smooth + 1.0);
    smoothed = output[front_bad];
    for (icase = front_bad + 1; icase < n; icase++) {
      smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed;
      output[icase] = smoothed;
    } // For all cases
  } // If n_to_smooth

  return out;
}

void register_stochastic_rsi(pybind11::module_ &m) {
  m.def("stochastic_rsi_cpp", &stochastic_rsi_cpp, pybind11::arg("prices"),
        pybind11::arg("rsi_lookback"), pybind11::arg("stoch_lookback"),
        pybind11::arg("n_to_smooth"), "Compute the Stochastic RSI indicator.");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  STOCHASTIC RSI
//*************************************************************

   else if (var_num == VAR_STOCHASTIC_RSI) {
      lookback = (int) (param1 + 0.5) ;      // RSI lookback
      lookback2 = (int) (param2 + 0.5) ;     // Stochastic lookback
      n_to_smooth = (int) (param3 + 0.5) ;   // Lookback for final exponential smoothing

      front_bad = lookback + lookback2 - 1 ; // Number of undefined values at start
      back_bad = 0 ;                         // Number of undefined values at end

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 50.0 ;   // Set undefined values to neutral value

      // Compute RSI and save it in work1

      upsum = dnsum = 1.e-60 ;   // Initialize
      for (icase=1 ; icase<lookback ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0)
            upsum += diff ;
         else
            dnsum -= diff ;
         }
      upsum /= (lookback - 1) ;
      dnsum /= (lookback - 1) ;

      // Initialization is done.  Compute RSI.

      for (icase=lookback ; icase<n ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0) {
            upsum = ((lookback - 1) * upsum + diff) / lookback ;
            dnsum *= (lookback - 1.0) / lookback ;
            }
         else {
            dnsum = ((lookback - 1) * dnsum - diff) / lookback ;
            upsum *= (lookback - 1.0) / lookback ;
            }
         work1[icase] = 100.0 * upsum / (upsum + dnsum) ;
         } // For all cases being computed

      // RSI is computed in work1.  Now do stochastic.

      for (icase=front_bad ; icase<n ; icase++) {
         min_val = 1.e60 ;
         max_val = -1.e60 ;
         for (j=0 ; j<lookback2 ; j++) {
            if (work1[icase-j] > max_val)
               max_val = work1[icase-j] ;
            if (work1[icase-j] < min_val)
               min_val = work1[icase-j] ;
            }

         output[icase] = 100.0 * (work1[icase] - min_val) / (max_val - min_val + 1.e-60) ;
         } // For icase, computing stochastic

      // Smooth if requested
      if (n_to_smooth > 1) {
         alpha = 2.0 / (n_to_smooth + 1.0) ;
         smoothed = output[front_bad] ;
         for (icase=front_bad+1 ; icase<n ; icase++) {
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed ;
            output[icase] = smoothed ;
            } // For all cases
         } // If n_to_smooth

      } // VAR_STOCHASTIC_RSI


//*************************************************************

*/