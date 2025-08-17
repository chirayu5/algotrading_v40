#include "../utils/io.cpp"
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double>
adx_cpp(const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &high,
        const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &low,
        const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &close,
        int lookback) {

  const auto [high_, n1] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n2] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n3] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3) {
    throw std::runtime_error("high, low, and close must have the same length");
  }
  const int n = n1;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  int front_bad, icase;
  double DMplus, DMminus, DMSplus, DMSminus, DIplus, DIminus, term, ATR, ADX;

  front_bad = 2 * lookback - 1; // Number of undefined values at start
  if (front_bad >= n) {
    for (int icase = 0; icase < n; icase++) {
      output[icase] = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
  }

  output[0] = 0; // This totally invalid case gets a neutral value

  // Primary initialization sums DMplus, DMminus, and ATR over first lookback
  // prices This gives the first (not yet smoothed or fully valid) ADX

  DMSplus = DMSminus = ATR = 0.0;
  for (icase = 1; icase <= lookback; icase++) {
    DMplus = high_[icase] - high_[icase - 1]; // Upward move
    DMminus = low_[icase - 1] - low_[icase];  // Downward move
    if (DMplus >= DMminus)                    // Pick whichever is larger
      DMminus = 0.0;                          // and discard the smaller
    else
      DMplus = 0.0;
    if (DMplus < 0.0) // Moves cannot be negative
      DMplus = 0.0;
    if (DMminus < 0.0)
      DMminus = 0.0;
    DMSplus += DMplus;
    DMSminus += DMminus;
    term = high_[icase] - low_[icase]; // Now cumulate ATR
    if (high_[icase] - close_[icase - 1] > term)
      term = high_[icase] - close_[icase - 1];
    if (close_[icase - 1] - low_[icase] > term)
      term = close_[icase - 1] - low_[icase];
    ATR += term;
    // Officially we don't need these computations yet, but might as well put
    // them in output during warmup
    DIplus = DMSplus / (ATR + 1.e-10);
    DIminus = DMSminus / (ATR + 1.e-10);
    ADX =
        fabs(DIplus - DIminus) /
        (DIplus + DIminus + 1.e-10); // When loop is done this will be first ADX
    output[icase] = 100.0 * ADX; // Not very settled down yet, but fairly valid
  }

  // Secondary initialization gets the next lookback-1 values,
  // adding them into ADX so we can get a simple average.
  // But from here on we use exponential smoothing for DMSplus, DMSminus, and
  // ATR

  for (icase = lookback + 1; icase < 2 * lookback; icase++) {
    DMplus = high_[icase] - high_[icase - 1]; // Upward move
    DMminus = low_[icase - 1] - low_[icase];  // Downward move
    if (DMplus >= DMminus)                    // Pick whichever is larger
      DMminus = 0.0;                          // and discard the smaller
    else
      DMplus = 0.0;
    if (DMplus < 0.0) // Moves cannot be negative
      DMplus = 0.0;
    if (DMminus < 0.0)
      DMminus = 0.0;
    DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus;
    DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus;
    term = high_[icase] - low_[icase]; // Now cumulate ATR
    if (high_[icase] - close_[icase - 1] > term)
      term = high_[icase] - close_[icase - 1];
    if (close_[icase - 1] - low_[icase] > term)
      term = close_[icase - 1] - low_[icase];
    ATR = (lookback - 1.0) / lookback * ATR + term;
    DIplus = DMSplus / (ATR + 1.e-10);
    DIminus = DMSminus / (ATR + 1.e-10);
    ADX += fabs(DIplus - DIminus) /
           (DIplus + DIminus + 1.e-10); // Cumulate for simple average
    output[icase] =
        100.0 * ADX /
        (icase - lookback + 1); // Not very settled down yet, but fairly valid
  }
  ADX /= lookback; // First valid value; we put it in output above

  // Whew, that was a chore!  And sadly, just using exponential smoothing
  // all along would have not only been a LOT simpler but probably even better
  // due to the faster response of exponential smoothing.
  // But I felt it important to follow Wilder's original algorithm.
  // In any case, it's a moot point because this was just initialization.
  // Either method would give similar results soon after this initialization.
  // From here on we use exponential smoothing for everything.

  for (icase = 2 * lookback; icase < n; icase++) {
    DMplus = high_[icase] - high_[icase - 1]; // Upward move
    DMminus = low_[icase - 1] - low_[icase];  // Downward move
    if (DMplus >= DMminus)                    // Pick whichever is larger
      DMminus = 0.0;                          // and discard the smaller
    else
      DMplus = 0.0;
    if (DMplus < 0.0) // Moves cannot be negative
      DMplus = 0.0;
    if (DMminus < 0.0)
      DMminus = 0.0;
    DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus;
    DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus;
    term = high_[icase] - low_[icase]; // Now cumulate ATR
    if (high_[icase] - close_[icase - 1] > term)
      term = high_[icase] - close_[icase - 1];
    if (close_[icase - 1] - low_[icase] > term)
      term = close_[icase - 1] - low_[icase];
    ATR = (lookback - 1.0) / lookback * ATR + term;
    DIplus = DMSplus / (ATR + 1.e-10);
    DIminus = DMSminus / (ATR + 1.e-10);

    term = fabs(DIplus - DIminus) / (DIplus + DIminus + 1.e-10); // This ADX
    ADX = (lookback - 1.0) / lookback * ADX + term / lookback;

    output[icase] = 100.0 * ADX;
  } // For all remaining cases, which are valid

  for (icase = 0; icase < front_bad; icase++) {
    output[icase] = std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

void register_adx(pybind11::module_ &m) {
  m.def("adx_cpp", &adx_cpp, pybind11::arg("high"), pybind11::arg("low"),
        pybind11::arg("close"), pybind11::arg("lookback"),
        "Compute the ADX indicator.");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  ADX
//*************************************************************

   else if (var_num == VAR_ADX) {
      lookback = (int) (param1 + 0.5) ;
      front_bad = 2 * lookback - 1 ; // Number of undefined values at start
      back_bad = 0 ;                 // Number of undefined values at end

      output[0] = 0 ;  // This totally invalid case gets a neutral value

      // Primary initialization sums DMplus, DMminus, and ATR over first lookback prices
      // This gives the first (not yet smoothed or fully valid) ADX

      DMSplus = DMSminus = ATR = 0.0 ;
      for (icase=1 ; icase<=lookback ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus += DMplus ;
         DMSminus += DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR += term ;
         // Officially we don't need these computations yet, but might as well put them in output during warmup
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
         ADX = fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ; // When loop is done this will be first ADX
         output[icase] = 100.0 * ADX ;  // Not very settled down yet, but fairly valid
#if 0
         char msg[256] ;
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf DI+=%8.3lf DI-=%8.3lf ADX=%.4lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR, DIplus, DIminus, ADX ) ;
         MEMTEXT ( msg ) ;
#endif
         }

      // Secondary initialization gets the next lookback-1 values,
      // adding them into ADX so we can get a simple average.
      // But from here on we use exponential smoothing for DMSplus, DMSminus, and ATR

      for (icase=lookback+1 ; icase<2*lookback ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus ;
         DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR = (lookback - 1.0) / lookback * ATR + term ;
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
         ADX += fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ;    // Cumulate for simple average
         output[icase] = 100.0 * ADX / (icase-lookback+1) ;  // Not very settled down yet, but fairly valid
#if 0
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf DI+=%8.3lf DI-=%8.3lf ADX=%.4lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR, DIplus, DIminus, output[icase] ) ;
         MEMTEXT ( msg ) ;
#endif
         }
      ADX /= lookback ;  // First valid value; we put it in output above

      // Whew, that was a chore!  And sadly, just using exponential smoothing
      // all along would have not only been a LOT simpler but probably even better
      // due to the faster response of exponential smoothing.
      // But I felt it important to follow Wilder's original algorithm.
      // In any case, it's a moot point because this was just initialization.
      // Either method would give similar results soon after this initialization.
      // From here on we use exponential smoothing for everything.

      for (icase=2*lookback ; icase<n ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus ;
         DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR = (lookback - 1.0) / lookback * ATR + term ;
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
#if 0
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR ) ;
         MEMTEXT ( msg ) ;
#endif
         term = fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ; // This ADX
         ADX = (lookback - 1.0) / lookback * ADX + term / lookback ;
#if 0
         sprintf_s ( msg, "   DI+=%8.3lf DI-=%8.3lf ADX=%.4lf", DIplus, DIminus, ADX ) ;
         MEMTEXT ( msg ) ;
#endif
         output[icase] = 100.0 * ADX ;
         }  // For all remaining cases, which are valid
      } // VAR_ADX


//*************************************************************

*/