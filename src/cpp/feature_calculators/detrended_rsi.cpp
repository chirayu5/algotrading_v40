
#include "../utils/io.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

static pybind11::array_t<double> detrended_rsi_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &prices,
    int short_length, int long_length, int length) {

  if (long_length < short_length) {
    throw std::runtime_error("long_length must be greater than short_length");
  }

  const auto [prices_, n] = atv40::io::get_input_ptr<double>(prices, "prices");
  auto [out, output] = atv40::io::make_output_array<double>(n);

  if (n == 0 ||
      n < std::max({short_length, long_length, long_length + length - 1})) {
    std::fill(output, output + n, std::numeric_limits<double>::quiet_NaN());
    return out;
  }

  int i, k, icase, front_bad, back_bad;
  std::vector<double> work1(n, 0.0);
  std::vector<double> work2(n, 0.0);
  double sum, diff, upsum, dnsum, xmean, ymean, xdiff, ydiff, xss, xy, coef;

  front_bad = long_length + length - 1; // Number of undefined values at start
  back_bad = 0;                         // Number of undefined values at end

  for (icase = 0; icase < front_bad; icase++)
    output[icase] =
        std::numeric_limits<double>::quiet_NaN(); // Set undefined values to NaN

  // Initialize short (detrended) RSI

  for (icase = 0; icase < short_length;
       icase++)           // Actually, these are not touched!
    work1[icase] = 1.e90; // Helps spot a bug

  upsum = dnsum = 1.e-60;
  for (icase = 1; icase < short_length; icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0)
      upsum += diff;
    else
      dnsum -= diff;
  }
  upsum /= (short_length - 1);
  dnsum /= (short_length - 1);

  for (icase = short_length; icase < n; icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0) {
      upsum = ((short_length - 1.0) * upsum + diff) / short_length;
      dnsum *= (short_length - 1.0) / short_length;
    } else {
      dnsum = ((short_length - 1.0) * dnsum - diff) / short_length;
      upsum *= (short_length - 1.0) / short_length;
    }
    work1[icase] = 100.0 * upsum / (upsum + dnsum);
    if (short_length == 2)
      work1[icase] =
          -10.0 * log(2.0 / (1 + 0.00999 * (2 * work1[icase] - 100)) - 1);
  }

  // Initialize long (detrender) RSI

  for (icase = 0; icase < long_length;
       icase++)            // Actually, these are not touched!
    work2[icase] = -1.e90; // Helps spot a bug

  upsum = dnsum = 1.e-60;
  for (icase = 1; icase < long_length; icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0)
      upsum += diff;
    else
      dnsum -= diff;
  }
  upsum /= (long_length - 1);
  dnsum /= (long_length - 1);

  for (icase = long_length; icase < n; icase++) {
    diff = prices_[icase] - prices_[icase - 1];
    if (diff > 0.0) {
      upsum = ((long_length - 1.0) * upsum + diff) / long_length;
      dnsum *= (long_length - 1.0) / long_length;
    } else {
      dnsum = ((long_length - 1.0) * dnsum - diff) / long_length;
      upsum *= (long_length - 1.0) / long_length;
    }
    work2[icase] = 100.0 * upsum / (upsum + dnsum);
  }

  // Process here

  for (icase = 0; icase < front_bad; icase++)
    output[icase] =
        std::numeric_limits<double>::quiet_NaN(); // Set undefined values to NaN

  for (icase = front_bad; icase < n; icase++) {

    xmean = ymean = 0.0;
    for (i = 0; i < length; i++) {
      k = icase - i;
      xmean += work2[k];
      ymean += work1[k];
    } // For length, cumulating means

    xmean /= length;
    ymean /= length;

    // Cumulate sum square and cross product; divide to get coef
    xss = xy = 0.0;
    for (i = 0; i < length; i++) {
      k = icase - i;
      xdiff = work2[k] - xmean;
      ydiff = work1[k] - ymean;
      xss += xdiff * xdiff;
      xy += xdiff * ydiff;
    }
    coef = xy / (xss + 1.e-60);

    // Compute difference: actual minus predicted
    xdiff = work2[icase] - xmean;
    ydiff = work1[icase] - ymean;
    output[icase] = ydiff - coef * xdiff;
  } // For all cases being computed

  return out;
}

void register_detrended_rsi(pybind11::module_ &m) {
  m.def("detrended_rsi_cpp", &detrended_rsi_cpp, pybind11::arg("prices"),
        pybind11::arg("short_length"), pybind11::arg("long_length"),
        pybind11::arg("length"),
        "Compute the detrended RSI indicator for a sequence of prices");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  DETRENDED RSI
//*************************************************************

   else if (var_num == VAR_DETRENDED_RSI) {
      short_length = (int) (param1 + 0.5) ;  // RSI being detrended
      long_length = (int) (param2 + 0.5) ;   // Detrender (greater than short_length)
      length = (int) (param3 + 0.5) ;        // Lookback for linear fit (as long as reasonably possible)
      front_bad = long_length + length - 1 ; // Number of undefined values at start
      back_bad = 0 ;                         // Number of undefined values at end

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 0.0 ;   // Set undefined values to neutral value

      // Initialize short (detrended) RSI

      for (icase=0 ; icase<short_length ; icase++) // Actually, these are not touched!
         work1[icase] = 1.e90 ;                    // Helps spot a bug

      upsum = dnsum = 1.e-60 ;
      for (icase=1 ; icase<short_length ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0)
            upsum += diff ;
         else
            dnsum -= diff ;
         }
      upsum /= (short_length - 1) ;
      dnsum /= (short_length - 1) ;

      for (icase=short_length ; icase<n ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0) {
            upsum = ((short_length - 1.0) * upsum + diff) / short_length ;
            dnsum *= (short_length - 1.0) / short_length ;
            }
         else {
            dnsum = ((short_length - 1.0) * dnsum - diff) / short_length ;
            upsum *= (short_length - 1.0) / short_length ;
            }
         work1[icase] = 100.0 * upsum / (upsum + dnsum) ;
         if (short_length == 2)
            work1[icase] = -10.0 * log ( 2.0 / (1 + 0.00999 * (2 * work1[icase] - 100)) - 1 ) ;
         }

      // Initialize long (detrender) RSI

      for (icase=0 ; icase<long_length ; icase++) // Actually, these are not touched!
         work2[icase] = -1.e90 ;    // Helps spot a bug

      upsum = dnsum = 1.e-60 ;
      for (icase=1 ; icase<long_length ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0)
            upsum += diff ;
         else
            dnsum -= diff ;
         }
      upsum /= (long_length - 1) ;
      dnsum /= (long_length - 1) ;

      for (icase=long_length ; icase<n ; icase++) {
         diff = close[icase] - close[icase-1] ;
         if (diff > 0.0) {
            upsum = ((long_length - 1.0) * upsum + diff) / long_length ;
            dnsum *= (long_length - 1.0) / long_length ;
            }
         else {
            dnsum = ((long_length - 1.0) * dnsum - diff) / long_length ;
            upsum *= (long_length - 1.0) / long_length ;
            }
         work2[icase] = 100.0 * upsum / (upsum + dnsum) ;
         }

      // Process here

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 0.0 ;   // Set undefined values to zero

      for (icase=front_bad ; icase<n ; icase++) {

         xmean = ymean = 0.0 ;
         for (i=0 ; i<length ; i++) {
            k = icase - i ;
            xmean += work2[k] ;
            ymean += work1[k] ;
            } // For length, cumulating means

         xmean /= length ;
         ymean /= length ;

         // Cumulate sum square and cross product; divide to get coef
         xss = xy = 0.0 ;
         for (i=0 ; i<length ; i++) {
            k = icase - i ;
            xdiff = work2[k] - xmean ;
            ydiff = work1[k] - ymean ;
            xss += xdiff * xdiff ;
            xy += xdiff * ydiff ;
            }
         coef = xy / (xss + 1.e-60) ;

         // Compute difference: actual minus predicted
         xdiff = work2[icase] - xmean ;
         ydiff = work1[icase] - ymean ;
         output[icase] = ydiff - coef * xdiff ;
         } // For all cases being computed
      } // VAR_DETRENDED_RSI


//*************************************************************

*/