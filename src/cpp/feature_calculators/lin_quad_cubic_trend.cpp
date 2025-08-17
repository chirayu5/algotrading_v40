#include "../utils/features.hpp"
#include "../utils/io.cpp"
#include "../utils/stats.hpp"
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/******************************************************************************/
/*                                                                            */
/*  LEGENDRE - Compute coeficients of discrete Legendre polynomial            */
/*                                                                            */
/******************************************************************************/

/*
--------------------------------------------------------------------------------

   Compute first, second, and third-order normalized orthogonal coefs
   for n data points

--------------------------------------------------------------------------------
*/

void legendre_3(int n, double *c1, double *c2, double *c3) {
  int i;
  double sum, mean, proj;

  /*
     Compute c1
  */

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c1[i] = 2.0 * i / (n - 1.0) - 1.0;
    sum += c1[i] * c1[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c1[i] /= sum;

  /*
     Compute c2
  */

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c2[i] = c1[i] * c1[i];
    sum += c2[i];
  }

  mean = sum / n; // Center it and normalize to unit length

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c2[i] -= mean;
    sum += c2[i] * c2[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c2[i] /= sum;

  /*
     Compute c3
  */

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c3[i] = c1[i] * c1[i] * c1[i];
    sum += c3[i];
  }

  mean = sum / n; // Center it and normalize to unit length
                  // Theoretically it is already centered but this
                  // tweaks in case of tiny fpt issues

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c3[i] -= mean;
    sum += c3[i] * c3[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c3[i] /= sum;

  // Remove the projection of c1

  proj = 0.0;
  for (i = 0; i < n; i++)
    proj += c1[i] * c3[i];

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c3[i] -= proj * c1[i];
    sum += c3[i] * c3[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c3[i] /= sum;
}

static pybind11::array_t<double> lin_quad_cubic_trend_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close,
    int poly_degree, int lookback, int atr_length) {
  const auto [high_, n1] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n2] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n3] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3) {
    throw std::runtime_error("high, low, and close must have the same length");
  }
  if ((poly_degree != 1) && (poly_degree != 2) && (poly_degree != 3)) {
    throw std::runtime_error("poly_degree must be 1, 2, or 3");
  }
  const int n = n1;
  int front_bad;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  front_bad = ((lookback - 1) > atr_length) ? (lookback - 1) : atr_length;
  if (front_bad > n)
    front_bad = n;

  double *work1 = new double[lookback];
  double *work2 = new double[lookback];
  double *work3 = new double[lookback];

  legendre_3(lookback, work1, work2,
             work3); // Compute all 3 even though we need just 1.  Fast.
  int icase, k;
  double dot_prod, mean, price, denom, yss, rsq, diff, pred, *dptr;

  for (icase = 0; icase < front_bad; icase++)
    output[icase] =
        std::numeric_limits<double>::quiet_NaN(); // Set undefined values to
                                                  // neutral value

  for (icase = front_bad; icase < n; icase++) {
    // Choose the correct set of Legendre coefficients
    if (poly_degree == 1) { // var_num == VAR_LINEAR_TREND
      dptr = work1;
    } else if (poly_degree == 2) { // var_num == VAR_QUADRATIC_TREND
      dptr = work2;
    } else if (poly_degree == 3) { // var_num == VAR_CUBIC_TREND
      dptr = work3;
    }

    // The regression coefficient (in dot_prod) is the dot product of the
    // log prices with the Legendre polynomial coefficients.

    dot_prod = 0.0;
    mean = 0.0; // We need this for rsq
    for (k = icase - lookback + 1; k <= icase;
         k++) { // The trend lookback window
      price = log(close_[k]);
      mean += price;
      dot_prod += price * *dptr++; // Cumulate dot product
    }
    mean /= lookback;
    dptr -= lookback; // Reset coefs pointer to start for finding rsq

    // Dot_prod is regression coef (log price change per unit X change)
    // Total X change over window is 1 - (-1) = 2 (domain of Legendre
    // polynomial) So dot_prod * 2 is the fitted change over the window Denom is
    // change over window based on ATR if all changes exactly match Legendre
    // polynomial Thus, the basic indicator (prior to rsq and compression) is
    // the ratio of the achieved fitted change to the theoretical change based
    // on ATR.

    k = lookback - 1;
    if (lookback == 2)
      k = 2;
    denom = atr_cpp(1, icase, atr_length, high, low, close) * k;
    output[icase] = dot_prod * 2.0 /
                    (denom + 1.e-60); // Change over window / window ATR path

    // At this point, output[icase] is the ratio of the observed change to the
    // theoretical change implied by ATR that follows the tested path.
    // Compute R-square for degrading the indicator if it is a poor fit

    yss = rsq = 0.0;
    for (k = icase - lookback + 1; k <= icase;
         k++) { // The trend lookback window
      price = log(close_[k]);
      diff = price - mean; // Y offset from its mean
      yss += diff * diff;  // Cumulate Y sum of squares
      pred = dot_prod *
             *dptr++; // Regression coefficient times X is predicted Y offset
      diff = diff - pred; // Y offset from mean minus predicted offset
      rsq += diff * diff; // Sum the squared error
    }
    rsq = 1.0 - rsq / (yss + 1.e-60); // Definition of R-square
    if (rsq < 0.0)                    // Should never happen
      rsq = 0.0;
    output[icase] *= rsq; // Degrade the indicator if it is a poor fit

    output[icase] = 100.0 * normal_cdf_cpp(output[icase]) -
                    50.0; // Weakly compress outliers
  } // For all cases being computed

  // free memory
  delete[] work1;
  delete[] work2;
  delete[] work3;
  return out;
}

void register_lin_quad_cubic_trend(pybind11::module_ &m) {
  m.def("lin_quad_cubic_trend_cpp", &lin_quad_cubic_trend_cpp,
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("close"),
        pybind11::arg("poly_degree"), pybind11::arg("lookback"),
        pybind11::arg("atr_length"),
        "Compute the linear, quadratic, or cubic trend");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

// --------------------------------------------------------------------------------

//    Compute first, second, and third-order normalized orthogonal coefs
//    for n data points

// --------------------------------------------------------------------------------


void legendre_3 ( int n , double *c1 , double *c2 , double *c3 )
{
   int i ;
   double sum, mean, proj ;


  //  Compute c1


   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c1[i] = 2.0 * i / (n - 1.0) - 1.0 ;
      sum += c1[i] * c1[i] ;
      }

   sum = sqrt ( sum ) ;
   for (i=0 ; i<n ; i++)
      c1[i] /= sum ;


  //  Compute c2


   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c2[i] = c1[i] * c1[i] ;
      sum += c2[i] ;
      }

   mean = sum / n ;               // Center it and normalize to unit length

   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c2[i] -= mean ;
      sum += c2[i] * c2[i] ;
      }

   sum = sqrt ( sum ) ;
   for (i=0 ; i<n ; i++)
      c2[i] /= sum ;


  //  Compute c3


   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c3[i] = c1[i] * c1[i] * c1[i] ;
      sum += c3[i] ;
      }

   mean = sum / n ;               // Center it and normalize to unit length
                                  // Theoretically it is already centered but this
                                  // tweaks in case of tiny fpt issues

   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c3[i] -= mean ;
      sum += c3[i] * c3[i] ;
      }

   sum = sqrt ( sum ) ;
   for (i=0 ; i<n ; i++)
      c3[i] /= sum ;

   // Remove the projection of c1

   proj = 0.0 ;
   for (i=0 ; i<n ; i++)
      proj += c1[i] * c3[i] ;

   sum = 0.0 ;
   for (i=0 ; i<n ; i++) {
      c3[i] -= proj * c1[i] ;
      sum += c3[i] * c3[i] ;
      }

   sum = sqrt ( sum ) ;
   for (i=0 ; i<n ; i++)
      c3[i] /= sum ;

#if 0
   double sum1, sum2, sum3, sum12, sum13, sum23 ;
   char msg[256] ;

   sum1 = sum2 = sum3 = sum12 = sum13 = sum23 = 0.0 ;

   for (i=0 ; i<n ; i++) {
      sum1 += c1[i] ;
      sum2 += c2[i] ;
      sum3 += c3[i] ;
      sum12 += c1[i] * c2[i] ;
      sum13 += c1[i] * c3[i] ;
      sum23 += c2[i] * c3[i] ;
      }

   sprintf ( msg, "----> %lf %lf %lf %lf %lf %lf", sum1, sum2, sum3, sum12, sum13, sum23 ) ;
   MEMTEXT ( msg ) ;
#endif
}

//*************************************************************
//  LINEAR/QUADRATIC/CUBIC TREND
//*************************************************************

   else if (var_num == VAR_LINEAR_TREND  ||  var_num == VAR_QUADRATIC_TREND  ||  var_num == VAR_CUBIC_TREND) {
      lookback = (int) (param1 + 0.5) ;    // Lookback for trend
      atr_length = (int) (param2 + 0.5) ;  // Lookback for ATR normalization (should greatly exceed lookback)
      front_bad = ((lookback-1) > atr_length) ? (lookback-1) : atr_length ;
      if (front_bad > n)
         front_bad = n ;
      back_bad = 0 ;

      legendre_3 ( lookback , work1 , work2 , work3 ) ;  // Compute all 3 even though we need just 1.  Fast.

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 0.0 ;   // Set undefined values to neutral value

      for (icase=front_bad ; icase<n ; icase++) {
         if (var_num == VAR_LINEAR_TREND)  // Choose the correct set of Legendre coefficients
            dptr = work1 ;
         else if (var_num == VAR_QUADRATIC_TREND)
            dptr = work2 ;
         else if (var_num == VAR_CUBIC_TREND)
            dptr = work3 ;

         // The regression coefficient (in dot_prod) is the dot product of the
         // log prices with the Legendre polynomial coefficients.

         dot_prod = 0.0 ;
         mean = 0.0 ;   // We need this for rsq
         for (k=icase-lookback+1 ; k<=icase ; k++) {  // The trend lookback window
            price = log ( close[k] ) ;
            mean += price ;
            dot_prod += price * *dptr++ ;             // Cumulate dot product
            }
         mean /= lookback ;
         dptr -= lookback ;   // Reset coefs pointer to start for finding rsq

         // Dot_prod is regression coef (log price change per unit X change)
         // Total X change over window is 1 - (-1) = 2 (domain of Legendre polynomial)
         // So dot_prod * 2 is the fitted change over the window
         // Denom is change over window based on ATR if all changes exactly match Legendre polynomial
         // Thus, the basic indicator (prior to rsq and compression) is the ratio
         // of the achieved fitted change to the theoretical change based on ATR.

         k = lookback - 1 ;
         if (lookback == 2)
            k = 2 ;
         denom = atr ( 1 , icase , atr_length , open , high , low , close ) * k ;
         output[icase] = dot_prod * 2.0 / (denom + 1.e-60) ;  // Change over window / window ATR path

         // At this point, output[icase] is the ratio of the observed change to the
         // theoretical change implied by ATR that follows the tested path.
         // Compute R-square for degrading the indicator if it is a poor fit

         yss = rsq = 0.0 ;
         for (k=icase-lookback+1 ; k<=icase ; k++) {  // The trend lookback window
            price = log ( close[k] ) ;
            diff = price - mean ;       // Y offset from its mean
            yss += diff * diff ;        // Cumulate Y sum of squares
            pred = dot_prod * *dptr++ ; // Regression coefficient times X is predicted Y offset
            diff = diff - pred ;        // Y offset from mean minus predicted offset
            rsq += diff * diff ;        // Sum the squared error 
            }
         rsq = 1.0 - rsq / (yss + 1.e-60) ;     // Definition of R-square
         if (rsq < 0.0)                 // Should never happen
            rsq = 0.0 ;
         output[icase] *= rsq ;         // Degrade the indicator if it is a poor fit

         output[icase] = 100.0 * normal_cdf ( output[icase] ) - 50.0 ;  // Weakly compress outliers
         } // For all cases being computed

      } // VAR_LINEAR/QUADRATIC/CUBIC_TREND


//*************************************************************

*/