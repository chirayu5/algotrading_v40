#include <cmath>

double normal_cdf_cpp(double z) {
  double zz = fabs(z);
  double pdf = exp(-0.5 * zz * zz) / sqrt(2.0 * 3.141592653589793);
  double t = 1.0 / (1.0 + zz * 0.2316419);
  double poly =
      ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782) *
           t +
       0.319381530) *
      t;
  return (z > 0.0) ? 1.0 - pdf * poly : pdf * poly;
}

// clang-format off
// RAW UNTOUCHED CODE FROM ORIGINAL
/*
// --------------------------------------------------------------------------------

//    Normal CDF   Accurate to 7.5 e-8

// --------------------------------------------------------------------------------

double normal_cdf ( double z )
{
   double zz = fabs ( z ) ;
   double pdf = exp ( -0.5 * zz * zz ) / sqrt ( 2.0 * 3.141592653589793 ) ;
   double t = 1.0 / (1.0 + zz * 0.2316419) ;
   double poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t -
                     0.356563782) * t + 0.319381530) * t ;
   return (z > 0.0)  ?  1.0 - pdf * poly  :  pdf * poly ;
}
*/