#include "io.cpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

double atr_cpp(
    bool use_log, int end_index, int length,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close) {

  const auto [high_, n1] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n2] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n3] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3) {
    throw std::runtime_error(
        "high_, low_, and close_ must have the same length");
  }
  const int n = n1;

  int i;
  double term, sum;

  assert(end_index >= length);

  // This is just a kludge to handle length=0
  if (length == 0) {
    if (use_log)
      return log(high_[end_index] / low_[end_index]);
    else
      return high_[end_index] - low_[end_index];
  }

  sum = 0.0;
  for (i = end_index - length + 1; i <= end_index; i++) {
    if (use_log) {
      term = high_[i] / low_[i];
      if (high_[i] / close_[i - 1] > term)
        term = high_[i] / close_[i - 1];
      if (close_[i - 1] / low_[i] > term)
        term = close_[i - 1] / low_[i];
      sum += log(term);
    } else {
      term = high_[i] - low_[i];
      if (high_[i] - close_[i - 1] > term)
        term = high_[i] - close_[i - 1];
      if (close_[i - 1] - low_[i] > term)
        term = close_[i - 1] - low_[i];
      sum += term;
    }
  }

  return sum / length;
}

void register_features(pybind11::module_ &m) {
  m.def("atr_cpp", &atr_cpp, pybind11::arg("use_log"),
        pybind11::arg("end_index"), pybind11::arg("length"),
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("close"),
        "Compute the ATR (Average True Range) value");
}

// clang-format off
// RAW UNTOUCHED CODE FROM ORIGINAL
/*
// --------------------------------------------------------------------------------

//    atr() - Compute historical average true range

// --------------------------------------------------------------------------------
double atr ( int use_log , int icase , int length ,
             double *open , double *high , double *low , double *close )
{
   int i ;
   double term, sum ;

   assert ( icase >= length ) ;

// This is just a kludge to handle length=0
   if (length == 0) {
      if (use_log)
         return log ( high[icase] / low[icase] ) ;
      else
         return high[icase] - low[icase] ;
      }


   sum = 0.0 ;
   for (i=icase-length+1 ; i<=icase ; i++) {
      if (use_log) {
         term = high[i] / low[i] ;
         if (high[i] / close[i-1] > term)
            term = high[i] / close[i-1] ;
         if (close[i-1] / low[i] > term)
            term = close[i-1] / low[i] ;
         sum += log ( term ) ;
         }
      else {
         term = high[i] - low[i] ;
         if (high[i] - close[i-1] > term)
            term = high[i] - close[i-1] ;
         if (close[i-1] - low[i] > term)
            term = close[i-1] - low[i] ;
         sum += term ;
         }
      }

   return sum / length ;
}

// --------------------------------------------------------------------------------
*/