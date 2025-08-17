#include "../utils/features.hpp"
#include "../utils/io.cpp"
#include "../utils/stats.hpp"
#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

static pybind11::array_t<double> price_intensity_cpp(
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &open,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &high,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &low,
    const pybind11::array_t<double, pybind11::array::c_style |
                                        pybind11::array::forcecast> &close,
    int n_to_smooth) {
  const auto [open_, n1] = atv40::io::get_input_ptr<double>(open, "open");
  const auto [high_, n2] = atv40::io::get_input_ptr<double>(high, "high");
  const auto [low_, n3] = atv40::io::get_input_ptr<double>(low, "low");
  const auto [close_, n4] = atv40::io::get_input_ptr<double>(close, "close");
  if (n1 != n2 || n1 != n3 || n1 != n4) {
    throw std::runtime_error(
        "open, high, low, and close must have the same length");
  }
  const int n = n1;
  auto [out, output] = atv40::io::make_output_array<double>(n);

  if (n_to_smooth < 1)
    throw std::runtime_error("n_to_smooth must be positive");

  int icase;
  double denom, alpha, smoothed;

  // no values are bad

  // The first bar has no prior bar
  denom = high_[0] - low_[0];
  if (denom < 1.e-60)
    denom = 1.e-60;
  output[0] = (close_[0] - open_[0]) / denom;

  // Compute raw values
  for (icase = 1; icase < n; icase++) {
    denom = high_[icase] - low_[icase];
    if (high_[icase] - close_[icase - 1] > denom)
      denom = high_[icase] - close_[icase - 1];
    if (close_[icase - 1] - low_[icase] > denom)
      denom = close_[icase - 1] - low_[icase];
    if (denom < 1.e-60)
      denom = 1.e-60;
    output[icase] = (close_[icase] - open_[icase]) / denom;
  } // For all cases being computed

  // Smooth if requested
  if (n_to_smooth > 1) {
    alpha = 2.0 / (n_to_smooth + 1.0);
    smoothed = output[0];
    for (icase = 1; icase < n; icase++) {
      smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed;
      output[icase] = smoothed;
    }
  } // If n_to_smooth

  // Final transformation and mild compression
  for (icase = 0; icase < n; icase++)
    output[icase] = 100.0 * normal_cdf_cpp(0.8 * sqrt((double)n_to_smooth) *
                                           output[icase]) -
                    50.0;

  return out;
}

void register_price_intensity(pybind11::module_ &m) {
  m.def("price_intensity_cpp", &price_intensity_cpp, pybind11::arg("open"),
        pybind11::arg("high"), pybind11::arg("low"), pybind11::arg("close"),
        pybind11::arg("n_to_smooth"), "Compute the Price Intensity indicator.");
}

// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  PRICE INTENSITY
//*************************************************************

   else if (var_num == VAR_PRICE_INTENSITY) {
      n_to_smooth = (int) (param1 + 0.5) ;
      if (n_to_smooth < 1)
         n_to_smooth = 1 ;
      front_bad = 0 ;          // Number of undefined values at start
      back_bad = 0 ;           // Number of undefined values at end

      // The first bar has no prior bar
      denom = high[0] - low[0] ;
      if (denom < 1.e-60)
         denom = 1.e-60 ;
      output[0] = (close[0] - open[0]) / denom ;

      // Compute raw values
      for (icase=1 ; icase<n ; icase++) {
         denom = high[icase] - low[icase] ;
         if (high[icase] - close[icase-1] > denom)
            denom = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > denom)
            denom = close[icase-1] - low[icase] ;
         if (denom < 1.e-60)
            denom = 1.e-60 ;
         output[icase] = (close[icase] - open[icase]) / denom ;
         } // For all cases being computed

      // Smooth if requested
      if (n_to_smooth > 1) {
         alpha = 2.0 / (n_to_smooth + 1.0) ;
         smoothed = output[0] ;
         for (icase=1 ; icase<n ; icase++) {
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed ;
            output[icase] = smoothed ;
            }
         } // If n_to_smooth

      // Final transformation and mild compression
      for (icase=0 ; icase<n ; icase++)
         output[icase] = 100.0 * normal_cdf ( 0.8 * sqrt((double) n_to_smooth) * output[icase] ) - 50.0 ;

      } // VAR_PRICE_INTENSITY


//*************************************************************

*/