#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

double
atr_cpp(bool use_log, int end_index, int length,
        const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &high,
        const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &low,
        const pybind11::array_t<double, pybind11::array::c_style |
                                            pybind11::array::forcecast> &close);

void register_features(pybind11::module_ &m);
