#include <cstddef>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace atv40::io {

// -----------------------------------------------------------------------------
// get_input_ptr : zero-copy view onto a 1-D NumPy array
// -----------------------------------------------------------------------------
template <typename T>
inline std::pair<const T *, std::size_t>
get_input_ptr(const pybind11::array_t<T, pybind11::array::c_style |
                                             pybind11::array::forcecast> &arr,
              const char *name = "array") {
  auto buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error(std::string{name} + " must be a 1-D array");
  }
  return {static_cast<const T *>(buf.ptr),
          static_cast<std::size_t>(buf.shape[0])};
}

// -----------------------------------------------------------------------------
// make_output_array : create writable NumPy array + handy data pointer
// -----------------------------------------------------------------------------
template <typename T>
inline std::pair<pybind11::array_t<T>, T *> make_output_array(std::size_t n) {
  pybind11::array_t<T> out(n); // owns memory
  auto buf = out.request();    // borrow pointer (no extra copy)
  return {out, static_cast<T *>(buf.ptr)};
}

} // namespace atv40::io