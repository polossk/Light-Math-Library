#ifndef LIGHT_MATH_LIBRARY_STREAM_HPP
#define LIGHT_MATH_LIBRARY_STREAM_HPP

#include "base.hpp"

namespace lmlib {
struct Stream {
  inline void Wait() {}
  inline bool CheckIdle() { return true; }
};
} // namespace lmlib
#endif // LIGHT_MATH_LIBRARY_STREAM_HPP

