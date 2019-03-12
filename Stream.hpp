#ifndef LMLIB_STREAM_HPP_
#define LMLIB_STREAM_HPP_

#include "LMBase.hpp"

namespace lmlib {
struct Stream {
  inline void Wait() {}
  inline bool CheckIdle() { return true; }
};
} // namespace lmlib

#endif // LMLIB_STREAM_HPP_
