#ifndef LMLIB_PACKET_HPP
#define LMLIB_PACKET_HPP

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include "./LMBase.hpp"
#include "./Dense.hpp"
#include "./Exp.hpp"

namespace lmlib {
namespace packet {

enum PacketArch {
  kPlain,
  kSSE2,
};

#if LMLIB_USE_SSE
#define LMLIB_DEFAULT_PACKEL ::lmlib::packet::kSSE2
#else
#define LMLIB_DEFAULT_PACKEL ::lmlib::packet::kPlain
#endif

template <typename DType, PacketArch Arch = LMLIB_DEFAULT_PACKEL> struct Packet;

template <PacketArch Arch> struct AlignBytes {
  static const index_t value = 4;
};

} // namespace packet
} // namespace lmlib

namespace lmlib {
namespace packet {

inline void *AlignedMallocPitch(size_t *out_pitch, size_t lspace,
                                size_t num_line);

} // namespace packet
} // namespace lmlib

#endif // LMLIB_PACKET_HPP