#ifndef LMLIB_EXTENSION_BROADCAST_HPP_
#define LMLIB_EXTENSION_BROADCAST_HPP_

#include "../Extension.h"
namespace lmlib {
namespace expr {
template <typename SrcExp, typename DType, int dimdst, int dimdst_m_cast>
struct Broadcast1DExp
    : public MakeTensorExp<Broadcast1DExp<SrcExp, DType, dimdst, dimdst_m_cast>,
                           SrcExp, dimdst, DType> {

  const SrcExp &src_;
  Broadcast1DExp(const SrcExp &src, Shape<dimdst> shape) : src_(src) {
    this->shape_ = shape;
  }
};

} // namespace expr

} // namespace lmlib

#endif // LMLIB_EXTENSION_BROADCAST_HPP_