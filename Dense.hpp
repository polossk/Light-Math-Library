#pragma once

#include "Exp.hpp"
#include "Shape.hpp"
#include "Stream.hpp"
#include "base.hpp"

namespace lmlib {
// comment todo
template <typename Container, int dimension, typename DType>
struct TRValue : public expr::RValueExp<Container, DType> {};

// comment todo
template <int dimension, typename DType LMLIB_DEFAULT_DTYPE>
struct Tensor : public TRValue<Tensor<dimension, DType>, dimension, DType> {
  static const int kSubDim = dimension - 1;
  DType *dptr_ = nullptr;
  Shape<dimension> shape_;
  index_t stride_;
  Stream *stream_;

  inline Tensor() : stream_(NULL) {}
  inline Tensor(const Shape<dimension> &shape) : shape_(shape), stream_(NULL) {}
  inline Tensor(DType *dptr, const Shape<dimension> &shape)
      : dptr_(dptr), shape_(shape), stride_(shape[kSubDim]), stream_(NULL) {}
  inline Tensor(DType *dptr, const Shape<dimension> &shape, Stream *stream)
      : dptr_(dptr), shape_(shape), stride_(shape[kSubDim]), stream_(stream) {}
  inline Tensor(DType *dptr, const Shape<dimension> &shape, index_t stride,
                Stream *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}

  inline void SetStream(Stream *stream) { this->stream_ = stream; }

  template <int startdim> inline index_t MemSize() const {
    index_t ret = this->stride_;
#pragma unroll
    for (int i = startdim; i < kSubdim; i++)
      ret *= this->shape_[i];
    return ret;
  }

  inline bool CheckContiguous() const {
    return this->shape_[dimension - 1] == stride_;
  }

  inline index_t MSize() const { this->MemSize<0>(); }

  inline index_t size(index_t idx) const { return shape_[idx]; }

  inline Tensor<1, DType> FlatTo1D() const {
    return Tensor<1, DType>(dptr_, shape_.FlatTo1D(), stride_, stream_);
  }

  inline Tensor<2, DType> FlatTo2D() const {
    return Tensor<2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }

  inline Tensor<kSubdim, DType> operator[](index_t idx) const {
    return Tensor<kSubdim, DType>(dptr_ + this->MemSize<1>() * idx,
                                  shape_.SubShape(), stride_, stream_);
  }

  inline Tensor<dimension, DType> Slice(index_t begin, index_t end) const {
    Shape<dimension> s = this->shape_;
    s[0] = end - begin;
    return Tensor<dimension, DType>(dptr_ + this->MemSize<1>() * begin, s,
                                    stride_, stream_);
  }

  inline Tensor<dimension, DType> &
  operator=(const Tensor<dimension, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }

  template <typename EType, int etype>
  inline Tensor<dimension, DType> &
  operator=(const expr::Exp<EType, DType, etype> &exp) {
    return this->__assign(exp);
  }

  inline Tensor<dimension, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};

//
template <typename Saver, typename RValue, int dim, typename DType,
          typename ExpType, int etype>
inline void MapExp(TRValue<RValue, dim, DType> *dst,
                   const expr::Exp<ExpType, DType, etype> &exp);

} // namespace lmlib
