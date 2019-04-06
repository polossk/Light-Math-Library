#ifndef LMLIB_DENSE_HPP_
#define LMLIB_DENSE_HPP_

#include <iostream>
#include <string>

#include "LMBase.hpp"
#include "Exp.hpp"
#include "Shape.hpp"
#include "Stream.hpp"

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

// Tensor1D
template <typename DType>
struct Tensor<1, DType> : public TRValue<Tensor<1, DType>, 1, DType> {
public:
  DType *dptr_;
  Shape<1> shape_;
  index_t stride_;
  Stream<Device> *stream_;

  // constructor
  inline Tensor(void) : stream_(NULL) {}

  inline Tensor(const Shape<1> &shape) : shape_(shape), stream_(NULL) {}

  inline Tensor(DType *dptr, Shape<1> shape)
      : dptr_(dptr), shape_(shape), stride_(shape[0]), stream_(NULL) {}

  inline Tensor(DType *dptr, Shape<1> shape, Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(shape[0]), stream_(stream) {}

  inline Tensor(DType *dptr, Shape<1> shape, index_t stride,
                Stream<Device> *stream)
      : dptr_(dptr), shape_(shape), stride_(stride), stream_(stream) {}

  inline void set_stream(Stream<Device> *stream) { this->stream_ = stream; }

  inline Tensor<1, DType> FlatTo1D(void) const { return *this; }

  inline Tensor<2, DType> FlatTo2D(void) const {
    return Tensor<2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);
  }

  inline Tensor<1, DType> Slice(index_t begin, index_t end) const {
    Shape<1> s;
    s[0] = end - begin;
    return Tensor<1, DType>(dptr_ + begin, s, s[0], stream_);
  }

  inline bool CheckContiguous(void) const { return true; }

  inline index_t MSize(void) const { return shape_[0]; }

  inline index_t size(index_t i) const { return shape_[0]; }

  inline DType &operator[](index_t idx) { return dptr_[idx]; }

  inline const DType &operator[](index_t idx) const { return dptr_[idx]; }

  /*!\brief implement the assignment of same type */
  inline Tensor<1, DType> &operator=(const Tensor<1, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    stride_ = exp.stride_;
    stream_ = exp.stream_;
    return *this;
  }

  template <typename E, int etype>
  inline Tensor<1, DType> &operator=(const expr::Exp<E, DType, etype> &exp) {
    return this->__assign(exp);
  }

  inline Tensor<1, DType> &operator=(const DType &exp) {
    return this->__assign(exp);
  }
};

//
inline void InitTensorEngine(int device_id = 0);

inline void ShutdownTensorEngine();

inline void SetDevice(int device_id);

inline Stream *NewStream(int device_id = -1);

inline void DeleteStream(Stream *stream);

template <int dim, typename DType>
inline void AllocSpace(Tensor<dim, DType> *obj, bool pad = MSHADOW_ALLOC_PAD);

template <int dim, typename DType>
inline void FreeSpace(Tensor<dim, DType> *obj);

template <int dim, typename DType>
inline Tensor<dim, DType> NewTensor(const Shape<dim> &shape, DType initv,
                                    bool pad = MSHADOW_ALLOC_PAD,
                                    Stream *stream = NULL);

template <int dim, typename DType>
inline void Copy(Tensor<dim, DType> dst, const Tensor<dim, DType> &src,
                 Stream *stream = NULL);

//
template <typename IndexType, typename DType>
inline void IndexFill(Tensor<2, DType> dst, const Tensor<1, IndexType> &index,
                      const Tensor<2, DType> &src);

//
template <typename KDType, typename VDType>
inline void SortByKey(Tensor<1, KDType> keys, Tensor<1, VDType> values,
                      bool is_ascend = true);

//
template <typename VDType, typename SDType>
inline void VectorizedSort(Tensor<1, VDType> values,
                           Tensor<1, SDType> segments);

//
template <typename Saver, typename RValue, int dim, typename DType,
          typename ExpType, int etype>
inline void MapExp(TRValue<RValue, dim, DType> *dst,
                   const expr::Exp<ExpType, DType, etype> &exp);

//
template <typename DType>
inline void VectorDot(Tensor<1, DType> dst, const Tensor<1, DType> &lhs,
                      const Tensor<1, DType> &rhs);
} // namespace lmlib

#endif // LMLIB_DENSE_HPP_
