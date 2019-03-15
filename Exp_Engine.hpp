#ifndef LMLIB_EXP_ENGINE_HPP_
#define LMLIB_EXP_ENGINE_HPP_

#include "./Dense.hpp"
#include "./LMBase.hpp"

namespace lmlib {
namespace expr {

template <typename SubType, typename SrcExp, int dim, typename DType>
struct MakeTensorExp : public Exp<MakeTensorExp<SubType, SrcExp, dim, DType>,
                                  DType, type::kChainer> {
  Shape<dim> shape_;
  inline const SubType &real_self() const {
    return *static_cast<const SubType *>(this);
  }
};

template <typename ExpType, typename DType> class Plan {
public:
  inline DType Eval(index_t y, index_t x) const;
};

template <int dim, typename DType> class Plan<Tensor<dim, DType>, DType> {
public:
  explicit Plan(const Tensor<dim, DType> &t)
      : dptr_(t.dptr_), stride_(t.stride_) {}
  inline DType &REval(index_t y, index_t x) { return dptr_[y * stride_ + x]; }
  inline const DType &REval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

private:
  DType *dptr_;
  index_t stride_;
};

template <typename DType> class Plan<Tensor<1, DType>, DType> {
public:
  explicit Plan(const Tensor<1, DType> &t) : dptr_(t.dptr_) {}
  inline DType &REval(index_t y, index_t x) { return dptr_[x]; }
  inline const DType &REval(index_t y, index_t x) const { return dptr_[x]; }

private:
  DType *dptr_;
};

template <typename DType> class Plan<ScalarExp<DType>, DType> {
public:
  explicit Plan(DType scalar) : scalar_(scalar) {}
  inline DType Eval(index_t y, index_t x) const { return scalar_; }

private:
  DType scalar_;
};

} // namespace expr

} // namespace lmlib

#endif // LMLIB_  EXP_ENGINE_HPP_
