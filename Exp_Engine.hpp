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

template <typename DstDType, typename SrcDType, typename EType, int etype>
class Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType> {
public:
  explicit Plan(const Plan<EType, SrcDType> &src) : src_(src) {}
  inline DstDType Eval(index_t y, index_t x) const {
    return DstDType(src_.Eval(y, x));
  }

private:
  Plan<EType, SrcDType> src_;
};

// ternary expression
template <typename OP, typename TA, typename TB, typename TC, typename DType,
          int etype>
class Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &item1, const Plan<TB, DType> &item2,
                const Plan<TC, DType> &item3)
      : item1_(item1), item2_(item2), item3_(item3) {}
  inline DType Eval(index_t y, index_t x) const {
    return OP::Map(item1_.Eval(y, x), item2_.Eval(y, x), item3_.Eval(y, x));
  }

private:
  Plan<TA, DType> item1_;
  Plan<TB, DType> item2_;
  Plan<TC, DType> item3_;
};

// binary expression
template <typename OP, typename TA, typename TB, typename DType, int etype>
class Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &lhs, const Plan<TB, DType> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  inline DType Eval(index_t y, index_t x) const {
    return OP::Map(item1_.Eval(y, x), rhs_.Eval(y, x));
  }

private:
  Plan<TA, DType> lhs_;
  Plan<TB, DType> rhs_;
};

// unary expression
template <typename OP, typename TA, typename DType, int etype>
class Plan<UnaryMapExp<OP, TA, DType, etype>, DType> {
public:
  explicit Plan(const Plan<TA, DType> &src) : src_(src) {}
  inline DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

private:
  Plan<TA, DType> src_;
};

// transpose
template <typename EType, typename DType>
class Plan<TransposeExp<EType, DType>, DType> {
public:
  explicit Plan(const Plan<EType, DType> &src) : src_(src) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return src_.Eval(x, y);
  }

private:
  Plan<EType, DType> src_;
};

} // namespace expr

} // namespace lmlib

#endif // LMLIB_  EXP_ENGINE_HPP_
