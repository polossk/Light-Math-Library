#ifndef LMLIB_EXP_ENGINE_HPP_
#define LMLIB_EXP_ENGINE_HPP_

#include "./Dense.hpp"
#include "./LMBase.hpp"
#include "./Logging.hpp"

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

template <typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template <typename OP, typename TA, typename TB, typename TC, typename DType,
          int etype>
inline Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType>
MakePlan(const TernaryMapExp<OP, TA, TB, TC, DType, etype> &e);

template <typename DType>
inline Plan<ScalarExp<DType>, DType> MakePlan(const ScalarExp<DType> &e) {
  return Plan<ScalarExp<DType>, DType>(e.scalar_);
}

template <typename DstDType, typename SrcDType, typename EType, int etype>
inline Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>
MakePlan(const TypecastExp<DstDType, SrcDType, EType, etype> &e) {
  return Plan<TypecastExp<DstDType, SrcDType, EType, etype>, DstDType>(
      MakePlan(e.exp));
}

template <typename T, typename DType>
inline Plan<T, DType> MakePlan(const RValueExp<T, DType> &e) {
  return Plan<T, DType>(e.self());
}

template <typename T, typename DType>
inline Plan<TransposeExp<T, DType>, DType>
MakePlan(const TransposeExp<T, DType> &e) {
  return Plan<TransposeExp<T, DType>, DType>(MakePlan(e.exp));
}

template <typename T, typename SrcExp, int dim, typename DType>
inline Plan<T, DType> MakePlan(const MakeTensorExp<T, SrcExp, dim, DType> &e) {
  return Plan<T, DType>(e.real_self());
}

template <typename OP, typename TA, typename DType, int etype>
inline Plan<UnaryMapExp<OP, TA, DType, etype>, DType>
MakePlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return Plan<UnaryMapExp<OP, TA, DType, etype>, DType>(MakePlan(e.src_));
}

template <typename OP, typename TA, typename TB, typename DType, int etype>
inline Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakePlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return Plan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>(MakePlan(e.lhs_),
                                                             MakePlan(e.rhs_));
}

// Ternary
template <typename OP, typename TA, typename TB, typename TC, typename DType,
          int etype>
inline Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType>
MakePlan(const TernaryMapExp<OP, TA, TB, TC, DType, etype> &e) {
  return Plan<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType>(
      MakePlan(e.item1_), MakePlan(e.item2_), MakePlan(e.item3_));
}

// if ExpInfo<E>::kDim == -1, mismatching expression
template <typename E> struct ExpInfo { static const int kDim = -1; };

template <typename DType> struct ExpInfo<ScalarExp<DType>> {
  static const int kDim = 0;
};

template <typename E, typename DType> struct ExpInfo<TransposeExp<E, DType>> {
  static const int kDim = ExpInfo<E>::kDim;
};

template <typename DstDType, typename SrcDType, typename EType, int etype>
struct ExpInfo<TypecastExp<DstDType, SrcDType, EType, etype>> {
  static const int kDim = ExpInfo<EType>::kDim;
};

template <int dim, typename DType> struct ExpInfo<Tensor<dim, DType>> {
  static const int kDim = dim;
};

template <typename T, typename SrcExp, int dim, typename DType>
struct ExpInfo<MakeTensorExp<T, SrcExp, dim, DType>> {
  static const int kDimSrc = ExpInfo<SrcExp>::kDim;
  static const int kDim = kDimSrc >= 0 ? dim : -1;
};

template <typename OP, typename TA, typename DType, int etype>
struct ExpInfo<UnaryMapExp<OP, TA, DType, etype>> {
  static const int kDim = ExpInfo<TA>::kDim;
};

template <typename OP, typename TA, typename TB, typename DType, int etype>
struct ExpInfo<BinaryMapExp<OP, TA, TB, DType, etype>> {
  static const int kDimLhs = ExpInfo<TA>::kDim;
  static const int kDimRhs = ExpInfo<TB>::kDim;
  static const int kDim =
      (kDimLhs >= 0 && kDimRhs >= 0)
          ? (kDimLhs == 0
                 ? kDimRhs
                 : ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1))
          : -1;
};

template <typename OP, typename TA, typename TB, typename TC, typename DType,
          int etype>
struct ExpInfo<TernaryMapExp<OP, TA, TB, TC, DType, etype>> {
  static const int kDimItem1 = ExpInfo<TA>::kDim;
  static const int kDimItem2 = ExpInfo<TB>::kDim;
  static const int kDimItem3 = ExpInfo<TC>::kDim;
  static const int kDim = kDimItem1;
};

template <int dim, typename DType, typename E> struct TypeCheck {
  static const int kExpDim = ExpInfo<E>::kDim;
  static const bool kMapPass = (kExpDim == 0 || kExpDim == dim);
  static const bool kRedPass = (kExpDim > dim);
};

template <bool kPass> struct TypeCheckPass;
// Todo : add static assert using C++11
template <> struct TypeCheckPass<false> {};
template <> struct TypeCheckPass<true> {
  inline static void Error_All_Tensor_in_Exp_Must_Have_Same_Type(void) {}
  inline static void Error_TypeCheck_Not_Pass_For_Reduce_Exp(void) {}
  inline static void Error_Expression_Does_Not_Meet_Dimension_Req(void) {}
};

template <int dim, typename E> struct ShapeCheck {
  inline static Shape<dim> Check(const E &t);
};
template <int dim, typename DType> struct ShapeCheck<dim, ScalarExp<DType>> {
  inline static Shape<dim> Check(const ScalarExp<DType> &exp) {
    // use lowest dimension to mark scalar exp
    Shape<dim> shape;
    for (int i = 0; i < dim; ++i) {
      shape[i] = 0;
    }
    return shape;
  }
};
} // namespace expr

} // namespace lmlib

#endif // LMLIB_  EXP_ENGINE_HPP_
