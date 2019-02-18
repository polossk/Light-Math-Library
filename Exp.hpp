#pragma once

#include "base.hpp"

namespace lmlib
{
	namespace expr
	{
		namespace type
		{
			const int nRValue = 0;
			const int nMapper = 1;
			const int nChainer = 3;
			const int nComplex = 7;
		} // namespace type


		template<typename TrueType, typename DType, int exp_type>
		struct Exp
		{
			inline const TrueType &self() const
			{
				return *static_cast<const TrueType*>(this);
			}
			inline TrueType *ptrself()
			{
				return static_cast<TrueType*>(this);
			}
		};

		template<typename Saver, typename RValue, typename DType>
		struct ExpEngine;

		// store a scalar value into a expression
		template<DType> struct ScalarExp : public Exp<ScalarExp<DType>, DType, type::nMapper>
		{
			DType scalar_;
			ScalarExp(DType scalar) : scalar_(scalar) {}
		};

		// auto pi = scalar<float>(3.14f);
		template<typename DType>
		inline ScalarExp<DType> scalar(DType _)
		{
			return ScalarExp<DType>(_);
		}

		// store a typecast function into a expression
		// from SrcDType to DstDType, DType means DataType of early Exp defination
		// EType means ExpressionType, which's itself
		template<typename DstDType, typename SrcDType, typename EType, int exp_type>
		struct TypecastExp
			: public Exp<TypecastExp<DstDType, SrcDType, EType, exp_type>, DstDType, exp_type>
		{
			const EType &expr;
			explicit TypecastExp(const EType &_) : expr(_) {}
		};

		// mat = 3.2f;
		// mat_integer = typecast<int>(mat);
		template<typename DstDType, typename SrcDType, typename EType, int exp_type>
		inline TypecastExp<DstDType, SrcDType, EType, (exp_type | type::nMapper)>
			typecast(const Exp<EType, SrcDType, exp_type> &exp)
		{
			return TypecastExp<DstDType, SrcDType, EType, (exp_type | type::nMapper)>(exp.self());
		}

		// comment todo
		template<typename EType, typename DType>
		struct TransposeExp : public Exp<TransposeExp<EType, DType>, DType, type::nChainer>
		{
			const EType &expr;
			explicit TransposeExp(const EType &_) : expr(_) {}
			inline const EType &T() const { return expr; }
		};

		// comment todo
		template<typename Container, typename DType>
		class RValueExp : public Exp<Container, DType, type::nRValue>
		{
		public:
			inline const TransposeExp<Container, DType> T() const
			{
				return TransposeExp<Container, DType>(this->self());
			}

			inline Container &operator+=(DType s)
			{
				ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
				return *(this->ptrself());
			}

			inline Container &operator-=(DType s)
			{
				ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
				return *(this->ptrself());
			}

			inline Container &operator*=(DType s)
			{
				ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
				return *(this->ptrself());
			}

			inline Container &operator/=(DType s)
			{
				ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
				return *(this->ptrself());
			}

			inline Container &__assign(DType s)
			{
				ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), scalar<DType>(s));
				return *(this->ptrself());
			}

			inline Container &__assign(const Exp<Container, DType, type::nRValue> &exp);

			template <typename E, int etype>
			inline Container &__assign(const Exp<E, DType, etype> &exp)
			{
				ExpEngine<sv::saveto, Container, DType>::Eval(this->ptrself(), exp.self());
				return *(this->ptrself());
			}

			template <typename E, int etype>
			inline Container &operator+=(const Exp<E, DType, etype> &exp)
			{
				ExpEngine<sv::plusto, Container, DType>::Eval(this->ptrself(), exp.self());
				return *(this->ptrself());
			}

			template <typename E, int etype>
			inline Container &operator-=(const Exp<E, DType, etype> &exp)
			{
				ExpEngine<sv::minusto, Container, DType>::Eval(this->ptrself(), exp.self());
				return *(this->ptrself());
			}

			template <typename E, int etype>
			inline Container &operator*=(const Exp<E, DType, etype> &exp)
			{
				ExpEngine<sv::multo, Container, DType>::Eval(this->ptrself(), exp.self());
				return *(this->ptrself());
			}

			template <typename E, int etype>
			inline Container &operator/=(const Exp<E, DType, etype> &exp)
			{
				ExpEngine<sv::divto, Container, DType>::Eval(this->ptrself(), exp.self());
				return *(this->ptrself());
			}
		};

		// comment todo
		template <typename Tlhs, typename Trhs, bool ltrans, bool rtrans, typenam DType>
		struct DotExp : public Exp<DotExp<Tlhs, Trhs, ltrans, rtrans, DType>, DType, type::nComplex>
		{
			const Tlhs &lhs_;
			const Trhs &rhs_;
			DType scale_;
			explicit DotExp(const Tlhs &lhs, const Trhs &rhs, DType scale)
				: lhs_(lhs), rhs_(rhs), scale_(scale) {}
		};

		template <typename Tlhs, typename Trhs, typename DType>
		inline DotExp<Tlhs, Trhs, false, false, DType>
			dot(const RValueExp<Tlhs, DType> &lhs, const RValueExp<Trhs, DType> &rhs)
		{
			return DotExp<Tlhs, Trhs, false, false, DType>(lhs.self(), rhs.self(), DType(1.0f));
		}

		template <typename Tlhs, typename Trhs, typename DType>
		inline DotExp<Tlhs, Trhs, false, true, DType>
			dot(const RValueExp<Tlhs, DType> &lhs, const RValueExp<Trhs, DType> &rhs)
		{
			return DotExp<Tlhs, Trhs, false, true, DType>(lhs.self(), rhs.expr, DType(1.0f));
		}

		template <typename Tlhs, typename Trhs, typename DType>
		inline DotExp<Tlhs, Trhs, true, false, DType>
			dot(const RValueExp<Tlhs, DType> &lhs, const RValueExp<Trhs, DType> &rhs)
		{
			return DotExp<Tlhs, Trhs, true, false, DType>(lhs.expr, rhs.self(), DType(1.0f));
		}

		template <typename Tlhs, typename Trhs, typename DType>
		inline DotExp<Tlhs, Trhs, true, true, DType>
			dot(const RValueExp<Tlhs, DType> &lhs, const RValueExp<Trhs, DType> &rhs)
		{
			return DotExp<Tlhs, Trhs, true, true, DType>(lhs.expr, rhs.expr, DType(1.0f));
		}

		template <bool transpose_left, bool transpose_right, typename Tlhs, typename Trhs, typename DType>
		inline DotExp<Tlhs, Trhs, transpose_left, transpose_right, DType>
			batch_dot(const RValueExp<Tlhs, DType> &lhs, const RValueExp<Trhs, DType> &rhs)
		{
			return DotExp<Tlhs, Trhs, transpose_left, transpose_right, DType>(lhs.self(), rhs.self(), DType(1.0f));
		}

		// comment todo
		template <typename OP, typename TA, typename TB, typename TC, typename DType, int etype>
		struct TernaryMapExp : public Exp<TernaryMapExp<OP, TA, TB, TC, DType, etype>, DType, etype>
		{
			const TA &_1_;
			const TB &_2_;
			const TC &_3_;
			explicit TernaryMapExp(const TA &_1, const TB &_2, const TC &_3)
				: _1_(_1), _2_(_2), _3_(_3) {}
		};

		template <typename OP, typename TA, typename TB, typename TC, typename DType,
			int eta, int etb, int etc>
			inline TernaryMapExp<OP, TA, TB, TC, DType, (eta | etb | etc | type::nMapper)>
			MakeExp(const Exp<TA, DType, eta> &_1, const Exp<TB, DType, etb> &_2, const Exp<TC, DType, etc> &_3)
		{
			return TernaryMapExp<OP, TA, TB, TC, DType, (eta | etb | etc | type::nMapper)>(_1.self(), _2.self(), _3.self());
		}

		template <typename OP, typename TA, typename TB, typename TC, typename DType,
			int eta, int etb, int etc>
			inline TernaryMapExp<OP, TA, TB, TC, DType, (eta | etb | etc | type::nMapper)>
			F(const Exp<TA, DType, eta> &_1, const Exp<TB, DType, etb> &_2, const Exp<TC, DType, etc> &_3)
		{
			return MakeExp<OP>(_1, _2, _3);
		}

		// comment todo
		template <typename OP, typename Tlhs, typename Trhs, typename DType, int etype>
		struct BinaryMapExp : public Exp < BinaryMapExp<OP, Tlhs, Trhs, DType, etype>
		{
			const Tlhs &lhs_;
			const Trhs &rhs_;
			explicit BinaryMapExp(const Tlhs &lhs, const Trhs &rhs) : lhs(lhs_), rhs(rhs_) {}
		};

		template <typename OP, typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<OP, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			MakeExp(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return BinaryMapExp<OP, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>(lhs.self(), rhs.self());
		}

		template <typename OP, typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<OP, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			F(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return MakeExp<OP>(lhs, rhs);
		}

		template <typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<op::plus, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			operator+(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return MakeExp<op::plus>(lhs, rhs);
		}

		template <typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<op::minus, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			operator-(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return MakeExp<op::minus>(lhs, rhs);
		}

		template <typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<op::mul, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			operator*(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return MakeExp<op::mul>(lhs, rhs);
		}

		template <typename Tlhs, typename Trhs, typename DType, int etlhs, int etrhs>
		inline BinaryMapExp<op::div, Tlhs, Trhs, DType, (etlhs | etrhs | type::nMapper)>
			operator/(const Exp<Tlhs, DType, etlhs> & lhs, const Exp<Trhs, DType, etrhs> & rhs)
		{
			return MakeExp<op::div>(lhs, rhs);
		}

		// comment todo
		template<typename OP, typename TA, typename DType, int etype>
		struct UnaryMapExp : public Exp<UnaryMapExp<OP, TA, DType, etype>, DType, etype>
		{
			const TA &src_;
			explicit UnaryMapExp(const TA &src) : src_(src) {}
		};

		template<typename OP, typename TA, typename DType, int etype>
		inline UnaryMapExp<OP, TA, DType, (etype | type::kMapper)>
			MakeExp(const Exp<TA, DType, etype> &src)
		{
			return UnaryMapExp<OP, TA, DType, (ta | type::kMapper)>(src.self());
		}

		template<typename OP, typename TA, typename DType, int etype>
		inline UnaryMapExp<OP, TA, DType, (etype | type::kMapper)>
			F(const Exp<TA, DType, etype> &src)
		{
			return MakeExp<OP>(src);
		}

	} // namespace expr
} // namespace lmlib
