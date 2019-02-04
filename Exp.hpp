#pragma once

#include "base.hpp"

namespace lmlib
{
	namespace type
	{
		const int nRValue = 0;
		const int nMapper = 1;
		const int nChainer = 3;
		const int nComplex = 7;
	}


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
	inline TypecastExp<DstDType, SrcDType, EType, (exp_type|type::nMapper)>
		typecast(const Exp<EType, SrcDType, exp_type> &exp)
	{
		return TypecastExp<DstDType, SrcDType, EType, (exp_type | type::nMapper)>(exp.self());
	}

	template<typename EType, typename DType>
	struct TransposeExp : public Exp<TransposeExp<EType, DType>, DType, type::nChainer>
	{
		const EType &expr;
		explicit TransposeExp(const EType &_) : expr(_) {}
		inline const EType &T() const { return expr; }
	};

	template<typename Saver, typename RValue, typename DType>
	struct ExpEngine;
	// todo: ExpEngine

	template<typename Container, typename, DTyper>
	class RValueExp : public Exp<Container, DType, type::nRValue>
	{
	public:
		inline const TransposeExp<Container, DType> T() const
		{
			return TransposeExp<Container, DType>(this->self());
		}

		// todo: RValueExp
	};


} // namespace lmlib
