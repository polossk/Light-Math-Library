#pragma once

#include "base.hpp"
#include "Shape.hpp"
#include "Exp.hpp"

namespace lmlib
{
	template <typename Container, int dimension, typename DType>
	struct TRValue : public expr::RValueExp<Container, DType> {};

	template <typename Saver, typename RValue, int dim,
		typename DType, typename ExpType, int etype>
		inline void MapExp(TRValue<RValue, dim, DType> *dst, const expr::Exp<ExpType, DType, etype> &exp);

}


