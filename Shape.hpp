#pragma once

#include "base.hpp"

namespace lmlib
{

	template<int ndimension>
	struct Shape
	{
		// 
		static const int nDimension = ndimension;

		//
		static const int nSubdim = ndimension - 1;

		// 
		index_t shape_[nDimension];

		// 
		inline Shape() {}

		// 
		inline Shape(const Shape<nDimension> &s)
		{
			#pragma unroll
			for (int i = 0; i < nDimension; i++)
				this->shape_[i] = s[i];
		}

		// 
		inline index_t &operator[](index_t idx) { return shape_[idx]; }
		inline const index_t &operator[](index_t idx) const { return shape_[idx]; }

		// 
		inline bool operator==(const Shape<nDimension> &s) const
		{
			#pragma unroll
			for (int i = 0; i < nDimension; i++)
			{
				if (s.shape_[i] != this->shape_[i])
					return false;
			}
			return true;
		}

		//
		inline bool operator!=(const Shape<nDimension> &s) const
		{
			return !(*this == s);
		}

		// (5, 3, 4).size() => 5 * 3 * 4 => 60
		inline index_t size(void) const
		{
			index_t ret = this->shape_[0];
			#pragma unroll
			for (int i = 1; i < nDimension; i++)
				ret *= this->shape_[i];
			return ret;
		}

		// (5, 3, 4).flat_to_1D() => (60,)
		inline Shape<1> flat_to_1D() const
		{
			Shape<1> ret;
			ret[0] = this->size();
			return ret;
		}

		// (5, 3, 4).flat_to_2D => (15, 4)
		inline Shape<2> flat_to_2D() const
		{
			Shape<2> ret;
			ret[1] = this->shape_[nDimension - 1];
			index_t placeholder = 1;
			#pragma unroll
			for (int i = 0; i < nSubdim; i++)
				placeholder *= this->shape_[i];
			ret[0] = placeholder;
			return ret;
		}

		// (5, 3, 4).subshape() => (3, 4)
		inline Shape<nSubdim> subshape() const
		{
			Shape<nSubdim> ret;
			#pragma unroll
			for (int i = 1; i < nDimension; i++)
				ret[i - 1] = this->shape_[i];
			return ret;
		}

		// (5, 6, 7, 4, 3).slice<2, 5>() => (7, 4, 3)
		template<int dimstart, int dimend>
		inline Shape<dimend - dimstart> slice() const
		{
			Shape<dimend - dimstart> ret;
			#pragma unroll
			for (int i = dimstart; i < dimend; i++)
				ret[i - dimstart] = this->shape_[i];
			return ret;
		}
	};

	inline Shape<1> Shape1(index_t _0)
	{
		Shape<1> _; _[0] = _0; return _;
	}

	inline Shape<2> Shape2(index_t _0, index_t _1)
	{
		Shape<2> _; _[0] = _0; _[1] = _1; return _;
	}

	inline Shape<3> Shape3(index_t _0, index_t _1, index_t _2)
	{
		Shape<3> _; _[0] = _0; _[1] = _1; _[2] = _2; return _;
	}

	inline Shape<4> Shape4(index_t _0, index_t _1, index_t _2, index_t _3)
	{
		Shape<4> _; _[0] = _0; _[1] = _1; _[2] = _2; _[3] = _3; return _;
	}

	inline Shape<5> Shape5(index_t _0, index_t _1, index_t _2, index_t _3, index_t _4)
	{
		Shape<5> _; _[0] = _0; _[1] = _1; _[2] = _2; _[3] = _3; _[4] = _4; return _;
	}

	template<int ndimension>
	inline std::ostream &operator<<(std::ostream &os, const Shape<ndimension> &_)
	{
		os << '(';
		for (int i = 0; i < ndimension; i++)
		{
			if (i != 0) os << ',';
			os << _[i];
		}
		if (ndimension == 1) os << ',';
		os << ')';
		return os;
	}

} // namespace lmlib