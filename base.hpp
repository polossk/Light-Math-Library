#pragma once

#include <iostream>

namespace lmlib
{
// ## unify all integer typename
#ifdef _MSC_VER
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <inttypes.h>
#endif

// ## define index_t for index usage
typedef int64_t index_t;

// ## define default datatype
using default_real_t = double;
#ifndef LMLIB_DEFAULT_DTYPE
#define LMLIB_DEFAULT_DTYPE = ::lmlib::default_real_t
#endif // !LMLIB_DEFAULT_DTYPE

// ## define baisc operators
namespace op
{
struct plus
{
  template <typename DType>
  inline static DType Map(DType a, DType b)
  {
    return a + b;
  }
};

struct minus
{
  template <typename DType>
  inline static DType Map(DType a, DType b)
  {
    return a - b;
  }
};

struct mul
{
  template <typename DType>
  inline static DType Map(DType a, DType b)
  {
    return a * b;
  }
};

struct div
{
  template <typename DType>
  inline static DType Map(DType a, DType b)
  {
    return a / b;
  }
};

struct rhs
{
  template <typename DType>
  inline static DType Map(DType a, DType b)
  {
    return b;
  }
};

struct identity
{
  template <typename DType>
  inline static DType Map(DType a)
  {
    return a;
  }
};
} // namespace op

// ## define saving operators
namespace sv
{
struct saveto
{
  using OPType = op::rhs;
  template <typename DType>
  inline static void Save(DType &a, DType b)
  {
    a = b;
  }
};

struct plusto
{
  using OPType = op::plus;
  template <typename DType>
  inline static void Save(DType &a, DType b)
  {
    a += b;
  }
};

struct minusto
{
  using OPType = op::minus;
  template <typename DType>
  inline static void Save(DType &a, DType b)
  {
    a -= b;
  }
};

struct multo
{
  using OPType = op::mul;
  template <typename DType>
  inline static void Save(DType &a, DType b)
  {
    a *= b;
  }
};

struct divto
{
  using OPType = op::div;
  template <typename DType>
  inline static void Save(DType &a, DType b)
  {
    a /= b;
  }
};
} // namespace sv

} // namespace lmlib
