#ifndef LIGHT_MATH_LIBRARY_SHAPE_HPP
#define LIGHT_MATH_LIBRARY_SHAPE_HPP

/**
 * @file Shape.hpp
 * @author polossk
 * @brief definition and implement of lmlib::Shape<int ndimmension>
 * @version 0.1
 * @date 2019-02-18
 *
 * @copyright Copyright (c) 2019
 *
 * lmlib::Shape<int ndimmension> 的定义与实现
 */

#include "base.hpp"

namespace lmlib {

/**
 * @brief storage the shape information
 *
 * 存储一个张量的尺寸信息
 * @tparam ndimension 表示一个 ndimension 阶张量
 */
template <int ndimension> struct Shape {
  /**
   * @brief indecating n-dimension tensor's shape
   *
   * 张量的维度
   * @note 例如 2 阶张量（矩阵）的 nDimension = 2，4 阶张量的 nDimension = 4
   */
  static const int nDimension = ndimension;

  /**
   * @brief basically nSubdim equals to nDimension - 1
   *
   * 基本上等同于 nDimension - 1
   * @note 主要用于后续处理子张量的尺寸 subshape()
   */
  static const int nSubdim = ndimension - 1;

  /**
   * @brief storage the shape information
   *
   * 记录尺寸信息
   * @note Basically a n-dinmension tensor needs a n-elements array to storage
   * the size of each dimension of tensor.
   * @note 一个 n 阶张量需要一个长度为 n 的数组来储存每一维度的尺寸信息。
   * 第 i 个元素记录第 i 维的大小。
   */
  index_t shape_[nDimension];

  /**
   * @brief Construct a new Shape object
   *
   */
  inline Shape() {}

  /**
   * @brief Construct a new Shape object
   *
   * @param s 从 s 复制一个新 Shape 对象
   */
  inline Shape(const Shape<nDimension> &s) {
#pragma unroll
    for (int i = 0; i < nDimension; i++)
      this->shape_[i] = s[i];
  }

  /**
   * @brief access/edit the shape_[]
   *
   * 读取/修改 shape_[]
   *
   * @param idx 维度下标
   * @return index_t& 该维度的尺寸
   */
  inline index_t &operator[](index_t idx) { return shape_[idx]; }

  /**
   * @brief access the shape_[]
   *
   * 读取 shape_[]
   *
   * @param idx 维度下标
   * @return index_t& 该维度的尺寸
   */
  inline const index_t &operator[](index_t idx) const { return shape_[idx]; }

  /**
   * @brief whether s is equal to this Shape or not
   *
   * 判断两个 Shape 对象是否相同
   *
   * @param s 另一个 Shape 对象
   * @return true 两者相同
   * @return false 两者不同
   */
  inline bool operator==(const Shape<nDimension> &s) const {
#pragma unroll
    for (int i = 0; i < nDimension; i++) {
      if (s.shape_[i] != this->shape_[i])
        return false;
    }
    return true;
  }

  /**
   * @brief opposite of previous method operator==
   *
   * 判断两个 Shape 对象是否不同
   *
   * @param s 另一个 Shape 对象
   * @return true 两者不同
   * @return false 两者相同
   */
  inline bool operator!=(const Shape<nDimension> &s) const {
    return !(*this == s);
  }

  /**
   * @brief number of valid elements
   *
   * 该形状下的张量可以储存多少元素
   *
   * @return index_t 张量的大小
   * @note (5, 3, 4).size() => 5 * 3 * 4 => 60
   */
  inline index_t size(void) const {
    index_t ret = this->shape_[0];
#pragma unroll
    for (int i = 1; i < nDimension; i++)
      ret *= this->shape_[i];
    return ret;
  }

  /**
   * @brief flat this Shape to 1-dimension tensor
   *
   * 把张量变换为一阶张量
   *
   * @return Shape<1> 变换后一阶张量的尺寸
   * @note (5, 3, 4).flat_to_1D() => (60,)
   */
  inline Shape<1> flat_to_1D() const {
    Shape<1> ret;
    ret[0] = this->size();
    return ret;
  }

  /**
   * @brief flat this Shape to 2-dimension tensor
   *
   * 把张量变换为二阶张量
   *
   * @return Shape<2> 变换后二阶张量的尺寸
   * @note (5, 3, 4).flat_to_2D => (15, 4)
   */
  inline Shape<2> flat_to_2D() const {
    Shape<2> ret;
    ret[1] = this->shape_[nDimension - 1];
    index_t placeholder = 1;
#pragma unroll
    for (int i = 0; i < nSubdim; i++)
      placeholder *= this->shape_[i];
    ret[0] = placeholder;
    return ret;
  }

  /**
   * @brief subshape that takes off largest dimension
   *
   * 子张量的尺寸大小
   *
   * @return Shape<nSubdim> 子张量的尺寸
   * @note (5, 3, 4).subshape() => (3, 4)
   */
  inline Shape<nSubdim> subshape() const {
    Shape<nSubdim> ret;
#pragma unroll
    for (int i = 1; i < nDimension; i++)
      ret[i - 1] = this->shape_[i];
    return ret;
  }

  /**
   * @brief slice the shape from start to end
   *
   * 张量切片的尺寸大小
   *
   * @tparam dimstart 开始维度
   * @tparam dimend 结束维度
   * @return Shape<dimend - dimstart> 切片的尺寸
   * @note (5, 6, 7, 4, 3).slice<2, 5>() => (7, 4, 3)
   */
  template <int dimstart, int dimend>
  inline Shape<dimend - dimstart> slice() const {
    Shape<dimend - dimstart> ret;
#pragma unroll
    for (int i = dimstart; i < dimend; i++)
      ret[i - dimstart] = this->shape_[i];
    return ret;
  }

  /**
   * @brief product shape in [dimstart,dimend)
   *
   * [dimstart,dimend) 的元素个数
   *
   * @param dimstart 开始维度
   * @param dimend 结束维度
   * @return index_t [dimstart,dimend) 的元素个数
   * @note (5, 6, 7, 4, 3).prodshape(2, 5) => 7 * 4 * 3 => 84
   */
  inline index_t prodshape(int dimstart, int dimend) {
    index_t ret = 1;
#pragma unroll
    for (int i = dimstart; i < dimend; i++)
      ret *= this->shape_[i];
    return ret;
  }
};

/**
 * @brief construct a one dimension shape, stride will equal _0
 *
 * 构造一个一阶张量的 Shape，stride 与 _0 相同
 *
 * @param _0 第 0 维的大小
 * @return Shape<1> 一阶张量的尺寸信息
 */
inline Shape<1> Shape1(index_t _0) {
  Shape<1> _;
  _[0] = _0;
  return _;
}

/**
 * @brief construct a two dimension shape, stride will equal _0
 *
 * 构造一个二阶张量的 Shape，stride 与 _0 相同
 *
 * @param _0 第 0 维的大小
 * @param _1 第 1 维的大小
 * @return Shape<2> 二阶张量的尺寸信息
 */
inline Shape<2> Shape2(index_t _0, index_t _1) {
  Shape<2> _;
  _[0] = _0;
  _[1] = _1;
  return _;
}

/**
 * @brief construct a three dimension shape, stride will equal _0
 *
 * 构造一个三阶张量的 Shape，stride 与 _0 相同
 *
 * @param _0 第 0 维的大小
 * @param _1 第 1 维的大小
 * @param _2 第 2 维的大小
 * @return Shape<3> 三阶张量的尺寸信息
 */
inline Shape<3> Shape3(index_t _0, index_t _1, index_t _2) {
  Shape<3> _;
  _[0] = _0;
  _[1] = _1;
  _[2] = _2;
  return _;
}

/**
 * @brief construct a four dimension shape, stride will equal _0
 *
 * 构造一个四阶张量的 Shape，stride 与 _0 相同
 *
 * @param _0 第 0 维的大小
 * @param _1 第 1 维的大小
 * @param _2 第 2 维的大小
 * @param _3 第 3 维的大小
 * @return Shape<2> 四阶张量的尺寸信息
 */
inline Shape<4> Shape4(index_t _0, index_t _1, index_t _2, index_t _3) {
  Shape<4> _;
  _[0] = _0;
  _[1] = _1;
  _[2] = _2;
  _[3] = _3;
  return _;
}

/**
 * @brief construct a five dimension shape, stride will equal _0
 *
 * 构造一个五阶张量的 Shape，stride 与 _0 相同
 *
 * @param _0 第 0 维的大小
 * @param _1 第 1 维的大小
 * @param _2 第 2 维的大小
 * @param _3 第 3 维的大小
 * @param _4 第 4 维的大小
 * @return Shape<2> 五阶张量的尺寸信息
 */
inline Shape<5> Shape5(index_t _0, index_t _1, index_t _2, index_t _3,
                       index_t _4) {
  Shape<5> _;
  _[0] = _0;
  _[1] = _1;
  _[2] = _2;
  _[3] = _3;
  _[4] = _4;
  return _;
}

/**
 * @brief output the Shape to std::ostream
 *
 * 输出 Shape 信息到标准输出流
 *
 * @tparam ndimension Shape 的阶数
 * @param os 输出流
 * @param _ 需要输出的 Shape
 * @return std::ostream& 输出流
 */
template <int ndimension>
inline std::ostream &operator<<(std::ostream &os, const Shape<ndimension> &_) {
  os << '(';
  for (int i = 0; i < ndimension; i++) {
    if (i != 0)
      os << ',';
    os << _[i];
  }
  if (ndimension == 1)
    os << ',';
  os << ')';
  return os;
}
} // namespace lmlib

#endif // LIGHT_MATH_LIBRARY_SHAPE_HPP