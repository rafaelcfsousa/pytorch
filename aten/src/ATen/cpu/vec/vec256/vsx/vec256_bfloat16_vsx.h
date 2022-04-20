#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#include <sleef.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

static inline void convertBF16ToF32(
    const vuint16& a,
    vfloat32& hi,
    vfloat32& lo) {
  vuint16 thi = vec_mergeh(reinterpret_cast<vuint16>(vi_0), a);
  vuint16 tlo = vec_mergel(reinterpret_cast<vuint16>(vi_0), a);
  hi = reinterpret_cast<vfloat32>(thi);
  lo = reinterpret_cast<vfloat32>(tlo);
}

static inline vuint16 convertF32ToBF16(const vfloat32& a, const vfloat32& b) {
  vuint32 hi = reinterpret_cast<vuint32>(a);
  vuint32 lo = reinterpret_cast<vuint32>(b);
  // uint32_t lsb = (input >> 16) & 1;
  vuint32 thi = vec_and(vec_sr(hi, vu_16), (vuint32)vi_1);
  vuint32 tlo = vec_and(vec_sr(lo, vu_16), (vuint32)vi_1);
  // uint32_t rounding_bias = 0x7fff + lsb;
  thi = vec_add(thi, (vuint32)v0x7fff);
  tlo = vec_add(tlo, (vuint32)v0x7fff);
  // input += rounding_bias;
  thi = vec_add(thi, hi);
  tlo = vec_add(tlo, lo);
  // input = input >> 16;
  thi = vec_sr(thi, vu_16);
  tlo = vec_sr(tlo, vu_16);
  // Pack the low-order half of each element in t_[hi,lo]
  vuint16 pack = vec_pack(thi, tlo);
  return pack;
}

static inline vuint16 convertF32ToBF16OnlyShift(
    const vfloat32& a,
    const vfloat32& b) {
  vuint32 hi = reinterpret_cast<vuint32>(a);
  vuint32 lo = reinterpret_cast<vuint32>(b);
  vuint32 thi = vec_sr(hi, vu_16);
  vuint32 tlo = vec_sr(lo, vu_16);
  vuint16 pack = vec_pack(thi, tlo);
  return pack;
}

template <>
class Vectorized<BFloat16> {
 private:
  union {
    struct {
      vuint16 _vec0;
      vuint16 _vec1;
    };
    struct {
      vbool16 _vecb0;
      vbool16 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = uint16_t;
  using vec_internal_type = vuint16;
  using vec_internal_mask_type = vbool16;
  using size_type = int;

  static constexpr size_type size() {
    return 16;
  }
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(vuint16 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool16 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vuint16 v1, vuint16 v2)
      : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool16 v1, vbool16 v2)
      : _vecb0{v1}, _vecb1{v2} {}
  C10_ALWAYS_INLINE Vectorized(BFloat16 scalar)
      : _vec0{vec_splats(scalar.x)}, _vec1{vec_splats(scalar.x)} {}
  C10_ALWAYS_INLINE Vectorized(BFloat16 scalar1, BFloat16 scalar2,
      BFloat16 scalar3, BFloat16 scalar4, BFloat16 scalar5, BFloat16 scalar6,
      BFloat16 scalar7, BFloat16 scalar8, BFloat16 scalar9, BFloat16 scalar10,
      BFloat16 scalar11, BFloat16 scalar12, BFloat16 scalar13,
      BFloat16 scalar14, BFloat16 scalar15, BFloat16 scalar16)
      : _vec0{vuint16{scalar1.x, scalar2.x, scalar3.x, scalar4.x, scalar5.x,
            scalar6.x, scalar7.x, scalar8.x}},
        _vec1{vuint16{scalar9.x, scalar10.x, scalar11.x, scalar12.x,
            scalar13.x, scalar14.x, scalar15.x, scalar16.x}} {}
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  const BFloat16& operator[](int idx) const  = delete;
  BFloat16& operator[](int idx) = delete;

  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 0, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    return a;
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 1, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    return b;
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 2, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    return {b._vec0, a._vec1};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 3, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    return {a._vec0, b._vec1};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 4, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    const vbool16 mask_1st = VsxHalfWMask1(mask);
    return {(vuint16)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 5, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    const vbool16 mask_1st = VsxHalfWMask1(mask);
    return {(vuint16)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 6, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    const vbool16 mask_2nd = VsxHalfWMask2(mask);
    return {a._vec0, (vuint16)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 7, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    const vbool16 mask_2nd = VsxHalfWMask2(mask);
    return {b._vec0, (vuint16)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }
  template <uint64_t mask>
  static std::
      enable_if_t<blendChoice(mask, 0xFF, 0xFF00) == 8, Vectorized<BFloat16>>
          C10_ALWAYS_INLINE
          blend(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    const vbool16 mask_1st = VsxHalfWMask1(mask);
    const vbool16 mask_2nd = VsxHalfWMask2(mask);
    return {
        (vuint16)vec_sel(a._vec0, b._vec0, mask_1st),
        (vuint16)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }
  static Vectorized<BFloat16> C10_ALWAYS_INLINE blendv(
      const Vectorized<BFloat16>& a,
      const Vectorized<BFloat16>& b,
      const Vectorized<BFloat16>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }
  template <typename step_t>
  static Vectorized<BFloat16> arange(
      BFloat16 base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<BFloat16>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }
  static Vectorized<BFloat16> set(const Vectorized<BFloat16>& a,
      const Vectorized<BFloat16>& b, int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
    }
    return b;
  }
  static Vectorized<BFloat16> C10_ALWAYS_INLINE loadu(const void* ptr) {
    return {
      vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
      vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
  }
  static Vectorized<BFloat16> C10_ALWAYS_INLINE
  loadu(const void* ptr, int16_t count) {
    __at_align__ value_type tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(value_type));
    return loadu(tmp_values);
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  int zero_mask() const {
    return 0;
  }

  Vectorized<BFloat16> abs() const {
    return *this;
  }
  Vectorized<BFloat16> angle() const {
    return *this;
  }
  Vectorized<BFloat16> real() const {
    return *this;
  }
  Vectorized<BFloat16> imag() const {
    return *this;
  }
  Vectorized<BFloat16> conj() const {
    return *this;
  }
  Vectorized<BFloat16> acos() const {
    return *this;
  }
  Vectorized<BFloat16> asin() const {
    return *this;
  }
  Vectorized<BFloat16> atan() const {
    return *this;
  }
  Vectorized<BFloat16> atan2(const Vectorized<BFloat16>& b) const {
    return *this;
  }
  Vectorized<BFloat16> copysign(const Vectorized<BFloat16>& sign) const {
    return *this;
  }
  Vectorized<BFloat16> erf() const {
    return *this;
  }
  Vectorized<BFloat16> erfc() const {
    return *this;
  }
  Vectorized<BFloat16> erfinv() const {
    return *this;
  }
  Vectorized<BFloat16> exp() const {
    return *this;
  }
  Vectorized<BFloat16> expm1() const {
    return *this;
  }
  Vectorized<BFloat16> fmod(const Vectorized<BFloat16>& q) const {
    return *this;
  }
  Vectorized<BFloat16> hypot(const Vectorized<BFloat16>& b) const {
    return *this;
  }
  Vectorized<BFloat16> i0() const {
    return *this;
  }
  Vectorized<BFloat16> i0e() const {
    return *this;
  }
  Vectorized<BFloat16> igamma(const Vectorized<BFloat16>& x) const {
    return *this;
  }
  Vectorized<BFloat16> igammac(const Vectorized<BFloat16>& x) const {
    return *this;
  }
  Vectorized<BFloat16> log() const {
    return *this;
  }
  Vectorized<BFloat16> log2() const {
    return *this;
  }
  Vectorized<BFloat16> log10() const {
    return *this;
  }
  Vectorized<BFloat16> log1p() const {
    return *this;
  }
  Vectorized<BFloat16> sin() const {
    return *this;
  }
  Vectorized<BFloat16> sinh() const {
    return *this;
  }
  Vectorized<BFloat16> cos() const {
    return *this;
  }
  Vectorized<BFloat16> cosh() const {
    return *this;
  }
  Vectorized<BFloat16> ceil() const {
    return *this;
  }
  Vectorized<BFloat16> floor() const {
    return *this;
  }
  Vectorized<BFloat16> neg() const {
    return *this;
  }
  Vectorized<BFloat16> round() const {
    return *this;
  }
  Vectorized<BFloat16> tan() const {
    return *this;
  }
  Vectorized<BFloat16> tanh() const {
    return *this;
  }
  Vectorized<BFloat16> trunc() const {
    return *this;
  }
  Vectorized<BFloat16> lgamma() const {
    return *this;
  }
  Vectorized<BFloat16> sqrt() const {
    return *this;
  }
  Vectorized<BFloat16> reciprocal() const {
    return *this;
  }
  Vectorized<BFloat16> rsqrt() const {
    return *this;
  }
  Vectorized<BFloat16> pow(const Vectorized<BFloat16>& b) const {
    return *this;
  }

  Vectorized<BFloat16> frac() const {
    return *this;
  }

  Vectorized<BFloat16> inline operator>(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> inline operator<(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> inline operator>=(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> inline operator<=(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> inline operator==(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> inline operator!=(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

template <typename Op>
Vectorized<BFloat16> static C10_ALWAYS_INLINE map_binary_op(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    Op op) {
  vfloat32 a1, a2, a3, a4;
  vfloat32 b1, b2, b3, b4;
  convertBF16ToF32(a.vec0(), a1, a2);
  convertBF16ToF32(a.vec1(), a3, a4);
  convertBF16ToF32(b.vec0(), b1, b2);
  convertBF16ToF32(b.vec1(), b3, b4);
  auto c1 = op(a1, b1);
  auto c2 = op(a2, b2);
  auto c3 = op(a3, b3);
  auto c4 = op(a4, b4);
  return {(vuint16)convertF32ToBF16(c1, c2), (vuint16)convertF32ToBF16(c3, c4)};
}

template <typename Op>
Vectorized<BFloat16> static C10_ALWAYS_INLINE map_comparison_op(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    Op op) {
  vfloat32 a1, a2, a3, a4;
  vfloat32 b1, b2, b3, b4;
  convertBF16ToF32(a.vec0(), a1, a2);
  convertBF16ToF32(a.vec1(), a3, a4);
  convertBF16ToF32(b.vec0(), b1, b2);
  convertBF16ToF32(b.vec1(), b3, b4);
  auto c1 = reinterpret_cast<vfloat32>(op(a1, b1));
  auto c2 = reinterpret_cast<vfloat32>(op(a2, b2));
  auto c3 = reinterpret_cast<vfloat32>(op(a3, b3));
  auto c4 = reinterpret_cast<vfloat32>(op(a4, b4));
  return {
      (vuint16)convertF32ToBF16OnlyShift(c1, c2),
      (vuint16)convertF32ToBF16OnlyShift(c3, c4)};
}

Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator>(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmpgt(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator==(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmpeq(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator>=(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmpge(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator<=(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmple(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator<(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmplt(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::operator!=(const Vectorized<BFloat16>& other) const {
  return map_comparison_op(
      *this, other, [](vfloat32& a, vfloat32& b) { return vec_cmpne(a, b); });
}

Vectorized<BFloat16> C10_ALWAYS_INLINE
operator+(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return map_binary_op(
      a, b, [](vfloat32& a, vfloat32& b) { return vec_add(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return map_binary_op(
      a, b, [](vfloat32& a, vfloat32& b) { return vec_sub(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return map_binary_op(
      a, b, [](vfloat32& a, vfloat32& b) { return vec_mul(a, b); });
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return map_binary_op(
      a, b, [](vfloat32& a, vfloat32& b) { return vec_div(a, b); });
}

Vectorized<BFloat16> C10_ALWAYS_INLINE
operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return {
      (vuint16)vec_and(a.vec0(), b.vec0()),
      (vuint16)vec_and(a.vec1(), b.vec1())};
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return {
      (vuint16)vec_or(a.vec0(), b.vec0()), (vuint16)vec_or(a.vec1(), b.vec1())};
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return {
      (vuint16)vec_xor(a.vec0(), b.vec0()),
      (vuint16)vec_xor(a.vec1(), b.vec1())};
}

Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::eq(const Vectorized<BFloat16>& other) const {
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::ne(const Vectorized<BFloat16>& other) const {
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::gt(const Vectorized<BFloat16>& other) const {
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::ge(const Vectorized<BFloat16>& other) const {
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::lt(const Vectorized<BFloat16>& other) const {
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}
Vectorized<BFloat16> C10_ALWAYS_INLINE
Vectorized<BFloat16>::le(const Vectorized<BFloat16>& other) const {
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

template <>
Vectorized<BFloat16> C10_ALWAYS_INLINE fmadd(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    const Vectorized<BFloat16>& c) {
  vfloat32 a1, a2, a3, a4;
  vfloat32 b1, b2, b3, b4;
  vfloat32 c1, c2, c3, c4;
  convertBF16ToF32(a.vec0(), a1, a2);
  convertBF16ToF32(a.vec1(), a3, a4);
  convertBF16ToF32(b.vec0(), b1, b2);
  convertBF16ToF32(b.vec1(), b3, b4);
  convertBF16ToF32(c.vec0(), c1, c2);
  convertBF16ToF32(c.vec1(), c3, c4);
  return Vectorized<BFloat16>{
      (vuint16)convertF32ToBF16(vec_madd(a1, b1, c1), vec_madd(a2, b2, c2)),
      (vuint16)convertF32ToBF16(vec_madd(a3, b3, c3), vec_madd(a4, b4, c4))};
}

template <>
Vectorized<BFloat16> inline maximum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return a;
}

template <>
Vectorized<BFloat16> inline minimum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return a;
}

template <>
Vectorized<BFloat16> inline clamp(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min,
    const Vectorized<BFloat16>& max) {
  return a;
}

template <>
Vectorized<BFloat16> inline clamp_max(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& max) {
  return a;
}

template <>
Vectorized<BFloat16> inline clamp_min(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min) {
  return a;
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {}

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<BFloat16>& a) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K];
  __at_align__ BFloat16 arr2[K];
  a.store(arr2);
  convert(arr2, arr, K);
  return std::make_tuple(
      Vectorized<float>::loadu(arr),
      Vectorized<float>::loadu(arr + Vectorized<float>::size()));
}

inline Vectorized<BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K];
  __at_align__ BFloat16 arr2[K];
  a.store(arr);
  b.store(arr + Vectorized<float>::size());
  convert(arr, arr2, K);
  return Vectorized<BFloat16>::loadu(arr2);
}

inline void load_fp32_from_bf16(
    const c10::BFloat16* data,
    Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()];
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k];
  }
  out = Vectorized<float>::loadu(values);
}

inline void load_fp32_from_bf16(
    const c10::BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_bf16(data, out1);
  data += Vectorized<float>::size();
  load_fp32_from_bf16(data, out2);
}

} // namespace
} // namespace vec
} // namespace at
