/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/q8gemm.h>
#include <requantization/runtime-vsx.h>

void pytorch_q8gemm_ukernel_4x4c2__vsx(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  vector int vacc0x0123 = vec_xl(0, (const int*)w);
  vector int vacc1x0123 = vacc0x0123;
  vector int vacc2x0123 = vacc0x0123;
  vector int vacc3x0123 = vacc0x0123;

  w = (const void*)((uintptr_t)w + 16);

  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride);
  if (mr != 4) {
    a3 = a2;
  }

  const vector short va_zero_point =
      vec_splats(quantization_params->vsx.input_zero_point);

  const int16_t vb_zero_point_0 =
      (int16_t)(uint16_t)
          quantization_params->vsx.kernel_zero_points[output_channel_index];
  const int16_t vb_zero_point_1 =
      (int16_t)(uint16_t)
          quantization_params->vsx.kernel_zero_points[output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      (int16_t)(uint16_t)
          quantization_params->vsx.kernel_zero_points[output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      (int16_t)(uint16_t)
          quantization_params->vsx.kernel_zero_points[output_channel_index + 3];

  const vector short vb_zero_point = {
      vb_zero_point_0,
      vb_zero_point_0,
      vb_zero_point_1,
      vb_zero_point_1,
      vb_zero_point_2,
      vb_zero_point_2,
      vb_zero_point_3,
      vb_zero_point_3};

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char shift_w = {
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_01 = {
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector unsigned char mask_23 = {
      4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
  const vector unsigned char mask_45 = {
      8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11};
  const vector unsigned char mask_67 = {
      12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  for (; k >= 16; k -= 16) {
    const vector unsigned char va0 = vec_xl(0, a0);
    const vector short vxa0_hi =
        sub_zero_point((vector short)vec_mergeh(va0, vzero), va_zero_point);
    const vector short vxa0_lo =
        sub_zero_point((vector short)vec_mergel(va0, vzero), va_zero_point);
    a0 += 16;

    const vector unsigned char va1 = vec_xl(0, a1);
    const vector short vxa1_hi =
        sub_zero_point((vector short)vec_mergeh(va1, vzero), va_zero_point);
    const vector short vxa1_lo =
        sub_zero_point((vector short)vec_mergel(va1, vzero), va_zero_point);
    a1 += 16;

    const vector unsigned char va2 = vec_xl(0, a2);
    const vector short vxa2_hi =
        sub_zero_point((vector short)vec_mergeh(va2, vzero), va_zero_point);
    const vector short vxa2_lo =
        sub_zero_point((vector short)vec_mergel(va2, vzero), va_zero_point);
    a2 += 16;

    const vector unsigned char va3 = vec_xl(0, a3);
    const vector short vxa3_hi =
        sub_zero_point((vector short)vec_mergeh(va3, vzero), va_zero_point);
    const vector short vxa3_lo =
        sub_zero_point((vector short)vec_mergel(va3, vzero), va_zero_point);
    a3 += 16;

    const vector unsigned char vb01 = vec_xl(0, (const unsigned char*)w);
    const vector short vxb0 =
        vec_sub((vector short)vec_mergeh(vb01, vzero), vb_zero_point);
    const vector short vxb1 =
        vec_sub((vector short)vec_mergel(vb01, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 16);

    const vector short vxa0_hi_01 =
        (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_01);
    const vector short vxa1_hi_01 =
        (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_01);
    const vector short vxa2_hi_01 =
        (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_01);
    const vector short vxa3_hi_01 =
        (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_01);

    vacc0x0123 = vec_msum(vxa0_hi_01, vxb0, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_01, vxb0, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_01, vxb0, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_01, vxb0, vacc3x0123);

    const vector short vxa0_hi_23 =
        (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_23);
    const vector short vxa1_hi_23 =
        (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_23);
    const vector short vxa2_hi_23 =
        (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_23);
    const vector short vxa3_hi_23 =
        (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_23);

    vacc0x0123 = vec_msum(vxa0_hi_23, vxb1, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_23, vxb1, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_23, vxb1, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_23, vxb1, vacc3x0123);

    const vector unsigned char vb23 = vec_xl(0, (const unsigned char*)w);
    const vector short vxb2 =
        vec_sub((vector short)vec_mergeh(vb23, vzero), vb_zero_point);
    const vector short vxb3 =
        vec_sub((vector short)vec_mergel(vb23, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 16);

    const vector short vxa0_hi_45 =
        (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_45);
    const vector short vxa1_hi_45 =
        (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_45);
    const vector short vxa2_hi_45 =
        (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_45);
    const vector short vxa3_hi_45 =
        (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_45);

    vacc0x0123 = vec_msum(vxa0_hi_45, vxb2, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_45, vxb2, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_45, vxb2, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_45, vxb2, vacc3x0123);

    const vector short vxa0_hi_67 =
        (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_67);
    const vector short vxa1_hi_67 =
        (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_67);
    const vector short vxa2_hi_67 =
        (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_67);
    const vector short vxa3_hi_67 =
        (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_67);

    vacc0x0123 = vec_msum(vxa0_hi_67, vxb3, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_67, vxb3, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_67, vxb3, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_67, vxb3, vacc3x0123);

    const vector unsigned char vb45 = vec_xl(0, (const unsigned char*)w);
    const vector short vxb4 =
        vec_sub((vector short)vec_mergeh(vb45, vzero), vb_zero_point);
    const vector short vxb5 =
        vec_sub((vector short)vec_mergel(vb45, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 16);

    const vector short vxa0_lo_01 =
        (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_01);
    const vector short vxa1_lo_01 =
        (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_01);
    const vector short vxa2_lo_01 =
        (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_01);
    const vector short vxa3_lo_01 =
        (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_01);

    vacc0x0123 = vec_msum(vxa0_lo_01, vxb4, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_01, vxb4, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_01, vxb4, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_01, vxb4, vacc3x0123);

    const vector short vxa0_lo_23 =
        (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_23);
    const vector short vxa1_lo_23 =
        (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_23);
    const vector short vxa2_lo_23 =
        (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_23);
    const vector short vxa3_lo_23 =
        (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_23);

    vacc0x0123 = vec_msum(vxa0_lo_23, vxb5, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_23, vxb5, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_23, vxb5, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_23, vxb5, vacc3x0123);

    const vector unsigned char vb67 = vec_xl(0, (const unsigned char*)w);
    const vector short vxb6 =
        vec_sub((vector short)vec_mergeh(vb67, vzero), vb_zero_point);
    const vector short vxb7 =
        vec_sub((vector short)vec_mergel(vb67, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 16);

    const vector short vxa0_lo_45 =
        (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_45);
    const vector short vxa1_lo_45 =
        (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_45);
    const vector short vxa2_lo_45 =
        (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_45);
    const vector short vxa3_lo_45 =
        (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_45);

    vacc0x0123 = vec_msum(vxa0_lo_45, vxb6, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_45, vxb6, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_45, vxb6, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_45, vxb6, vacc3x0123);

    const vector short vxa0_lo_67 =
        (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_67);
    const vector short vxa1_lo_67 =
        (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_67);
    const vector short vxa2_lo_67 =
        (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_67);
    const vector short vxa3_lo_67 =
        (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_67);

    vacc0x0123 = vec_msum(vxa0_lo_67, vxb7, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_67, vxb7, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_67, vxb7, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_67, vxb7, vacc3x0123);
  }

  if (k != 0) {
    const size_t a_predecrement = 16 - k;
    const vector unsigned char va_shift = {
        8 * a_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // TODO(Rafael): Check if the accesses here are accessing valid memory
    // valgrind shows that it does not hvae leak of memory
    const vector unsigned char va0 =
        vec_sro(vec_xl(-a_predecrement, a0), va_shift);
    const vector short vxa0_hi =
        sub_zero_point((vector short)vec_mergeh(va0, vzero), va_zero_point);
    const vector short vxa0_lo =
        sub_zero_point((vector short)vec_mergel(va0, vzero), va_zero_point);

    const vector unsigned char va1 =
        vec_sro(vec_xl(-a_predecrement, a1), va_shift);
    const vector short vxa1_hi =
        sub_zero_point((vector short)vec_mergeh(va1, vzero), va_zero_point);
    const vector short vxa1_lo =
        sub_zero_point((vector short)vec_mergel(va1, vzero), va_zero_point);

    const vector unsigned char va2 =
        vec_sro(vec_xl(-a_predecrement, a2), va_shift);
    const vector short vxa2_hi =
        sub_zero_point((vector short)vec_mergeh(va2, vzero), va_zero_point);
    const vector short vxa2_lo =
        sub_zero_point((vector short)vec_mergel(va2, vzero), va_zero_point);

    const vector unsigned char va3 =
        vec_sro(vec_xl(-a_predecrement, a3), va_shift);
    const vector short vxa3_hi =
        sub_zero_point((vector short)vec_mergeh(va3, vzero), va_zero_point);
    const vector short vxa3_lo =
        sub_zero_point((vector short)vec_mergel(va3, vzero), va_zero_point);

    const vector unsigned char vb0 =
        vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
    const vector short vxb0 =
        vec_sub((vector short)vec_mergeh(vb0, vzero), vb_zero_point);

    const vector short vxa0_hi_01 =
        (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_01);
    const vector short vxa1_hi_01 =
        (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_01);
    const vector short vxa2_hi_01 =
        (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_01);
    const vector short vxa3_hi_01 =
        (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_01);

    vacc0x0123 = vec_msum(vxa0_hi_01, vxb0, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_01, vxb0, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_01, vxb0, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_01, vxb0, vacc3x0123);

    if (k > 2) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb1 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb1 =
          vec_sub((vector short)vec_mergeh(vb1, vzero), vb_zero_point);

      const vector short vxa0_hi_23 =
          (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_23);
      const vector short vxa1_hi_23 =
          (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_23);
      const vector short vxa2_hi_23 =
          (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_23);
      const vector short vxa3_hi_23 =
          (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_23);

      vacc0x0123 = vec_msum(vxa0_hi_23, vxb1, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_23, vxb1, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_23, vxb1, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_23, vxb1, vacc3x0123);
    }

    // Should be within the previous if stmt
    if (k > 4) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb2 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb2 =
          vec_sub((vector short)vec_mergeh(vb2, vzero), vb_zero_point);

      const vector short vxa0_hi_45 =
          (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_45);
      const vector short vxa1_hi_45 =
          (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_45);
      const vector short vxa2_hi_45 =
          (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_45);
      const vector short vxa3_hi_45 =
          (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_45);

      vacc0x0123 = vec_msum(vxa0_hi_45, vxb2, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_45, vxb2, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_45, vxb2, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_45, vxb2, vacc3x0123);
    }

    if (k > 6) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb3 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb3 =
          vec_sub((vector short)vec_mergeh(vb3, vzero), vb_zero_point);

      const vector short vxa0_hi_67 =
          (vector short)vec_perm(vxa0_hi, vxa0_hi, mask_67);
      const vector short vxa1_hi_67 =
          (vector short)vec_perm(vxa1_hi, vxa1_hi, mask_67);
      const vector short vxa2_hi_67 =
          (vector short)vec_perm(vxa2_hi, vxa2_hi, mask_67);
      const vector short vxa3_hi_67 =
          (vector short)vec_perm(vxa3_hi, vxa3_hi, mask_67);

      vacc0x0123 = vec_msum(vxa0_hi_67, vxb3, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_67, vxb3, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_67, vxb3, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_67, vxb3, vacc3x0123);
    }

    if (k > 8) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb4 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb4 =
          vec_sub((vector short)vec_mergeh(vb4, vzero), vb_zero_point);

      const vector short vxa0_lo_01 =
          (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_01);
      const vector short vxa1_lo_01 =
          (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_01);
      const vector short vxa2_lo_01 =
          (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_01);
      const vector short vxa3_lo_01 =
          (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_01);

      vacc0x0123 = vec_msum(vxa0_lo_01, vxb4, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_01, vxb4, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_01, vxb4, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_01, vxb4, vacc3x0123);
    }

    if (k > 10) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb5 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb5 =
          vec_sub((vector short)vec_mergeh(vb5, vzero), vb_zero_point);

      const vector short vxa0_lo_23 =
          (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_23);
      const vector short vxa1_lo_23 =
          (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_23);
      const vector short vxa2_lo_23 =
          (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_23);
      const vector short vxa3_lo_23 =
          (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_23);

      vacc0x0123 = vec_msum(vxa0_lo_23, vxb5, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_23, vxb5, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_23, vxb5, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_23, vxb5, vacc3x0123);
    }

    if (k > 12) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb6 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb6 =
          vec_sub((vector short)vec_mergeh(vb6, vzero), vb_zero_point);

      const vector short vxa0_lo_45 =
          (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_45);
      const vector short vxa1_lo_45 =
          (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_45);
      const vector short vxa2_lo_45 =
          (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_45);
      const vector short vxa3_lo_45 =
          (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_45);

      vacc0x0123 = vec_msum(vxa0_lo_45, vxb6, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_45, vxb6, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_45, vxb6, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_45, vxb6, vacc3x0123);
    }

    if (k > 14) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb7 =
          vec_sro(vec_xl(-8, (const unsigned char*)w), shift_w);
      const vector short vxb7 =
          vec_sub((vector short)vec_mergeh(vb7, vzero), vb_zero_point);

      const vector short vxa0_lo_67 =
          (vector short)vec_perm(vxa0_lo, vxa0_lo, mask_67);
      const vector short vxa1_lo_67 =
          (vector short)vec_perm(vxa1_lo, vxa1_lo, mask_67);
      const vector short vxa2_lo_67 =
          (vector short)vec_perm(vxa2_lo, vxa2_lo, mask_67);
      const vector short vxa3_lo_67 =
          (vector short)vec_perm(vxa3_lo, vxa3_lo, mask_67);

      vacc0x0123 = vec_msum(vxa0_lo_67, vxb7, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_67, vxb7, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_67, vxb7, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_67, vxb7, vacc3x0123);
    }
  }

  // Requantize
  const vector float vmultiplier = vec_xl(
      0, &quantization_params->vsx.requantization_scales[output_channel_index]);
  vacc0x0123 =
      vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc0x0123))));
  vacc1x0123 =
      vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc1x0123))));
  vacc2x0123 =
      vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc2x0123))));
  vacc3x0123 =
      vec_signed(vec_round(vec_mul(vmultiplier, vec_float(vacc3x0123))));

  const vector short voutput_zero_point =
      vec_splats(quantization_params->vsx.output_zero_point);

  const vector short vacc01x0123 =
      vec_add(vec_packs(vacc0x0123, vacc1x0123), voutput_zero_point);
  const vector short vacc23x0123 =
      vec_add(vec_packs(vacc2x0123, vacc3x0123), voutput_zero_point);

  vector unsigned char vout = vec_packsu(vacc01x0123, vacc23x0123);

  vector unsigned char vmin = vec_splats(quantization_params->vsx.output_min);
  vector unsigned char vmax = vec_splats(quantization_params->vsx.output_max);
  vout = vec_min(vout, vmax);
  vout = vec_max(vout, vmin);

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  if (mr != 4) {
    c3 = c2;
  }

  if (nr == 4) {
    *(uint32_t *)c0 = ((vector unsigned int)vout)[0];
    *(uint32_t *)c1 = ((vector unsigned int)vout)[1];
    *(uint32_t *)c2 = ((vector unsigned int)vout)[2];
    *(uint32_t *)c3 = ((vector unsigned int)vout)[3];
  } else {
    if (nr >= 2) {
      *(uint16_t *)c0 = ((vector unsigned short)vout)[0];
      *(uint16_t *)c1 = ((vector unsigned short)vout)[2];
      *(uint16_t *)c2 = ((vector unsigned short)vout)[4];
      *(uint16_t *)c3 = ((vector unsigned short)vout)[6];

      nr -= 2;
      if (nr != 0) {
        c0[2] = vout[2];
        c1[2] = vout[6];
        c2[2] = vout[10];
        c3[2] = vout[14];
      }
    } else {
      *c0 = vout[0];
      *c1 = vout[4];
      *c2 = vout[8];
      *c3 = vout[12];
    }
  }
}
