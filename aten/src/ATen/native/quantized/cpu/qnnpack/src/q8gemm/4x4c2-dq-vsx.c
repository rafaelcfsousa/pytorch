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
#include <stdio.h>

void pytorch_q8gemm_dq_ukernel_4x4c2__vsx(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    const float* restrict b,
    float* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {

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

  vector signed int vacc0x0123 = {0, 0, 0, 0};
  vector signed int vacc1x0123 = {0, 0, 0, 0};
  vector signed int vacc2x0123 = {0, 0, 0, 0};
  vector signed int vacc3x0123 = {0, 0, 0, 0};

  const vector signed short va_zero_point =
    vec_splats(quantization_params->input_zero_point);
  const int16_t vb_zero_point_0 =
    (int16_t)(uint16_t)quantization_params->kernel_zero_points[
    output_channel_index];
  const int16_t vb_zero_point_1 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
        output_channel_index + 3];

  vector signed short vb_zero_point = {vb_zero_point_0, vb_zero_point_0,
                                       vb_zero_point_1, vb_zero_point_1,
                                       vb_zero_point_2, vb_zero_point_2,
                                       vb_zero_point_3, vb_zero_point_3};
  const vector unsigned char vzero =
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char shift_w =
      {64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_01 =
      {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector unsigned char mask_23 =
      {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
  const vector unsigned char mask_45 =
      {8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11};
  const vector unsigned char mask_67 =
      {12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  for (; k >= 16; k -= 16) {
    const vector unsigned char va0 = vec_xl(0, a0);
    const vector signed short vxa0_hi =
        sub_zero_point((vector signed short)vec_mergeh(va0, vzero),
        va_zero_point);
    const vector signed short vxa0_lo =
        sub_zero_point((vector signed short)vec_mergel(va0, vzero),
        va_zero_point);
    a0 += 16;

    const vector unsigned char va1 = (vector unsigned char)vec_xl(0, a1);
    const vector signed short vxa1_hi =
        sub_zero_point((vector signed short)vec_mergeh(va1, vzero),
        va_zero_point);
    const vector signed short vxa1_lo =
        sub_zero_point((vector signed short)vec_mergel(va1, vzero),
        va_zero_point);
    a1 += 16;

    const vector unsigned char va2 = (vector unsigned char)vec_xl(0, a2);
    const vector signed short vxa2_hi =
        sub_zero_point((vector signed short)vec_mergeh(va2, vzero),
        va_zero_point);
    const vector signed short vxa2_lo =
        sub_zero_point((vector signed short)vec_mergel(va2, vzero),
        va_zero_point);
    a2 += 16;

    const vector unsigned char va3 = (vector unsigned char)vec_xl(0, a3);
    const vector signed short vxa3_hi =
        sub_zero_point((vector signed short)vec_mergeh(va3, vzero),
        va_zero_point);
    const vector signed short vxa3_lo =
        sub_zero_point((vector signed short)vec_mergel(va3, vzero),
        va_zero_point);
    a3 += 16;

    const vector unsigned char vb0 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb0 =
        vec_sub((vector signed short)vec_mergeh(vb0, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_hi_01 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_01);
    const vector signed short vxa1_hi_01 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_01);
    const vector signed short vxa2_hi_01 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_01);
    const vector signed short vxa3_hi_01 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_01);

    vacc0x0123 = vec_msum(vxa0_hi_01, vxb0, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_01, vxb0, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_01, vxb0, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_01, vxb0, vacc3x0123);

    const vector unsigned char vb1 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb1 =
        vec_sub((vector signed short)vec_mergeh(vb1, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_hi_23 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_23);
    const vector signed short vxa1_hi_23 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_23);
    const vector signed short vxa2_hi_23 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_23);
    const vector signed short vxa3_hi_23 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_23);

    vacc0x0123 = vec_msum(vxa0_hi_23, vxb1, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_23, vxb1, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_23, vxb1, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_23, vxb1, vacc3x0123);

    const vector unsigned char vb2 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb2 =
        vec_sub((vector signed short)vec_mergeh(vb2, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_hi_45 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_45);
    const vector signed short vxa1_hi_45 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_45);
    const vector signed short vxa2_hi_45 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_45);
    const vector signed short vxa3_hi_45 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_45);

    vacc0x0123 = vec_msum(vxa0_hi_45, vxb2, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_45, vxb2, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_45, vxb2, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_45, vxb2, vacc3x0123);

    const vector unsigned char vb3 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb3 =
        vec_sub((vector signed short)vec_mergeh(vb3, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_hi_67 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_67);
    const vector signed short vxa1_hi_67 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_67);
    const vector signed short vxa2_hi_67 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_67);
    const vector signed short vxa3_hi_67 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_67);

    vacc0x0123 = vec_msum(vxa0_hi_67, vxb3, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_67, vxb3, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_67, vxb3, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_67, vxb3, vacc3x0123);

    const vector unsigned char vb4 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb4 =
        vec_sub((vector signed short)vec_mergeh(vb4, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_lo_01 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_01);
    const vector signed short vxa1_lo_01 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_01);
    const vector signed short vxa2_lo_01 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_01);
    const vector signed short vxa3_lo_01 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_01);

    vacc0x0123 = vec_msum(vxa0_lo_01, vxb4, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_01, vxb4, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_01, vxb4, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_01, vxb4, vacc3x0123);

    const vector unsigned char vb5 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb5 =
        vec_sub((vector signed short)vec_mergeh(vb5, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_lo_23 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_23);
    const vector signed short vxa1_lo_23 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_23);
    const vector signed short vxa2_lo_23 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_23);
    const vector signed short vxa3_lo_23 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_23);

    vacc0x0123 = vec_msum(vxa0_lo_23, vxb5, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_23, vxb5, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_23, vxb5, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_23, vxb5, vacc3x0123);

    const vector unsigned char vb6 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb6 =
        vec_sub((vector signed short)vec_mergeh(vb6, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_lo_45 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_45);
    const vector signed short vxa1_lo_45 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_45);
    const vector signed short vxa2_lo_45 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_45);
    const vector signed short vxa3_lo_45 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_45);

    vacc0x0123 = vec_msum(vxa0_lo_45, vxb6, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_45, vxb6, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_45, vxb6, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_45, vxb6, vacc3x0123);

    const vector unsigned char vb7 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb7 =
        vec_sub((vector signed short)vec_mergeh(vb7, vzero), vb_zero_point);
    w = (const void*)((uintptr_t)w + 8);

    const vector signed short vxa0_lo_67 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_67);
    const vector signed short vxa1_lo_67 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_67);
    const vector signed short vxa2_lo_67 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_67);
    const vector signed short vxa3_lo_67 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_67);

    vacc0x0123 = vec_msum(vxa0_lo_67, vxb7, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_lo_67, vxb7, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_lo_67, vxb7, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_lo_67, vxb7, vacc3x0123);
  }

  if (k != 0) {
    const size_t a_predecrement = 16 - k;
    const vector unsigned char va_shift =
        {8 * a_predecrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const vector unsigned char va0 =
        vec_sro(vec_xl(-a_predecrement, a0), va_shift);
    const vector signed short vxa0_hi =
        sub_zero_point((vector signed short)vec_mergeh(va0, vzero),
        va_zero_point);
    const vector signed short vxa0_lo =
        sub_zero_point((vector signed short)vec_mergel(va0, vzero),
        va_zero_point);

    const vector unsigned char va1 =
        vec_sro(vec_xl(-a_predecrement, a1), va_shift);
    const vector signed short vxa1_hi =
        sub_zero_point((vector signed short)vec_mergeh(va1, vzero),
        va_zero_point);
    const vector signed short vxa1_lo =
        sub_zero_point((vector signed short)vec_mergel(va1, vzero),
        va_zero_point);

    const vector unsigned char va2 =
        vec_sro(vec_xl(-a_predecrement, a2), va_shift);
    const vector signed short vxa2_hi =
        sub_zero_point((vector signed short)vec_mergeh(va2, vzero),
        va_zero_point);
    const vector signed short vxa2_lo =
        sub_zero_point((vector signed short)vec_mergel(va2, vzero),
        va_zero_point);

    const vector unsigned char va3 =
        vec_sro(vec_xl(-a_predecrement, a3), va_shift);
    const vector signed short vxa3_hi =
        sub_zero_point((vector signed short)vec_mergeh(va3, vzero),
        va_zero_point);
    const vector signed short vxa3_lo =
        sub_zero_point((vector signed short)vec_mergel(va3, vzero),
        va_zero_point);

    const vector unsigned char vb0 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
    const vector signed short vxb0 =
        vec_sub((vector signed short)vec_mergeh(vb0, vzero), vb_zero_point);

    const vector signed short vxa0_hi_01 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_01);
    const vector signed short vxa1_hi_01 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_01);
    const vector signed short vxa2_hi_01 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_01);
    const vector signed short vxa3_hi_01 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_01);

    vacc0x0123 = vec_msum(vxa0_hi_01, vxb0, vacc0x0123);
    vacc1x0123 = vec_msum(vxa1_hi_01, vxb0, vacc1x0123);
    vacc2x0123 = vec_msum(vxa2_hi_01, vxb0, vacc2x0123);
    vacc3x0123 = vec_msum(vxa3_hi_01, vxb0, vacc3x0123);

    if (k > 2) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb1 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb1 =
        vec_sub((vector signed short)vec_mergeh(vb1, vzero), vb_zero_point);

      const vector signed short vxa0_hi_23 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_23);
      const vector signed short vxa1_hi_23 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_23);
      const vector signed short vxa2_hi_23 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_23);
      const vector signed short vxa3_hi_23 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_23);

      vacc0x0123 = vec_msum(vxa0_hi_23, vxb1, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_23, vxb1, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_23, vxb1, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_23, vxb1, vacc3x0123);
    }

    // Should be within the previous if stmt
    if (k > 4) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb2 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb2 =
        vec_sub((vector signed short)vec_mergeh(vb2, vzero), vb_zero_point);

      const vector signed short vxa0_hi_45 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_45);
      const vector signed short vxa1_hi_45 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_45);
      const vector signed short vxa2_hi_45 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_45);
      const vector signed short vxa3_hi_45 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_45);

      vacc0x0123 = vec_msum(vxa0_hi_45, vxb2, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_45, vxb2, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_45, vxb2, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_45, vxb2, vacc3x0123);
    }

    if (k > 6) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb3 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb3 =
        vec_sub((vector signed short)vec_mergeh(vb3, vzero), vb_zero_point);

      const vector signed short vxa0_hi_67 =
        (vector signed short)vec_perm(vxa0_hi, vxa0_hi, mask_67);
      const vector signed short vxa1_hi_67 =
        (vector signed short)vec_perm(vxa1_hi, vxa1_hi, mask_67);
      const vector signed short vxa2_hi_67 =
        (vector signed short)vec_perm(vxa2_hi, vxa2_hi, mask_67);
      const vector signed short vxa3_hi_67 =
        (vector signed short)vec_perm(vxa3_hi, vxa3_hi, mask_67);

      vacc0x0123 = vec_msum(vxa0_hi_67, vxb3, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_hi_67, vxb3, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_hi_67, vxb3, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_hi_67, vxb3, vacc3x0123);
    }

    if (k > 8) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb4 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb4 =
        vec_sub((vector signed short)vec_mergeh(vb4, vzero), vb_zero_point);

      const vector signed short vxa0_lo_01 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_01);
      const vector signed short vxa1_lo_01 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_01);
      const vector signed short vxa2_lo_01 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_01);
      const vector signed short vxa3_lo_01 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_01);

      vacc0x0123 = vec_msum(vxa0_lo_01, vxb4, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_01, vxb4, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_01, vxb4, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_01, vxb4, vacc3x0123);
    }

    if (k > 10) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb5 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb5 =
        vec_sub((vector signed short)vec_mergeh(vb5, vzero), vb_zero_point);

      const vector signed short vxa0_lo_23 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_23);
      const vector signed short vxa1_lo_23 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_23);
      const vector signed short vxa2_lo_23 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_23);
      const vector signed short vxa3_lo_23 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_23);

      vacc0x0123 = vec_msum(vxa0_lo_23, vxb5, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_23, vxb5, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_23, vxb5, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_23, vxb5, vacc3x0123);
    }

    if (k > 12) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb6 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb6 =
        vec_sub((vector signed short)vec_mergeh(vb6, vzero), vb_zero_point);

      const vector signed short vxa0_lo_45 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_45);
      const vector signed short vxa1_lo_45 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_45);
      const vector signed short vxa2_lo_45 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_45);
      const vector signed short vxa3_lo_45 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_45);

      vacc0x0123 = vec_msum(vxa0_lo_45, vxb6, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_45, vxb6, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_45, vxb6, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_45, vxb6, vacc3x0123);
    }

    if (k > 14) {
      w = (const void*)((uintptr_t)w + 8);
      const vector unsigned char vb7 =
        vec_sro(vec_xl(-8, (const unsigned char *)w), shift_w);
      const vector signed short vxb7 =
        vec_sub((vector signed short)vec_mergeh(vb7, vzero), vb_zero_point);

      const vector signed short vxa0_lo_67 =
        (vector signed short)vec_perm(vxa0_lo, vxa0_lo, mask_67);
      const vector signed short vxa1_lo_67 =
        (vector signed short)vec_perm(vxa1_lo, vxa1_lo, mask_67);
      const vector signed short vxa2_lo_67 =
        (vector signed short)vec_perm(vxa2_lo, vxa2_lo, mask_67);
      const vector signed short vxa3_lo_67 =
        (vector signed short)vec_perm(vxa3_lo, vxa3_lo, mask_67);

      vacc0x0123 = vec_msum(vxa0_lo_67, vxb7, vacc0x0123);
      vacc1x0123 = vec_msum(vxa1_lo_67, vxb7, vacc1x0123);
      vacc2x0123 = vec_msum(vxa2_lo_67, vxb7, vacc2x0123);
      vacc3x0123 = vec_msum(vxa3_lo_67, vxb7, vacc3x0123);
    }
  }

  const vector float vmultiplier =
      vec_xl(0, &quantization_params->multipliers[output_channel_index]);
  vector float vout0 = vec_mul(vmultiplier, vec_float(vacc0x0123));
  vector float vout1 = vec_mul(vmultiplier, vec_float(vacc1x0123));
  vector float vout2 = vec_mul(vmultiplier, vec_float(vacc2x0123));
  vector float vout3 = vec_mul(vmultiplier, vec_float(vacc3x0123));

  const vector float vbias = vec_xl(0, b);
  vout0 = vec_add(vout0, vbias);
  vout1 = vec_add(vout1, vbias);
  vout2 = vec_add(vout2, vbias);
  vout3 = vec_add(vout3, vbias);

  float* c0 = c;
  float* c1 = c0 + c_stride;
  if (mr < 2) {
    c1 = c0;
  }
  float* c2 = c1 + c_stride;
  if (mr <= 2) {
    c2 = c1;
  }
  float* c3 = c2 + c_stride;
  if (mr != 4) {
    c3 = c2;
  }

  if (nr == 4) {
    vec_xst(vout0, 0, c0);
    vec_xst(vout1, 0, c1);
    vec_xst(vout2, 0, c2);
    vec_xst(vout3, 0, c3);
  } else {
    if (nr >= 2) {
      c0[0] = vout0[0]; c0[1] = vout0[1];
      c1[0] = vout1[0]; c1[1] = vout1[1];
      c2[0] = vout2[0]; c2[1] = vout2[1];
      c3[0] = vout3[0]; c3[1] = vout3[1];

      nr -= 2;
      if (nr != 0) {
        c0[2] = vout0[2];
        c1[2] = vout1[2];
        c2[2] = vout2[2];
        c3[2] = vout3[2];
      }
    } else {
      *c0 = vout0[0];
      *c1 = vout1[0];
      *c2 = vout2[0];
      *c3 = vout3[0];
    }
  }
}
