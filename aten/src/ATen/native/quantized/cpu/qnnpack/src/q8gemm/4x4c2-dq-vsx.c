/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/q8gemm.h>
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

  w = (const void*)((uintptr_t)w + 4);
  uint8_t * w1 = (int8_t *)w;

  int16_t zero_point_a = quantization_params->input_zero_point;
  int16_t zero_point_b = (int16_t)(uint16_t)quantization_params->kernel_zero_points[output_channel_index];
  float scale = (float)quantization_params->multipliers[output_channel_index];

  int32_t resi = 0;
  for (int i = 0; i < k; i++) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
    resi += (int32_t)((int16_t)a[i] - zero_point_a) * ((int16_t)w1[i] - zero_point_b);
#else
    resi += (int32_t)((int16_t)a[i] * (int16_t)w1[i]);
#endif
  }

  float resf = scale * (float)resi;
  resf += *b;
  *c = resf;
}
