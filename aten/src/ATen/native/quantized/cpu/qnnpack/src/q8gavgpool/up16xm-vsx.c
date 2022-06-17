/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_up16xm__vsx(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m >= 1);
  assert(n < 16);

  const vector int vbias = vec_splats(quantization_params->vsx.bias);
  const vector float vscale = vec_splats(quantization_params->vsx.scale);
  const vector float vfmin = vec_splats(quantization_params->vsx.vfmin);
  const vector float vfmax = vec_splats(quantization_params->vsx.vfmax);
  const vector float vfmagic = vec_splats(quantization_params->vsx.vfmagic);
  const vector int vimagic = vec_splats(quantization_params->vsx.vimagic);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_2bytes = {
      16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_4bytes = {
      32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_8bytes = {
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  vector int vacc_hi_hi = vbias;
  vector int vacc_hi_lo = vbias;
  vector int vacc_lo_hi, vacc_lo_lo;
  if (n >= 8) {
    vacc_lo_hi = vbias;
    vacc_lo_lo = vbias;
  }

  while (m-- != 0) {
    input += n;
    vector unsigned char vinput = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (n & 1) {
      input -= 1;
      vinput = vec_insert(*input, vinput, 0);
    }
    if (n & 2) {
      input -= 2;
      vinput = vec_slo(vinput, mask_shift_2bytes);
      vinput = (vector unsigned char)vec_insert(
          *(uint16_t*)input, (vector unsigned short)vinput, 0);
    }
    if (n & 4) {
      input -= 4;
      vinput = vec_slo(vinput, mask_shift_4bytes);
      vinput = (vector unsigned char)vec_insert(
          *(uint32_t*)input, (vector unsigned int)vinput, 0);
    }
    if (n & 8) {
      input -= 8;
      vinput = vec_slo(vinput, mask_shift_8bytes);
      vinput = (vector unsigned char)vec_insert(
          *(uint64_t*)input, (vector unsigned long long)vinput, 0);

      // Compute the lower part of the vector register vinput
      const vector short vxi_lo = (vector short)vec_mergel(vinput, vzero);
      vacc_lo_hi = vec_add(
          vacc_lo_hi, (vector int)vec_mergeh(vxi_lo, (vector short)vzero));
      vacc_lo_lo = vec_add(
          vacc_lo_lo, (vector int)vec_mergel(vxi_lo, (vector short)vzero));
    }
    // Compute the higher part of the vector register vinput
    const vector short vxi_hi = (vector short)vec_mergeh(vinput, vzero);
    vacc_hi_hi = vec_add(
        vacc_hi_hi, (vector int)vec_mergeh(vxi_hi, (vector short)vzero));
    vacc_hi_lo = vec_add(
        vacc_hi_lo, (vector int)vec_mergel(vxi_hi, (vector short)vzero));

    input += input_stride;
  }

  vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
  vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);
  vacc_hi_hi_f = vec_min(vec_max(vacc_hi_hi_f, vfmin), vfmax);
  vacc_hi_lo_f = vec_min(vec_max(vacc_hi_lo_f, vfmin), vfmax);
  vacc_hi_hi = vec_sub((vector int)(vec_add(vacc_hi_hi_f, vfmagic)), vimagic);
  vacc_hi_lo = vec_sub((vector int)(vec_add(vacc_hi_lo_f, vfmagic)), vimagic);
  const vector short vout_hi = vec_packs(vacc_hi_hi, vacc_hi_lo);

  vector unsigned char vout;
  if (n > 8) {
    vector float vacc_lo_hi_f = vec_mul(vec_float(vacc_lo_hi), vscale);
    vector float vacc_lo_lo_f = vec_mul(vec_float(vacc_lo_lo), vscale);
    vacc_lo_hi_f = vec_min(vec_max(vacc_lo_hi_f, vfmin), vfmax);
    vacc_lo_lo_f = vec_min(vec_max(vacc_lo_lo_f, vfmin), vfmax);
    vacc_lo_hi = vec_sub((vector int)(vec_add(vacc_lo_hi_f, vfmagic)), vimagic);
    vacc_lo_lo = vec_sub((vector int)(vec_add(vacc_lo_lo_f, vfmagic)), vimagic);
    const vector short vout_lo = vec_packs(vacc_lo_hi, vacc_lo_lo);
    vout = vec_packsu(vout_hi, vout_lo);
  } else {
    vout = vec_packsu(vout_hi, vout_hi);
  }

  if (n & 8) {
    *((uint64_t*)output) = ((vector unsigned long long)vout)[0];
    vout = vec_sro(vout, mask_shift_8bytes);
    output += 8;
  }
  if (n & 4) {
    *((uint32_t*)output) = ((vector unsigned int)vout)[0];
    vout = vec_sro(vout, mask_shift_4bytes);
    output += 4;
  }
  if (n & 2) {
    *((uint16_t*)output) = ((vector unsigned short)vout)[0];
    vout = vec_sro(vout, mask_shift_2bytes);
    output += 2;
  }
  if (n & 1) {
    output[0] = vout[0];
  }
}
