/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/q8avgpool.h>

void pytorch_q8avgpool_ukernel_up16x9__vsx(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(n != 0);
  assert(ks <= 9);
  assert(kc >= 16);

  const vector int vbias = vec_splats(quantization_params->vsx.bias);
  const vector float vscale = vec_splats(quantization_params->vsx.scale);
  const vector float vfmin = vec_splats(quantization_params->vsx.vfmin);
  const vector float vfmax = vec_splats(quantization_params->vsx.vfmax);
  const vector float vfmagic = vec_splats(quantization_params->vsx.vfmagic);
  const vector int vimagic = vec_splats(quantization_params->vsx.vimagic);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  do {
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    if (ks < 2) {
      i1 = zero;
    }
    if (ks <= 2) {
      i2 = zero;
    }
    if (ks < 4) {
      i3 = zero;
    }
    if (ks <= 4) {
      i4 = zero;
    }
    if (ks < 6) {
      i5 = zero;
    }
    if (ks <= 6) {
      i6 = zero;
    }
    if (ks < 8) {
      i7 = zero;
    }
    if (ks <= 8) {
      i8 = zero;
    }

    size_t k = kc;
    while (k >= 16) {
      const vector unsigned char vi0 = vec_xl(0, i0);
      i0 += 16;
      const vector unsigned char vi1 = vec_xl(0, i1);
      i1 += 16;
      const vector unsigned char vi2 = vec_xl(0, i2);
      i2 += 16;
      const vector unsigned char vi3 = vec_xl(0, i3);
      i3 += 16;
      const vector unsigned char vi4 = vec_xl(0, i4);
      i4 += 16;
      const vector unsigned char vi5 = vec_xl(0, i5);
      i5 += 16;
      const vector unsigned char vi6 = vec_xl(0, i6);
      i6 += 16;
      const vector unsigned char vi7 = vec_xl(0, i7);
      i7 += 16;
      const vector unsigned char vi8 = vec_xl(0, i8);
      i8 += 16;

      const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
      const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
      const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
      const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
      const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
      const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
      const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
      const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
      const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
      const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
      const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
      const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
      const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);
      const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);
      const vector short vxi7_hi = (vector short)vec_mergeh(vi7, vzero);
      const vector short vxi7_lo = (vector short)vec_mergel(vi7, vzero);
      const vector short vxi8_hi = (vector short)vec_mergeh(vi8, vzero);
      const vector short vxi8_lo = (vector short)vec_mergel(vi8, vzero);

      const vector short vsum018_hi =
          vec_add(vec_add(vxi0_hi, vxi1_hi), vxi8_hi);
      const vector short vsum018_lo =
          vec_add(vec_add(vxi0_lo, vxi1_lo), vxi8_lo);
      const vector short vsum23_hi = vec_add(vxi2_hi, vxi3_hi);
      const vector short vsum23_lo = vec_add(vxi2_lo, vxi3_lo);
      const vector short vsum45_hi = vec_add(vxi4_hi, vxi5_hi);
      const vector short vsum45_lo = vec_add(vxi4_lo, vxi5_lo);
      const vector short vsum67_hi = vec_add(vxi6_hi, vxi7_hi);
      const vector short vsum67_lo = vec_add(vxi6_lo, vxi7_lo);

      const vector short vsum2345_hi = vec_add(vsum23_hi, vsum45_hi);
      const vector short vsum2345_lo = vec_add(vsum23_lo, vsum45_lo);
      const vector short vsum01678_hi = vec_add(vsum018_hi, vsum67_hi);
      const vector short vsum01678_lo = vec_add(vsum018_lo, vsum67_lo);
      const vector short vsum_hi = vec_add(vsum2345_hi, vsum01678_hi);
      const vector short vsum_lo = vec_add(vsum2345_lo, vsum01678_lo);

      vector int vacc_hi_hi = vec_add(vbias, vec_unpackh(vsum_hi));
      vector int vacc_hi_lo = vec_add(vbias, vec_unpackl(vsum_hi));
      vector int vacc_lo_hi = vec_add(vbias, vec_unpackh(vsum_lo));
      vector int vacc_lo_lo = vec_add(vbias, vec_unpackl(vsum_lo));

      vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
      vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);
      vector float vacc_lo_hi_f = vec_mul(vec_float(vacc_lo_hi), vscale);
      vector float vacc_lo_lo_f = vec_mul(vec_float(vacc_lo_lo), vscale);

      vacc_hi_hi_f = vec_min(vec_max(vacc_hi_hi_f, vfmin), vfmax);
      vacc_hi_lo_f = vec_min(vec_max(vacc_hi_lo_f, vfmin), vfmax);
      vacc_lo_hi_f = vec_min(vec_max(vacc_lo_hi_f, vfmin), vfmax);
      vacc_lo_lo_f = vec_min(vec_max(vacc_lo_lo_f, vfmin), vfmax);

      vacc_hi_hi =
          vec_sub((vector int)(vec_add(vacc_hi_hi_f, vfmagic)), vimagic);
      vacc_hi_lo =
          vec_sub((vector int)(vec_add(vacc_hi_lo_f, vfmagic)), vimagic);
      vacc_lo_hi =
          vec_sub((vector int)(vec_add(vacc_lo_hi_f, vfmagic)), vimagic);
      vacc_lo_lo =
          vec_sub((vector int)(vec_add(vacc_lo_lo_f, vfmagic)), vimagic);

      const vector short vout_hi = vec_packs(vacc_hi_hi, vacc_hi_lo);
      const vector short vout_lo = vec_packs(vacc_lo_hi, vacc_lo_lo);

      const vector unsigned char vout = vec_packsu(vout_hi, vout_lo);

      vec_xst(vout, 0, output);
      output += 16;

      k -= 16;
    }

    if (k != 0) {
      const size_t address_decrement = 16 - k;
      i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
      i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
      i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
      i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
      i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
      i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
      i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);
      i7 = (const uint8_t*)((uintptr_t)i7 - address_decrement);
      i8 = (const uint8_t*)((uintptr_t)i8 - address_decrement);
      const vector unsigned char vshift = {
        8 * address_decrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      const vector unsigned char vi0 = vec_sro(vec_xl(0, i0), vshift);
      const vector unsigned char vi1 = vec_sro(vec_xl(0, i1), vshift);
      const vector unsigned char vi2 = vec_sro(vec_xl(0, i2), vshift);
      const vector unsigned char vi3 = vec_sro(vec_xl(0, i3), vshift);
      const vector unsigned char vi4 = vec_sro(vec_xl(0, i4), vshift);
      const vector unsigned char vi5 = vec_sro(vec_xl(0, i5), vshift);
      const vector unsigned char vi6 = vec_sro(vec_xl(0, i6), vshift);
      const vector unsigned char vi7 = vec_sro(vec_xl(0, i7), vshift);
      const vector unsigned char vi8 = vec_sro(vec_xl(0, i8), vshift);

      const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
      const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
      const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
      const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
      const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
      const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
      const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);
      const vector short vxi7_hi = (vector short)vec_mergeh(vi7, vzero);
      const vector short vxi8_hi = (vector short)vec_mergeh(vi8, vzero);

      const vector short vsum018_hi =
          vec_add(vec_add(vxi0_hi, vxi1_hi), vxi8_hi);
      const vector short vsum23_hi = vec_add(vxi2_hi, vxi3_hi);
      const vector short vsum45_hi = vec_add(vxi4_hi, vxi5_hi);
      const vector short vsum67_hi = vec_add(vxi6_hi, vxi7_hi);

      const vector short vsum2345_hi = vec_add(vsum23_hi, vsum45_hi);
      const vector short vsum01678_hi = vec_add(vsum018_hi, vsum67_hi);
      const vector short vsum_hi = vec_add(vsum2345_hi, vsum01678_hi);

      vector int vacc_hi_hi = vec_add(vbias, vec_unpackh(vsum_hi));
      vector int vacc_hi_lo = vec_add(vbias, vec_unpackl(vsum_hi));

      vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
      vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);
      vacc_hi_hi_f = vec_min(vec_max(vacc_hi_hi_f, vfmin), vfmax);
      vacc_hi_lo_f = vec_min(vec_max(vacc_hi_lo_f, vfmin), vfmax);

      vacc_hi_hi =
          vec_sub((vector int)(vec_add(vacc_hi_hi_f, vfmagic)), vimagic);
      vacc_hi_lo =
          vec_sub((vector int)(vec_add(vacc_hi_lo_f, vfmagic)), vimagic);

      const vector short vout_hi = vec_packs(vacc_hi_hi, vacc_hi_lo);

      vector unsigned char vout;
      if (k > 8) {
        const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
        const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
        const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
        const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
        const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
        const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
        const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);
        const vector short vxi7_lo = (vector short)vec_mergel(vi7, vzero);
        const vector short vxi8_lo = (vector short)vec_mergel(vi8, vzero);

        const vector short vsum018_lo =
          vec_add(vec_add(vxi0_lo, vxi1_lo), vxi8_lo);
        const vector short vsum23_lo = vec_add(vxi2_lo, vxi3_lo);
        const vector short vsum45_lo = vec_add(vxi4_lo, vxi5_lo);
        const vector short vsum67_lo = vec_add(vxi6_lo, vxi7_lo);

        const vector short vsum2345_lo = vec_add(vsum23_lo, vsum45_lo);
        const vector short vsum01678_lo = vec_add(vsum018_lo, vsum67_lo);
        const vector short vsum_lo = vec_add(vsum2345_lo, vsum01678_lo);

        vector int vacc_lo_hi = vec_add(vbias, vec_unpackh(vsum_lo));
        vector int vacc_lo_lo = vec_add(vbias, vec_unpackl(vsum_lo));

        vector float vacc_lo_hi_f = vec_mul(vec_float(vacc_lo_hi), vscale);
        vector float vacc_lo_lo_f = vec_mul(vec_float(vacc_lo_lo), vscale);
        vacc_lo_hi_f = vec_min(vec_max(vacc_lo_hi_f, vfmin), vfmax);
        vacc_lo_lo_f = vec_min(vec_max(vacc_lo_lo_f, vfmin), vfmax);

        vacc_lo_hi =
            vec_sub((vector int)(vec_add(vacc_lo_hi_f, vfmagic)), vimagic);
        vacc_lo_lo =
            vec_sub((vector int)(vec_add(vacc_lo_lo_f, vfmagic)), vimagic);

        const vector short vout_lo = vec_packs(vacc_lo_hi, vacc_lo_lo);

        vout = vec_packsu(vout_hi, vout_lo);

      } else {
        vout = vec_packsu(vout_hi, vout_hi);
      }

      if (k & 8) {
        output[0] = vout[0];
        output[1] = vout[1];
        output[2] = vout[2];
        output[3] = vout[3];
        output[4] = vout[4];
        output[5] = vout[5];
        output[6] = vout[6];
        output[7] = vout[7];
        output += 8;
        const vector unsigned char vshift2 = {
          8 * 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        vout = vec_sro(vout, vshift2);
      }
      if (k & 4) {
        output[0] = vout[0];
        output[1] = vout[1];
        output[2] = vout[2];
        output[3] = vout[3];
        output += 4;
        const vector unsigned char vshift2 = {
          8 * 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        vout = vec_sro(vout, vshift2);
      }
      if (k & 2) {
        output[0] = vout[0];
        output[1] = vout[1];
        output += 2;
        const vector unsigned char vshift2 = {
          8 * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        vout = vec_sro(vout, vshift2);
      }
      if (k & 1) {
        output[0] = vout[0];
        output += 1;
      }
    }

    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}
