#pragma once
#include <immintrin.h>  // For AVX/SIMD instructions
#include "hnswlib.h"

namespace hnswlib {

// Define a mask for clearing the sign bit (absolute value)
const __m128 ABS_MASK_SSE = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
const __m256 ABS_MASK_AVX = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
// const __m512 ABS_MASK_AVX512 = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

// Scalar L1 distance function
static float L1Dist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += fabs(*pVect1 - *pVect2);
    pVect1++;
    pVect2++;
  }
  return res;
}

// Scalar L1 distance function with mask
static float L1DistMasked(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *mask_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  float *mask = (float *)mask_ptr;
  size_t qty = *((size_t *)qty_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += fabs(pVect1[i] - pVect2[i]) * mask[i];
  }
  return res;
}

#if defined(USE_AVX512)

// AVX512 L1 distance function
static float L1DistSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN64 TmpRes[16];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m512 diff, v1, v2;
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    diff = _mm512_sub_ps(v1, v2);
    diff = _mm512_and_ps(diff, ABS_MASK_AVX512);  // Use absolute value
    sum = _mm512_add_ps(sum, diff);
  }

  _mm512_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 16; i++) {
    res += TmpRes[i];
  }

  return res;
}

// AVX512 L1 distance function with mask
static float
L1DistMaskedSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *mask_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  float *mask = (float *)mask_ptr;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN64 TmpRes[16];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m512 diff, v1, v2, vMask;
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    vMask = _mm512_loadu_ps(mask);
    mask += 16;
    diff = _mm512_sub_ps(v1, v2);
    diff = _mm512_and_ps(diff, ABS_MASK_AVX512);  // Use absolute value
    diff = _mm512_mul_ps(diff, vMask);  // Apply mask
    sum = _mm512_add_ps(sum, diff);
  }

  _mm512_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 16; i++) {
    res += TmpRes[i];
  }

  return res;
}
#endif

#if defined(USE_AVX)

// AVX L1 distance function
static float L1DistSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    diff = _mm256_and_ps(diff, ABS_MASK_AVX);  // Use absolute value
    sum = _mm256_add_ps(sum, diff);

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    diff = _mm256_and_ps(diff, ABS_MASK_AVX);  // Use absolute value
    sum = _mm256_add_ps(sum, diff);
  }

  _mm256_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 8; i++) {
    res += TmpRes[i];
  }

  return res;
}

// AVX L1 distance function with mask
static float
L1DistMaskedSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *mask_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  float *mask = (float *)mask_ptr;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2, vMask;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    vMask = _mm256_loadu_ps(mask);
    mask += 8;
    diff = _mm256_sub_ps(v1, v2);
    diff = _mm256_and_ps(diff, ABS_MASK_AVX);  // Use absolute value
    diff = _mm256_mul_ps(diff, vMask);  // Apply mask
    sum = _mm256_add_ps(sum, diff);

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    vMask = _mm256_loadu_ps(mask);
    mask += 8;
    diff = _mm256_sub_ps(v1, v2);
    diff = _mm256_and_ps(diff, ABS_MASK_AVX);  // Use absolute value
    diff = _mm256_mul_ps(diff, vMask);  // Apply mask
    sum = _mm256_add_ps(sum, diff);
  }

  _mm256_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 8; i++) {
    res += TmpRes[i];
  }

  return res;
}
#endif

#if defined(USE_SSE)

// SSE L1 distance function
static float L1DistSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[4];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    sum = _mm_add_ps(sum, diff);
  }

  _mm_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 4; i++) {
    res += TmpRes[i];
  }

  return res;
}

// SSE L1 distance function with mask
static float
L1DistMaskedSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *mask_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  float *mask = (float *)mask_ptr;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[4];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m128 diff, v1, v2, vMask;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    vMask = _mm_loadu_ps(mask);
    mask += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    diff = _mm_mul_ps(diff, vMask);  // Apply mask
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    vMask = _mm_loadu_ps(mask);
    mask += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    diff = _mm_mul_ps(diff, vMask);  // Apply mask
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    vMask = _mm_loadu_ps(mask);
    mask += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    diff = _mm_mul_ps(diff, vMask);  // Apply mask
    sum = _mm_add_ps(sum, diff);

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    vMask = _mm_loadu_ps(mask);
    mask += 4;
    diff = _mm_sub_ps(v1, v2);
    diff = _mm_and_ps(diff, ABS_MASK_SSE);  // Use absolute value
    diff = _mm_mul_ps(diff, vMask);  // Apply mask
    sum = _mm_add_ps(sum, diff);
  }

  _mm_store_ps(TmpRes, sum);
  float res = 0;
  for (int i = 0; i < 4; i++) {
    res += TmpRes[i];
  }

  return res;
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L1DistSIMD16Ext = L1DistSIMD16ExtSSE;

static float L1DistSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4 << 4;
  float res = L1DistSIMD16Ext(pVect1v, pVect2v, &qty16);
  float *pVect1 = (float *)pVect1v + qty16;
  float *pVect2 = (float *)pVect2v + qty16;

  size_t qty_left = qty - qty16;
  float res_tail = L1Dist(pVect1, pVect2, &qty_left);
  return res + res_tail;
}

static float L1DistMaskedSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *mask_ptr) {
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4 << 4;
  float res = L1DistMaskedSIMD16ExtSSE(pVect1v, pVect2v, &qty16, mask_ptr);
  float *pVect1 = (float *)pVect1v + qty16;
  float *pVect2 = (float *)pVect2v + qty16;
  float *pMask = (float *)mask_ptr + qty16;

  size_t qty_left = qty - qty16;
  float res_tail = L1DistMasked(pVect1, pVect2, &qty_left, pMask);

  return res + res_tail;
}
#endif

class L1Space : public SpaceInterface<float> {
  DISTFUNC<float> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

 public:
  L1Space(size_t dim) {
    if (dim % 4 == 0) {
      fstdistfunc_ = L1DistSIMD16ExtSSE;  // Use SIMD optimized function for divisible by 4
    } else {
      fstdistfunc_ = L1DistSIMD16ExtResiduals;  // Use residual fallback
    }
    dim_ = dim;
    data_size_ = dim * sizeof(float);
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &dim_; }

  ~L1Space() {}
};

template <typename MTYPE>
using MASKEDDISTFUNC = MTYPE (*)(const void *, const void *, const void *, const void *);

class L1MaskedSpace {
  MASKEDDISTFUNC<float> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

 public:
  L1MaskedSpace(size_t dim) {
    if (dim % 16 == 0) {
      fstdistfunc_ = L1DistMaskedSIMD16ExtSSE;  // Use SIMD optimized function for divisible by 16
    } else {
      fstdistfunc_ = L1DistMaskedSIMD16ExtResiduals;  // Use residual fallback
    }
    dim_ = dim;
    data_size_ = dim * sizeof(float);
  }

  size_t get_data_size() { return data_size_; }

  MASKEDDISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &dim_; }

  ~L1MaskedSpace() {}
};

}  // namespace hnswlib
