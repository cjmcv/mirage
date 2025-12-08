/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tasks/common/common_header.cuh"
#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/numeric_conversion.h"

#define DEBUG 0

#if DEBUG
#define DCHECK(condition)                                                      \
  if ((condition) == 0) {                                                      \
    printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);                    \
  }
#else
#define DCHECK(condition)
#endif // DEBUG

namespace kernel {

using bfloat16 = type::bfloat16_t;
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  // if (threadIdx.x == 0) {
  //   printf("gemv input_ptr: %lld, weight_ptr: %lld, output_ptr: %lld: %d,%d,%d.\n", input_ptr, weight_ptr, output_ptr, num_active_tokens, OUTPUT_SIZE, REDUCTION_SIZE);
  // }
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  static int const kElementsPerAccess = 8;
  using FragmentA = cutlass::Array<ElementA, kElementsPerAccess>;
  using FragmentB = cutlass::Array<ElementB, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  static cutlass::FloatRoundStyle const Round = cutlass::FloatRoundStyle::round_to_nearest;
  // static int const kThreadCount = 128;
  static int const kThreadsPerRow = 32;

  int idx_col_k = threadIdx.x % kThreadsPerRow;
  int idx_row_m = threadIdx.x / kThreadsPerRow;
  for (; idx_row_m < OUTPUT_SIZE; idx_row_m += blockDim.x / kThreadsPerRow) {
    // problem_size (row = m, column = k)
    // matrix A (batch, m, k)
    // vector B (batch, 1, k)
    // vector C (batch, m, 1)
    // vector D (batch, m, 1)

    // move in the batch dimension
    ElementA const *ptr_A = (ElementA const *)input_ptr;
    ElementB const *ptr_B = (ElementB const *)weight_ptr;
    ElementC *ptr_D = (ElementC *)output_ptr;

    // move in the k dimension
    ptr_A += idx_col_k * kElementsPerAccess;
    ptr_B += idx_col_k * kElementsPerAccess;

    // move in the m dimension
    ptr_B += idx_row_m * REDUCTION_SIZE;
    // ptr_C += idx_row_m;
    ptr_D += idx_row_m;

    cutlass::NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
    cutlass::NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;

    ElementAccumulator accum = 0.f;

    FragmentB fragB;
    FragmentA fragA;

    // rows of the rolling tile
    int const tileA_k = kThreadsPerRow * kElementsPerAccess;
    
    int unroll_col_k = 0;
    for (; unroll_col_k < REDUCTION_SIZE / tileA_k * tileA_k; unroll_col_k += tileA_k) {

      // fetch from matrix A
      cutlass::arch::global_load<FragmentA,
                        sizeof(FragmentA),
                        cutlass::arch::CacheOperation::LastUse>(fragA, (ptr_A + unroll_col_k), true);

      // fetch from vector B
      cutlass::arch::global_load<FragmentB,
                        sizeof(FragmentB),
                        cutlass::arch::CacheOperation::Always>(fragB, (ptr_B + unroll_col_k), true);

      FragmentCompute fragB_Compute = srcB_converter(fragB);
      FragmentCompute fragA_Compute = srcA_converter(fragA);

      // Math
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kElementsPerAccess; e++) {
        accum += fragA_Compute.at(e) * fragB_Compute.at(e);
      }
    }

    // calculate the rest of K elements
    // each thread fetch 1 element each time
    // for (int k = unroll_col_k + idx_col_k; k < params.problem_size.column(); k += kThreadsPerRow) {
    for (int k = unroll_col_k + idx_col_k; k < REDUCTION_SIZE; k += kThreadsPerRow) {
      ElementB b = *(ptr_B - idx_col_k * kElementsPerAccess + k);
      ElementA a = *(ptr_A - idx_col_k * kElementsPerAccess + k);

      accum += ElementAccumulator(a) * ElementAccumulator(b);
    }

    for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
      accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask, 32);
    }

    if (idx_col_k == 0) {
      *ptr_D = (ElementC)accum;
    }
  }
}

} // namespace kernel
