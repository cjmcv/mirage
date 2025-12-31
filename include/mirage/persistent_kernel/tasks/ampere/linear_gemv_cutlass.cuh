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

// A[1, k] * B[n, k] => C[n]
// K => k
// N => real_n
template <typename T,
          int THREAD_NUM,
          int TILE_DIM_X, 
          int TILE_DIM_Y, 
          int TILE_DIM_Z,
          int M,
          int N,
          int K,
          int O_STRIDE = N,
          int PIPE_MAX = 3,
          bool FUSE_RES = false>
__device__ __forceinline__ void linear_kernel(const int bx, const int by, const int bz,
                                              void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  // if (threadIdx.x == 0) {
  //   printf("[%d-(%d,%d,%d)]-tile(%d,%d,%d)(%d)\n", blockIdx.x, bx, by, bz, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, THREAD_NUM);
  //   printf("[%d] gemv input_ptr: %lld, weight_ptr: %lld, output_ptr: %lld: %d,%d,%d.\n", blockIdx.x, input_ptr, weight_ptr, output_ptr, num_active_tokens, N, K);
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

  // 一个block有128个线程，一行32个线程，共4行。
  // 一行中，32个线程，每个线程处理连续的 8(kElementsPerAccess) 个元素，一次处理256(tileA_k)个元素，循环遍历REDUCTION_SIZE同时取AB，做相乘，并累加在寄存器。
  // 一个线程处理8个元素，所以内层还有一个8的小循环，执行后得到该线程负责的所有内容。32个线程再累加起来(warp reduce)就得到这一行的计算结果，即C矩阵的一个值。
  static int const kThreadsPerRow = 32;

  int n_step = THREAD_NUM / kThreadsPerRow;
  int idx_col_k = threadIdx.x % kThreadsPerRow; // 32
  int idx_row_n = threadIdx.x / kThreadsPerRow; //  4 
  for (; idx_row_n < TILE_DIM_X; idx_row_n += n_step) {
    // problem_size (row = m, column = k)
    // matrix A (batch, m, k)
    // vector B (batch, 1, k)
    // vector C (batch, m, 1)
    // vector D (batch, m, 1)

    // move in the batch dimension
    ElementA const *ptr_A = (ElementA const *)input_ptr;
    ElementB const *ptr_B = (ElementB const *)weight_ptr + bx * TILE_DIM_X * K;
    ElementC *ptr_D = (ElementC *)output_ptr + bx * TILE_DIM_X;
    ElementC *ptr_R = (ElementC *)residual_ptr + bx * TILE_DIM_X;

    // move in the k dimension
    ptr_A += idx_col_k * kElementsPerAccess;
    ptr_B += idx_col_k * kElementsPerAccess;

    // move in the m dimension
    ptr_B += idx_row_n * K;
    ptr_D += idx_row_n;
    if constexpr (FUSE_RES) {
      ptr_R += idx_row_n;
    }
    cutlass::NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
    cutlass::NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;

    ElementAccumulator accum = 0.f;

    FragmentB fragB;
    FragmentA fragA;

    // rows of the rolling tile
    int const tileA_k = kThreadsPerRow * kElementsPerAccess;
    
    int unroll_col_k = 0;
    for (; unroll_col_k < K / tileA_k * tileA_k; unroll_col_k += tileA_k) {

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
    for (int k = unroll_col_k + idx_col_k; k < K; k += kThreadsPerRow) {
      ElementB b = *(ptr_B - idx_col_k * kElementsPerAccess + k);
      ElementA a = *(ptr_A - idx_col_k * kElementsPerAccess + k);

      accum += ElementAccumulator(a) * ElementAccumulator(b);
    }

    for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
      accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask, 32);
    }

    if (idx_col_k == 0) {
      if constexpr (FUSE_RES) {
        accum += *ptr_R;
      }
      *ptr_D = (ElementC)accum;
    }
  }
}

template <typename T,
          int THREAD_NUM,
          int TILE_DIM_X, 
          int TILE_DIM_Y, 
          int TILE_DIM_Z,
          int M,
          int N,
          int K,
          int O_STRIDE = N,
          int PIPE_MAX = 3,
          bool FUSE_RES = false>
__device__ __forceinline__ void linear_postfix_kernel(const int bx, const int by, const int bz,
                                              void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  // if (threadIdx.x == 0) {
  //   printf("[%d,%d, %d,%d] gemv_postfix input_ptr: %lld, weight_ptr: %lld, output_ptr: %lld: %d,%d,%d.\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, input_ptr, weight_ptr, output_ptr, num_active_tokens, N, K);
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

  // [1, 9728] * [2560, 9728] => [spk, 2560]
  // 假设外层repeat[2,2], 共4个block。
  // N = 2560/rp.x = 1280, K = 9728/rp.y = 4864
  // 以linear_postfix_layer的布局，会拿到
  // BLOCK1 , BLOCK2 ,    BLOCK3 ,     BLOCK4 
  // A(0, 0), A(0, 4864), A(0, 0),     A(0, 4864) 
  // B(0, 0), B(0, 4864), B(1280, 0),  B(1280, 4864)
  // C(0, 0), C(1, 0),    C(0, 1280),  C(1, 1280)
  // BLOCK1 负责A[0, 0:K] * B[0:N, 0:K] = C[0, 0:N]
  // 一个block有128个线程，一行32个线程，共4行。
  // 每个线程一次处理8个元素，32个线程一次处理256个元素。

  // BLOCK3 [1,0, 20,1] gemv_postfix input_ptr: 21552433664, weight_ptr: 21527003136, output_ptr: 21552455680: 1,1280,4864.
  // BLOCK2 [2,0, 20,1] gemv_postfix input_ptr: 21552443392, weight_ptr: 21502109184, output_ptr: 21552458240: 1,1280,4864.
  // BLOCK4 [3,0, 20,1] gemv_postfix input_ptr: 21552443392, weight_ptr: 21527012864, output_ptr: 21552460800: 1,1280,4864.
  // BLOCK1 [0,0, 20,1] gemv_postfix input_ptr: 21552433664, weight_ptr: 21502099456, output_ptr: 21552453120: 1,1280,4864.

  // int row_chunk = N / gridDim.y;
  // int base_row_m = blockIdx.y * row_chunk;
  
  static int const kThreadsPerRow = 32;
  int row_m = THREAD_NUM / kThreadsPerRow;      // 4
  int idx_col_k = threadIdx.x % kThreadsPerRow; // [0-31]
  int idx_row_n = threadIdx.x / kThreadsPerRow; // [0-3]

  for (; idx_row_n < N; idx_row_n += row_m) {
    ElementA const *ptr_A = (ElementA const *)input_ptr;
    ElementB const *ptr_B = (ElementB const *)weight_ptr;
    ElementC *ptr_D = (ElementC *)output_ptr;

    // move in the k dimension
    ptr_A += idx_col_k * kElementsPerAccess;
    ptr_B += idx_col_k * kElementsPerAccess;

    // move in the m dimension
    ptr_B += idx_row_n * 9728;
    ptr_D += idx_row_n;

    cutlass::NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
    cutlass::NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;

    ElementAccumulator accum = 0.f;

    FragmentB fragB;
    FragmentA fragA;

    // rows of the rolling tile
    int const tileA_k = kThreadsPerRow * kElementsPerAccess;
    
    int unroll_col_k = 0;
    for (; unroll_col_k < K / tileA_k * tileA_k; unroll_col_k += tileA_k) {

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
    for (int k = unroll_col_k + idx_col_k; k < K; k += kThreadsPerRow) {
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
