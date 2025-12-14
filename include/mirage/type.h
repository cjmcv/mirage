/* Copyright 2023-2024 CMU
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

// #include "mirage/utils/json_utils.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mirage {
namespace type {

typedef uint16_t FPType;
typedef int64_t GuidType;

// only to be used in create_op in search.cc
inline std::unordered_map<std::string, float> CLAMP_MIN_MAX;

// enum BackendType {
//   BT_CUDA = 0,
//   BT_NKI = 1,
// };

enum DataType {
  // 1-bit types
  // range: 900-909
  // 2-bit types
  // range: 910-919
  // 4-bit types
  // range: 920-929
  DT_FLOAT4 = 920,
  DT_INT4 = 925,
  DT_UINT4 = 926,
  // 8-bit types
  // range(float types): 930-934
  // range(int types): 935-939
  DT_FLOAT8 = 930,
  DT_INT8 = 935,
  DT_UINT8 = 936,
  // 16-bit types
  // range(float types): 940-944
  // range(int types): 945-949
  DT_FLOAT16 = 940,
  DT_BFLOAT16 = 941,
  DT_INT16 = 945,
  DT_UINT16 = 946,
  // 32-bit types
  // range(float type): 950-954
  // range(int type): 955-959
  DT_FLOAT32 = 950,
  DT_INT32 = 955,
  DT_UINT32 = 956,
  // 64-bit types
  // range(float types): 960-964
  // range(int type): 965-969
  DT_DOUBLE = 960,
  DT_INT64 = 965,
  DT_UINT64 = 966,
  DT_UNKNOWN = 999,
};

size_t get_datatype_size(DataType type);
// std::string get_datatype_str(DataType dtype);

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_CUSTOMIZED_OP = 1999,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_OUTPUT_OP, "kn_output_op"},
                                 {KN_CUSTOMIZED_OP, "kn_customized_op"},
                             })

enum TBOperatorType {
  TB_UNKOWN = 2000,
  TB_INPUT_OP = 2001,
  TB_OUTPUT_OP = 2002,
  TB_CUSTOMIZED_OP = 2999
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    TBOperatorType,
    {
        {TB_UNKOWN, "tb_unkown"},
        {TB_INPUT_OP, "tb_input_op"},
        {TB_OUTPUT_OP, "tb_output_op"},
        {TB_CUSTOMIZED_OP, "tb_customized_op"},
    })

// bool is_threadblock_element_unary(TBOperatorType op_type);

enum ActivationType {
  ACT_UNKOWN = 3000,
  ACT_EXP = 3001,
  ACT_RELU = 3002,
  ACT_GELU = 3003,
  ACT_SILU = 3004,
  ACT_NONE = 3099,
};

enum TBEpilogueType {
  TB_EPILOGUE_NONE = 3100,
  TB_EPILOGUE_ALLREDUCE = 3101,
  TB_EPILOGUE_ALLTOALL = 3102,
  TB_EPILOGUE_INVALID = 3199,
};

NLOHMANN_JSON_SERIALIZE_ENUM(TBEpilogueType,
                             {
                                 {TB_EPILOGUE_NONE, "tb_epilogue_none"},
                                 {TB_EPILOGUE_ALLREDUCE,
                                  "tb_epilogue_allreduce"},
                                 {TB_EPILOGUE_ALLTOALL, "tb_epilogue_alltoall"},
                                 {TB_EPILOGUE_INVALID, "tb_epilogue_invalid"},
                             })

} // namespace type
} // namespace mirage
