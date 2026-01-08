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

#include "mirage/config.h"
#include "mirage/layout.h"
#include "mirage/type.h"
#include <atomic>
#include <cstddef>
#include <functional>

namespace mirage {
namespace kernel {

class KNOperator;

struct alignas(16) DTensor {
  DTensor(void) {
    data_type = mirage::type::DT_UNKNOWN;
    layout = mirage::layout::DmemUnknownLayout;
    num_dims = 0;
    for (int i = 0; i < mirage::config::MAX_TENSOR_DIMS; i++) {
      dim[i] = 0;
      // stride[i] = 0;
    }
    owner_op = nullptr;
    owner_ts_idx = -1000;
    data_offset = -1000;
  }
  inline bool operator==(DTensor const &b) const {
    if (data_type != b.data_type) {
      return false;
    }
    if (layout != b.layout) {
      return false;
    }
    if (num_dims != b.num_dims) {
      return false;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return false;
      }
      // if (stride[i] != b.stride[i]) {
      //   return false;
      // }
    }
    if (owner_op != b.owner_op) {
      return false;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return false;
    }
    assert(data_offset == b.data_offset);
    return true;
  }
  inline bool operator!=(DTensor const &b) const {
    if (data_type != b.data_type) {
      return true;
    }
    if (layout != b.layout) {
      return true;
    }
    if (num_dims != b.num_dims) {
      return true;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return true;
      }
      // if (stride[i] != b.stride[i]) {
      //   return true;
      // }
    }
    if (owner_op != b.owner_op) {
      return true;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return true;
    }
    assert(data_offset == b.data_offset);
    return false;
  }

  inline size_t num_elements() const {
    size_t num = 1;
    for (int i = 0; i < num_dims; i++) {
      num *= dim[i];
    }
    return num;
  }

  inline size_t data_size() const {
    using namespace mirage::type;
    size_t data_type_size = get_datatype_size(data_type);
    return num_elements() * data_type_size;
  }

  static const DTensor EMPTY_TENSOR;

public:
  mirage::type::DataType data_type;
  mirage::layout::DmemLayout layout;
  int num_dims;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  type::GuidType guid;
  //  DTensor fields
  KNOperator *owner_op;
  int owner_ts_idx;
  // offset in device memory
  int64_t data_offset;

  static std::atomic<int64_t> next_guid;
};

inline const DTensor DTensor::EMPTY_TENSOR = {/*zero-initialization*/};
inline std::atomic<int64_t> DTensor::next_guid = 10000000;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    DTensor, data_type, layout, num_dims, dim, guid)

} // namespace kernel
} // namespace mirage