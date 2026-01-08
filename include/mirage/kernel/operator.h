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

#include "mirage/kernel/device_tensor.h"
#include <vector>

namespace mirage {
namespace kernel {

class Graph;

class KNOperator {
public:
  KNOperator(mirage::type::KNOperatorType _type): op_type(_type) {}
  KNOperator(mirage::type::KNOperatorType _type, DTensor const &input1):op_type(_type) {
    input_tensors.push_back(input1);
  }

  KNOperator(mirage::type::KNOperatorType _type,
             DTensor const &input1, DTensor const &input2) : op_type(_type) {
    input_tensors.push_back(input1);
    input_tensors.push_back(input2);
  }

  KNOperator(mirage::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs) : op_type(_type) {
    for (auto const &i : inputs) {
      input_tensors.push_back(i);
    }
  }

  int get_input_dtensors(DTensor **inputs) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      inputs[i] = &input_tensors[i];
    }
    return input_tensors.size();
  }
  
  int get_output_dtensors(DTensor **outputs) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      outputs[i] = &output_tensors[i];
    }
    return output_tensors.size();
  }
  
  std::vector<DTensor>& get_output_dtensors() {
    return output_tensors;
  }
  virtual ~KNOperator() {}

public:
  mirage::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(std::vector<int> const &dims,
            std::vector<size_t> const &strides,
            mirage::type::DataType data_type,
            mirage::layout::DmemLayout layout,
            int3 _input_map = {-1, -1, -1})
      : KNOperator(mirage::type::KN_INPUT_OP), input_strides(strides),
    input_map(_input_map) {
    assert(dims.size() == strides.size());
    DTensor tensor;
    tensor.num_dims = dims.size();
    for (int i = tensor.num_dims - 1; i >= 0; i--) {
      tensor.dim[i] = dims[i];
    }
    tensor.data_type = data_type;
    tensor.layout = layout;
    tensor.owner_op = this;
    tensor.owner_ts_idx = 0;
    tensor.guid = DTensor::next_guid++;
    // kgraph->allocate(tensor);
    output_tensors.push_back(tensor);
  }
  ~KNInputOp() {
    // kgraph->free(output_tensors[0]);
  }

public:
  std::vector<size_t> input_strides;
  int3 input_map;
};

} // namespace kernel
} // namespace mirage
