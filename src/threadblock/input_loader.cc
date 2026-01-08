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

#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {


TBInputOp::TBInputOp(Graph *_graph,
                     mirage::kernel::DTensor const &_dtensor,
                     int3 _input_map,
                     mirage::layout::SmemLayout _layout,
                     bool store_in_dmem)
    : TBOperator(_graph, mirage::type::TB_INPUT_OP), dtensor(_dtensor),
      input_map(_input_map)  {
  STensor tensor;
  tensor.layout = _layout;
  tensor.num_dims = dtensor.num_dims;
  tensor.data_type = dtensor.data_type;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = dtensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && bgraph->grid_dim.x > 1) {
      dim_idx = input_map.x;
      dim_div = bgraph->grid_dim.x;
    }
    if (d == 1 && bgraph->grid_dim.y > 1) {
      dim_idx = input_map.y;
      dim_div = bgraph->grid_dim.y;
    }
    if (d == 2 && bgraph->grid_dim.z > 1) {
      dim_idx = input_map.z;
      dim_div = bgraph->grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(tensor.dim[dim_idx] > 0);
      // assert(tensor.dim[dim_idx] % dim_div == 0);
      if (tensor.dim[dim_idx] % dim_div != 0) {
        fprintf(stderr, "(tensor.dim[dim_idx] %% dim_div != 0): [tensor.dim[%d]=%d, dim_div=%d]\n", dim_idx, tensor.dim[dim_idx], dim_div);
        abort();
      }
      tensor.dim[dim_idx] /= dim_div;
    }
  }

  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.guid = STensor::next_guid++;
  tensor.after_accum = false;
  tensor.store_in_dmem = store_in_dmem;
  tensor.smem_offset = bgraph->smem_offset; // bgraph->allocate_fingerprint(tensor);
  output_tensors.push_back(tensor);
}

TBInputOp::~TBInputOp() {}

size_t TBInputOp::get_dtensor_guid() {
  return dtensor.guid;
}

} // namespace threadblock
} // namespace mirage
