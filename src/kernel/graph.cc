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

#include "mirage/kernel/graph.h"
#include "mirage/config.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/task_register.h"

#include <algorithm>
#include <iostream>

namespace mirage {
namespace kernel {

Graph::Graph(dim3 _gpu_dim)
    : gpu_dim(_gpu_dim) {
  dmem_data_offset = 0;
}

Graph::~Graph() {
  while (!operators.empty()) {
    delete operators.back();
    operators.pop_back();
  }
}

int Graph::get_input_dtensors(DTensor **inputs) const {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      assert(op->output_tensors.size() == 1);
      inputs[num_inputs++] = &op->output_tensors[0];
    }
  }
  return num_inputs;
}

int Graph::get_num_input_dtensors() const {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      num_inputs++;
    }
  }
  return num_inputs;
}

int Graph::get_input_dtensor_shape_and_stride(DTensor const *input,
                                              int *strides,
                                              int *dims) const {
  for (auto const &op : this->operators) {
    if (op == input->owner_op) {
      assert(op->op_type == mirage::type::KN_INPUT_OP &&
             "input is not an KNInputOp");
      KNInputOp *input_op = static_cast<KNInputOp *>(op);
      int num_dims = (int)input_op->input_strides.size();
      for (int i = 0; i < num_dims; i++) {
        strides[i] = input_op->input_strides[i];
        dims[i] = input->dim[i];
      }
      return num_dims;
    }
  }
  assert(false && "Cannot find input dtensor");
  return 0;
}

bool Graph::allocate(DTensor &tensor) {
  // assert that the start of the tensor is 16 bytes aligned
  assert(dmem_data_offset % 16 == 0);
  off_t ret = dmem_data_offset;

  size_t aligns_size = ((tensor.data_size() + 15) & ~15);
  dmem_data_offset += aligns_size;

  allocated_data_tensors.push_back(std::make_pair(ret, aligns_size));
  tensor.data_offset = ret;

  return true;
}

void Graph::free(DTensor &tensor) {
  assert(allocated_data_tensors.size() > 0);
  assert(allocated_data_tensors.back().first == tensor.data_offset);
  assert(allocated_data_tensors.back().second ==
         ((tensor.data_size() + 15) & ~15));
  dmem_data_offset -= allocated_data_tensors.back().second;
  allocated_data_tensors.pop_back();
  tensor.data_offset = -1;
}

// Persistent kernel functions
using namespace mirage::runtime;
void Graph::attach_torch_tensor(DTensor const *input,
                                void *torch_data_ptr,
                                char const *name) {
  io_config.emplace(
      input->guid,
      IODesc(IODesc::TorchTensor, std::string(name), *input, torch_data_ptr));
}

void Graph::attach_cuda_tensor(DTensor const *input, char const *name) {
  io_config.emplace(
      input->guid, IODesc(IODesc::CUDAMallocTensor, std::string(name), *input));
}

void Graph::attach_nvshmem_tensor(DTensor const *input, char const *name) {
  io_config.emplace(
      input->guid,
      IODesc(IODesc::NVSHMEMMallocTensor, std::string(name), *input));
}

DTensor *Graph::fuse_tensors(std::vector<DTensor const *> inputs,
                             int fused_dim,
                             int num_groups,
                             char const *name) {
  // Currently assert that we fuse along the 0-th dim (for weights)
  assert(fused_dim == 0);
  assert(inputs.size() > 0);
  std::vector<int> dims;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    dims.push_back(inputs[0]->dim[i]);
  }
  for (size_t t = 1; t < inputs.size(); t++) {
    dims[0] += inputs[t]->dim[0];
    assert(inputs[0]->num_dims == inputs[t]->num_dims);
    for (int i = 1; i < inputs[t]->num_dims; i++) {
      assert(dims[i] == inputs[t]->dim[i]);
    }
    assert(inputs[0]->data_type == inputs[t]->data_type);
  }
  std::vector<size_t> strides(dims.size(), 1);
  for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
    if (i == inputs[0]->num_dims - 1) {
      strides[i] = 1;
    } else {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }
  DTensor *fused =
      new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
  IODesc desc(IODesc::FusedTorchTensor, std::string(name), *fused);
  desc.num_groups = num_groups;
  for (size_t t = 0; t < inputs.size(); t++) {
    assert(io_config.find(inputs[t]->guid) != io_config.end());
    IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
    desc.sub_descs.push_back(sub_desc);
    io_config.erase(inputs[t]->guid);
  }
  io_config.emplace(fused->guid, desc);
  return fused;
}

DTensor *Graph::shuffle_tensors(std::vector<DTensor const *> inputs,
                                int shuffled_dim,
                                int num_groups,
                                char const *name) {
  // Currently assert that we shuffle along the 0-th dim (for weights)
  assert(shuffled_dim == 0);
  assert(inputs.size() > 0);
  std::vector<int> dims;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    dims.push_back(inputs[0]->dim[i]);
  }
  for (size_t t = 1; t < inputs.size(); t++) {
    dims[0] += inputs[t]->dim[0];
    assert(inputs[0]->num_dims == inputs[t]->num_dims);
    for (int i = 1; i < inputs[t]->num_dims; i++) {
      assert(dims[i] == inputs[t]->dim[i]);
    }
    assert(inputs[0]->data_type == inputs[t]->data_type);
  }
  std::vector<size_t> strides(dims.size(), 1);
  for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
    if (i == inputs[0]->num_dims - 1) {
      strides[i] = 1;
    } else {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }
  DTensor *shuffled =
      new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
  IODesc desc(IODesc::ShuffledTorchTensor, std::string(name), *shuffled);
  desc.num_groups = num_groups;
  for (size_t t = 0; t < inputs.size(); t++) {
    assert(io_config.find(inputs[t]->guid) != io_config.end());
    IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
    desc.sub_descs.push_back(sub_desc);
    io_config.erase(inputs[t]->guid);
  }
  io_config.emplace(shuffled->guid, desc);
  return shuffled;
}

void Graph::register_task(char const *task_type, std::vector<int> params) {
  std::string name = std::string(task_type);
  KNOperator const *op = operators.back();
  assert(op->op_type == type::KN_CUSTOMIZED_OP);
  KNCustomizedOp const *customized = static_cast<KNCustomizedOp const *>(op);
  TaskRegister *task_register = TaskRegister::get_instance();
  if (name == "embedding") {
    int variant_id =
        task_register->register_embedding_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING, variant_id);
  } else if (name == "rmsnorm") {
    int variant_id =
        task_register->register_rmsnorm_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM, variant_id);
  } else if (name == "rmsnorm_linear") {
    int variant_id =
        task_register->register_rmsnorm_linear_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(3, 1, TASK_RMS_NORM_LINEAR, variant_id);
  } else if (name == "attention") {
    int variant_id =
        task_register->register_attention_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_ATTENTION_1, variant_id);
  } else if (name == "paged_attention") {
    int variant_id = task_register->register_paged_attention_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_PAGED_ATTENTION_1, variant_id);
  } else if (name == "single_batch_extend_attention") {
    int variant_id = task_register->register_single_batch_extend_attention_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 1, TASK_SINGLE_BATCH_EXTEND_ATTENTION, variant_id);
  } else if (name == "linear") {
    int variant_id = task_register->register_linear_task(
        customized->bgraph, params, false /*with_residual*/, 0);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR, variant_id);
  } else if (name == "linear_postfix") {
    int variant_id = task_register->register_linear_task(
        customized->bgraph, params, false /*with_residual*/, 1);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_POSTFIX, variant_id);
  } else if (name == "linear_with_residual") {
    int variant_id = task_register->register_linear_task(
        customized->bgraph, params, true /*with_residual*/, 0);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL, variant_id);
  } else if (name == "silu_mul") {
    int variant_id =
        task_register->register_silu_mul_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
  } else if (name == "identity") {
    int variant_id =
        task_register->register_identity_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_IDENTITY, variant_id);
  } else if (name == "silu_mul_linear_with_residual") {
    int variant_id = task_register->register_silu_mul_linear_with_residual_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(3, 1, TASK_SILU_MUL_LINEAR_WITH_RESIDUAL, variant_id);
  } else if (name == "argmax") {
    task_config[op] = std::make_tuple(1, 1, TASK_ARGMAX, 0);
  } else if (name == "argmax_partial") {
    int variant_id =
        task_register->register_argmax_partial_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL, variant_id);
  } else if (name == "argmax_reduce") {
    int variant_id =
        task_register->register_argmax_reduce_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_ARGMAX_REDUCE, variant_id);
  } else if (name == "allreduce") {
    // `register_reduce_task` will register two tasks, but we only record one
    int variant_id =
        task_register->register_reduce_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_ALLREDUCE, variant_id);
  } else if (name == "find_ngram_partial") {
    int variant_id = task_register->register_find_ngram_partial_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 1, TASK_FIND_NGRAM_PARTIAL, variant_id);
  } else if (name == "find_ngram_global") {
    int variant_id = task_register->register_find_ngram_global_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_FIND_NGRAM_GLOBAL, variant_id);
  } else if (name == "target_verify_greedy") {
    int variant_id = task_register->register_target_verify_greedy_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(2, 1, TASK_TARGET_VERIFY_GREEDY, variant_id);
  }
  // Hopper tasks
  else if (name == "linear_hopper") {
    int variant_id = task_register->register_linear_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_HOPPER, variant_id);
  } else if (name == "linear_with_residual_hopper") {
    int variant_id = task_register->register_linear_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "paged_attention_hopper") {
    int variant_id = task_register->register_paged_attention_hopper_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 1, TASK_PAGED_ATTENTION_HOPPER, variant_id);
  } else if (name == "rmsnorm_hopper") {
    int variant_id =
        task_register->register_rmsnorm_hopper_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM_HOPPER, variant_id);
  } else if (name == "linear_swapAB_hopper") {
    int variant_id = task_register->register_linear_swapAB_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_LINEAR_SWAPAB_HOPPER, variant_id);
  } else if (name == "linear_swapAB_with_residual_hopper") {
    int variant_id = task_register->register_linear_swapAB_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] = std::make_tuple(
        3, 1, TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "linear_cutlass_hopper") {
    int variant_id = task_register->register_linear_cutlass_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_LINEAR_CUTLASS_HOPPER, variant_id);
  } else if (name == "linear_cutlass_with_residual_hopper") {
    int variant_id = task_register->register_linear_cutlass_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] = std::make_tuple(
        3, 1, TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "silu_mul_hopper") {
    int variant_id = task_register->register_silu_mul_hopper_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL_HOPPER, variant_id);
  } else if (name == "embedding_hopper") {
    int variant_id = task_register->register_embedding_hopper_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING_HOPPER, variant_id);
  } else if (name == "moe_w13_linear_sm90") {
    int variant_id = task_register->register_moe_linear_sm90_task(
        customized->bgraph, params, true /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM90, variant_id);
  } else if (name == "moe_w2_linear_sm90") {
    int variant_id = task_register->register_moe_linear_sm90_task(
        customized->bgraph, params, false /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM90, variant_id);
  } else if (name == "splitk_linear_swapAB_hopper") {
    int variant_id = task_register->register_splitk_linear_swapAB_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SWAPAB_HOPPER, variant_id);
  } else if (name == "paged_attention_split_kv_hopper") {
    int variant_id =
        task_register->register_paged_attention_split_kv_hopper_task(
            customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 2, TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER, variant_id);
  }
  // SM100 tasks
  else if (name == "linear_sm100") {
    int variant_id = task_register->register_linear_sm100_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_SM100, variant_id);
  } else if (name == "splitk_linear_sm100") {
    int variant_id = task_register->register_splitk_linear_sm100_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SM100, variant_id);
  } else if (name == "linear_with_residual_sm100") {
    int variant_id = task_register->register_linear_sm100_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_SM100, variant_id);
  } else if (name == "paged_attention_sm100") {
    int variant_id = task_register->register_paged_attention_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_ATTN_SM100, variant_id);
  } else if (name == "argmax_partial_sm100") {
    int variant_id = task_register->register_argmax_partial_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL_SM100, variant_id);
  } else if (name == "argmax_reduce_sm100") {
    int variant_id = task_register->register_argmax_reduce_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(2, 1, TASK_ARGMAX_REDUCE_SM100, variant_id);
  } else if (name == "sampling_sm100") {
    int variant_id =
        task_register->register_sampling_sm100_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SAMPLING_SM100, variant_id);
  } else if (name == "tensor_init") {
    int variant_id =
        task_register->register_tensor_init_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_TENSOR_INIT, variant_id);
  } else if (name == "moe_topk_softmax_sm100") {
    int variant_id = task_register->register_moe_topk_softmax_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 3, TASK_MOE_TOPK_SOFTMAX_SM100, variant_id);
  } else if (name == "moe_w13_linear_sm100") {
    int variant_id = task_register->register_moe_linear_sm100_task(
        customized->bgraph, params, true /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM100, variant_id);
  } else if (name == "moe_silu_mul") {
    int variant_id =
        task_register->register_moe_silu_mul_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
  } else if (name == "moe_w2_linear_sm100") {
    int variant_id = task_register->register_moe_linear_sm100_task(
        customized->bgraph, params, false /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM100, variant_id);
  } else if (name == "moe_mul_sum_add_sm100") {
    int variant_id = task_register->register_moe_mul_sum_add_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(3, 1, TASK_MOE_MUL_SUM_ADD_SM100, variant_id);
  } else if (name == "paged_attention_split_kv_sm100") {
    int variant_id =
        task_register->register_paged_attention_split_kv_sm100_task(
            customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 2, TASK_PAGED_ATTENTION_SPLIT_KV_SM100, variant_id);
  } else if (name == "paged_attention_split_kv_merge_sm100") {
    int variant_id =
        task_register->register_paged_attention_split_kv_merge_sm100_task(
            customized->bgraph, params);
    task_config[op] = std::make_tuple(
        2, 1, TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100, variant_id);
  } else {
    printf("Unsupported task name: %s\n", name);
    assert(false && "Unsupported task type");
  }
}

} // namespace kernel
} // namespace mirage
