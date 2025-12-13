import torch

import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
from typing import *

from .core import *
from .threadblock import *
from .visualizer import *
from .utils import *
# from .global_config import global_config
# from .graph_dataset import graph_dataset

from collections import deque

MAX_THREADS = os.cpu_count()

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOO", &input_list, &output_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  nullptr,                  // m_slots     
  nullptr,                  // m_traverse  
  nullptr,                  // m_clear     
  nullptr,                  // m_free      
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


# Because pip install -e . and pip install . have different directory structure,
# we need to check the directory structure to find the correct MIRAGE_ROOT.
def get_key_paths():
    root_dir = os.path.join(
        os.path.dirname(__file__), "../.."
    )  # Using pip install -e .
    if not os.path.exists(os.path.join(root_dir, "deps")):  # Using pip install .
        root_dir = os.path.dirname(__file__)

    # If MIRAGE_ROOT is not set, use the root_dir as MIRAGE_ROOT
    MIRAGE_ROOT = os.environ.get("MIRAGE_ROOT", root_dir)

    INCLUDE_PATH = ""
    DEPS_PATH = ""
    if os.path.exists(os.path.join(MIRAGE_ROOT, "deps")):
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "deps")
    else:
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "include/deps")

    assert os.path.exists(
        MIRAGE_ROOT
    ), "No MIRAGE_ROOT directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        INCLUDE_PATH
    ), "No /include directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        DEPS_PATH
    ), "No /deps directory found. Likely using the wrong MIRAGE_ROOT."

    return MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH


def get_cc_cmd(
    target, cc, FILE_NAME, py_include_dir, INCLUDE_PATH, DEPS_PATH, so_path, profiling
):
    common_cmd = [
        cc,
        FILE_NAME,
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(DEPS_PATH, 'cutlass/include')}",
        "-DMIRAGE_BACKEND_USE_CUDA",
        "-shared",
        "-std=c++17",
        "-use_fast_math",
        "-lcublas",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        so_path,
    ]

    if target == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target == 100:
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    else:
        specific_cmd = [
            "-arch=native",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])

    return common_cmd[:6] + specific_cmd + common_cmd[6:]


def check_stride(dims, strides, layout="row-major"):
    curr_stride = 1
    if layout == "row-major":
        for i in range(len(dims) - 1, -1, -1):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    elif layout == "column-major":
        for i in range(len(dims)):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return True


def gen_empty_tensor(alloc_size, shape, stride, device, dtype=torch.float16):
    return torch.empty(alloc_size, dtype=dtype, device=device).as_strided(shape, stride)


class Handle:
    def __init__(self, handles=[], remain_op=None) -> None:
        self.handles = handles
        self.remain_op = remain_op

    def wait(self):
        for handle in self.handles:
            handle.wait()
        if self.remain_op:
            self.remain_op()


class KNGraph:
    def __init__(self, graph):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self._valid_cuda_kernels = False
        self._cached_results = None
        self.visualizer = None

        self.backend = "cuda"

    def new_input(
        self, dims: tuple, strides: tuple = None, dtype: dtype = float16
    ) -> DTensor:
        # use the default strided layout if strides = None
        if strides is None:
            total_elements = 1
            strides = []
            for d in reversed(dims):
                strides.append(total_elements)
                total_elements *= d
            strides = reversed(strides)
        else:
            assert len(dims) == len(strides)
            # assert check_stride(dims, strides, "row-major") | check_stride(
            #     dims, strides, "column-major"
            # )
        return self.cygraph.new_input(dims, tuple(strides), dtype)

    def mark_output(self, A: DTensor, strides: tuple = None):
        return self.cygraph.mark_output(A, strides)

    def matmul(self, A: DTensor, B: DTensor) -> DTensor:
        return self.cygraph.matmul(A, B)

    def reduction(self, A: DTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def exp(self, A: DTensor):
        return self.cygraph.exp(A)

    def silu(self, A: DTensor):
        return self.cygraph.silu(A)

    def gelu(self, A: DTensor):
        return self.cygraph.gelu(A)

    def relu(self, A: DTensor):
        return self.cygraph.relu(A)

    def clamp(self, A: DTensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def sqrt(self, A: DTensor):
        return self.cygraph.sqrt(A)

    def square(self, A: DTensor):
        return self.cygraph.square(A)

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def pow(self, A: DTensor, B: DTensor):
        return self.cygraph.pow(A, B)

    def rms_norm(self, A: DTensor, normalized_shape: tuple):
        return self.cygraph.rms_norm(A, normalized_shape)

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    def get_owner_independent_hash(self):
        return self.cygraph.get_owner_independent_hash()

    def valid_kernels(self):
        assert self._is_compiled, "Should check kernel validness after compilation"
        return self._valid_cuda_kernels

    def get_error_message(self):
        assert self._is_compiled, "Should check error message after compilation"
        return self._error_message

    def __call__(self, **kwargs):
        if self.backend == "cuda":
            return self.cuda_call(**kwargs)

    def cuda_call(self, **kwargs):
        results = self.compile(**kwargs)

        # directly return if the Transpiler cannot generate valid CUDA kernels
        if not self._valid_cuda_kernels:
            return None

        assert self.run is not None, "The graph is not compiled yet."

        input_tensors = kwargs.get("inputs", [])
        stream = kwargs.get("stream", None)
        if stream is None:
            stream = torch.cuda.default_stream()

        assert self.cygraph.get_num_inputs() == len(
            input_tensors
        ), "Expected {} input tensors, got {}".format(
            self.cygraph.get_num_inputs(), len(input_tensors)
        )

        # TODO: dtype and device
        buffer_tensor = torch.empty(
            results["buf_size"], dtype=torch.uint8, device=input_tensors[0].device
        ).contiguous()

        output_tensors = [
            gen_empty_tensor(
                meta["alloc_size"],
                meta["shape"],
                meta["strides"],
                device=input_tensors[0].device,
                dtype=input_tensors[0].dtype,
            )
            for meta in results["output_directives"]
        ]

        prodiler_buffer_tensor = torch.empty(
            results["profiler_buf_size"],
            dtype=torch.uint64,
            device=input_tensors[0].device,
        ).contiguous()

        buffer_tensor_ptr = buffer_tensor.data_ptr()
        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        output_tensors_ptr = [tensor.data_ptr() for tensor in output_tensors]
        prodiler_buffer_tensor_ptr = prodiler_buffer_tensor.data_ptr()
        self.run(
            input_tensors_ptr,
            output_tensors_ptr,
            buffer_tensor_ptr,
            stream.cuda_stream,
            prodiler_buffer_tensor_ptr,
        )

        if results["profiler_buf_size"] > 0:
            from .profiler import export_to_perfetto_trace

            profiler_result_dir = "./profiling_results"
            profiler_result_file = os.path.join(
                profiler_result_dir, "mirage.perfetto-trace"
            )
            os.makedirs(profiler_result_dir, exist_ok=True)
            export_to_perfetto_trace(prodiler_buffer_tensor, profiler_result_file)
            print(
                f"Exported profiling results to {profiler_result_file}, please view it with perfetto: https://ui.perfetto.dev/"
            )
        return output_tensors

    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)

    def to_json(self, filename):
        cy_to_json(self.cygraph, filename)

    def from_json(self, filename):
        self.cygraph = cy_from_json(filename)

    # Persistent Kernel functions
    def attach_torch_tensor(self, t: DTensor, torch_tensor: torch.Tensor, name: str):
        return self.cygraph.attach_torch_tensor(t, torch_tensor, name)

    def attach_cuda_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_cuda_tensor(t, name)

    def attach_nvshmem_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_nvshmem_tensor(t, name)

    def fuse_tensors(
        self, input: list[DTensor], fuse_dim: int, num_groups: int, name: str
    ):
        return self.cygraph.fuse_tensors(input, fuse_dim, num_groups, name)

    def shuffle_tensors(
        self, input: list[DTensor], shuffled_dim: int, num_groups: int, name: str
    ):
        return self.cygraph.shuffle_tensors(input, shuffled_dim, num_groups, name)

    def register_task(self, bgraph: TBGraph, task_type: str, params: list[int] = None):
        return self.cygraph.register_task(bgraph.cygraph, task_type, params)

    def generate_task_graph(self, num_gpus: int, my_gpu_id: int):
        return self.cygraph.generate_task_graph(num_gpus, my_gpu_id)
