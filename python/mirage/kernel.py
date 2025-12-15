import torch

import os
from typing import *

from .core import *
from .threadblock import *
from .visualizer.kernel_visualizer import *
from .utils import *

MAX_THREADS = os.cpu_count()

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

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    def valid_kernels(self):
        assert self._is_compiled, "Should check kernel validness after compilation"
        return self._valid_cuda_kernels

    def get_error_message(self):
        assert self._is_compiled, "Should check error message after compilation"
        return self._error_message


    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)

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
