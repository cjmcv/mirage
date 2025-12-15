from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi

# print limitation
torch.set_printoptions(profile="full")

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    
def test_torch_mlp2(x, w_gatedup, w_down_proj):
    import torch.nn.functional as F
    O1 = F.linear(x, w_gatedup)
    D = silu_and_mul(O1)
    return F.linear(D, w_down_proj)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=1, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", default="./gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")

    args = parser.parse_args()
    world_size = 1
    rank = 0

    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    # model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    # with torch.device("cuda"):
    #     model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B"
    #     model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.profiling:
        profiler_tensor = torch.zeros(
            3000 * 128, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None

    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    print("num_workers: ", num_workers)
    print("num_schedulers: ", num_schedulers)
    
    qo_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=world_size,
        mpi_rank=rank,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        meta_tensors={
            "qo_indptr_buffer": qo_indptr_buffer,
        },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        # spec_decode_config=spec_decode_config,
        use_cutlass_kernel=False,
    )
    
    splitk = 1 # 8
    batch_size = 1
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    mlp_out_torch = torch.zeros((splitk, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w2")
    mlp_out = mpk.attach_input(torch_tensor=mlp_out_torch, name="mlp_out")
    
    # mlp_mid_torch = torch.zeros((batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    mlp_mid = mpk.new_tensor(dims=(batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=x,
        weight=w_gatedup,
        output=mlp_mid,
        # grid_dim=(96, 1, 1),
        # grid_dim=(128, 1, 1),
        grid_dim=(32, 1, 1),  # (64, 1, 1)
        block_dim=(128, 1, 1),
    )
    
    # silu_mul_out_torch = torch.zeros((batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    # mlp_out_torch = silu_mul_out_torch
    silu_mul_out = mpk.new_tensor(dims=(batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mpk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(16, 1, 1),
        block_dim=(128, 1, 1),
    )
    if splitk == 1:
        mpk.linear_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj,
            output=mlp_out,
            grid_dim=(32, 1, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    else:
        mpk.linear_postfix_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj,
            output=mlp_out,
            grid_dim=(splitk, 8, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    
    # results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
    # with open(f"./gen/t/task_graph.json", "w") as f:
    #     f.write(results["json_file"])
    # with open(f"./gen/t/kernel.cu", "w") as f:
    #     f.write(results["cuda_code"])
        
    if args.nc is True:
        module_path = args.output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
        mpk.load_module(module_path)
    else:
        module_path = mpk.compile(output_dir=args.output_dir)
        print("module_path: ", module_path)
        mpk.load_module(module_path)
    
    with torch.device("cuda"):
        model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B"
        model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    ###
    warnup_iter = 100
    test_iter = 200
    
    x_torch2 = x_torch.clone()
    w_gatedup_torch2 = w_gatedup_torch.clone()
    w_down_proj_torch2 = w_down_proj_torch.clone()
    for _ in range(warnup_iter):
        test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    ###
    
    # ###############################################################
    mpk()
    torch_out = test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    
    print("torch_out: ", torch_out[0])
    
    if splitk != 1:
        print("mpk0: ", mlp_out_torch[0], "\nmpk1", mlp_out_torch[1])
        for i in range(1, splitk):
            mlp_out_torch[0] += mlp_out_torch[i]
    print("diff: ", mlp_out_torch[0] - torch_out[0])
    print("allclose 0: ", torch.allclose(mlp_out_torch[0], torch_out[0], rtol=1e-2))
    
    for _ in range(5):
        mlp_out_torch.zero_()
        # mpk.reinitialize()
        mpk()
        if splitk != 1:
            for i in range(1, splitk):
                mlp_out_torch[0] += mlp_out_torch[i]
        if (torch.allclose(mlp_out_torch[0], torch_out[0], rtol=1e-2)):
            print("allclose: True")
        else:
            print("diff: ", mlp_out_torch[0] - torch_out[0])
        
    #############################################################
        
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
        
    starter.record()
    for _ in range(test_iter):
        # mpk.reinitialize()
        mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    print("MPK run time (ms): ", run_time / test_iter)
    
    ##
    
    starter.record()
    for _ in range(test_iter):
        test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    print("torch run time (ms): ", run_time / test_iter)

    from torch.profiler import profile, ProfilerActivity
    if 1:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            mpk()
            # test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/
    #########################################################
    
    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/mirage-project/mirage
    # pip install -e . -v
    # export MIRAGE_HOME=$(pwd)
    # python demo/demo_debug_mlp.py --use-mirage
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python...
    # "kernel"
    # python scripts/display_task_graph.py ./gen/task_graph.json
