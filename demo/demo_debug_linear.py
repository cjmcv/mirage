from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import os
import mirage as mi

def create_matrix_arange_row(M, N, dtype=torch.bfloat16, device='cuda'):
    row_indices = torch.arange(M, dtype=dtype, device=device)
    matrix = row_indices.unsqueeze(1).expand(M, N).contiguous()  # contiguous is very important!
    return matrix

def create_matrix_arange_col(M, N, dtype=torch.bfloat16, device='cuda'):
    col_indices = torch.arange(N, dtype=dtype, device=device)
    matrix = col_indices.unsqueeze(0).expand(M, N).contiguous()
    return matrix

# print limitation
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=1, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", default="./gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=1024,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    args = parser.parse_args()
    world_size = 1
    rank = 0


    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B"
        model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")


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
        use_cutlass_kernel=False,
    )

    # x_torch = torch.randn((1, 2560), dtype=torch.bfloat16, device="cuda")
    # w_qkv_torch = torch.randn((19456, 2560), dtype=torch.bfloat16, device="cuda")
    # mpk_out_torch = torch.zeros((1, 19456), dtype=torch.bfloat16, device="cuda")


    splitk = 1
    x_torch = torch.randn((1, 9728), dtype=torch.bfloat16, device="cuda")
    w_qkv_torch = torch.randn((2560, 9728), dtype=torch.bfloat16, device="cuda")
    mpk_out_torch = torch.zeros((splitk, 2560), dtype=torch.bfloat16, device="cuda")
    # x_torch = create_matrix_arange_col(1, 256, dtype=torch.bfloat16, device="cuda")
    # w_qkv_torch = torch.ones(256, 256, dtype=torch.bfloat16, device="cuda")
    # mpk_out_torch = torch.zeros((splitk, 256), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_qkv = mpk.attach_input(torch_tensor=w_qkv_torch, name="w")
    mpk_out = mpk.attach_input(torch_tensor=mpk_out_torch, name="out")
    
    print("base ptr: ", x_torch.data_ptr(), w_qkv_torch.data_ptr(), mpk_out_torch.data_ptr())
    if splitk == 1:
        mpk.linear_layer(
            input=x,
            weight=w_qkv,
            output=mpk_out,
            grid_dim=(64, 1, 1),
            block_dim=(128, 1, 1),
        )
    else:
        mpk.linear_postfix_layer(
            input=x,
            weight=w_qkv,
            output=mpk_out,
            grid_dim=(splitk, 16, 1),
            block_dim=(128, 1, 1),
        )
    results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
    with open(f"task_graph_{rank}.json", "w") as f:
        f.write(results["json_file"])
    with open(f"kernel_{rank}.cu", "w") as f:
        f.write(results["cuda_code"])

    if args.nc is True:
        module_path = args.output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
        mpk.load_module(module_path)
    else:
        module_path = mpk.compile(output_dir=args.output_dir)
        print("module_path: ", module_path)
        mpk.load_module(module_path)
    ###############################################################
    
    ###    
    warnup_iter = 100
    test_iter = 100
    for _ in range(warnup_iter):
        O1 = F.linear(x_torch, w_qkv_torch)
    ###
    O1 = F.linear(x_torch, w_qkv_torch)
    # print("torch0: ", O1[0]) 
    mpk()
    torch.cuda.synchronize()
    
    if splitk != 1:
        # print("mpk0: ", mpk_out_torch[0], "\nmpk1: ", mpk_out_torch[1])
        for i in range(1, splitk):
            mpk_out_torch[0] += mpk_out_torch[i]
            
    # !! A potential memory out-of-bounds issue has occurred, where part of the data in O1 was overwritten during the execution of mpk()
    print("torch: ", O1[0], "\nmpk: ", mpk_out_torch[0], "\ndiff: ", O1[0] - mpk_out_torch[0])
    
    print("allclose1:", torch.allclose(mpk_out_torch[0], O1[0], rtol=2e-1, atol=2e-1))
        
    starter.record()
    for i in range(test_iter):
        mpk()
        print(i)
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    print("Best muGraph run time (ms): ", run_time / test_iter)
    
    ##
    starter.record()
    for _ in range(test_iter):
        O1 = F.linear(x_torch, w_qkv_torch)
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    print("torch run time (ms): ", run_time / test_iter)

    # if world_size > 1:
    #     dist.destroy_process_group()
        
    
