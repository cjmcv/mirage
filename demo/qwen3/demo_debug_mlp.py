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

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    args = parser.parse_args()
    world_size = 1
    rank = 0

    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    # prompt = "Give me a short introduction to large language model."
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    question = "Can you please change x axis to start from 0"
    prompt = code_text + "\n" + question
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")

    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")

    if args.profiling:
        profiler_tensor = torch.zeros(
            3000 * 128, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None
        
    spec_decode_config = mi.speculative.spec_decode_class(
        args.spec_decode,
        ngram_size=args.ngram_size,
        spec_length=args.spec_length,
    )
        
    num_workers, num_schedulers = 15, 30 #mi.get_configurations_from_gpu(rank)
    print("num_workers: ", num_workers)
    print("num_schedulers: ", num_schedulers)
    qo_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(
        args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")
    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=world_size,
        mpi_rank=rank,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=model.config.eos_token_id,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=spec_decode_config,
        use_cutlass_kernel=False,
    )
    
    batch_size = 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    mlp_out_torch = torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w2")
    mlp_out = mpk.attach_input(torch_tensor=mlp_out_torch, name="mlp_out")
    
    mlp_mid_torch = torch.zeros((batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    # mlp_mid = mpk.new_tensor(dims=(batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=x,
        weight=w_gatedup,
        output=mlp_mid,
        # grid_dim=(96, 1, 1),
        # grid_dim=(128, 1, 1),
        grid_dim=(64, 1, 1),  # (64, 1, 1)
        block_dim=(128, 1, 1),
    )
    
    # silu_mul_out_torch = torch.zeros((batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    silu_mul_out = mpk.new_tensor(dims=(batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mpk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    mpk.linear_layer(
        input=silu_mul_out,
        weight=w_down_proj,
        output=mlp_out,
        # grid_dim=(96, 1, 1),
        # grid_dim=(128, 1, 1),
        grid_dim=(64, 1, 1), # (64, 1, 1)
        block_dim=(128, 1, 1),
    )
    mpk.compile(output_dir=args.output_dir)
  
  
    ###
    warnup_iter = 20
    test_iter = 100
    for _ in range(warnup_iter):
        test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    ###
    
    ###############################################################
    mpk()
    # print("mpk: ", mlp_out_torch[0])
    torch_out = test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    # print("torch: ", torch_out[0])
    print("allclose: ", torch.allclose(mlp_out_torch[0], torch_out[0], rtol=1e-2))
    
    for _ in range(5):
        mlp_out_torch.zero_()
        print("allclose clear: ", torch.allclose(mlp_out_torch[0], torch_out[0], rtol=1e-2))
        mpk.reinitialize()
        mpk()
        print("allclose: ", torch.allclose(mlp_out_torch[0], torch_out[0], rtol=1e-2))
    ###############################################################
        
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
        
    starter.record()
    for _ in range(test_iter):
        mpk.reinitialize()
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
            mpk.reinitialize()
            mpk()
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/
    ##########################################################
    
    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/mirage-project/mirage
    # pip install -e . -v
    # export MIRAGE_HOME=$(pwd)
    # python demo/qwen3/demo_debug_mlp.py --model=/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B --use-mirage
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -o my_profile
