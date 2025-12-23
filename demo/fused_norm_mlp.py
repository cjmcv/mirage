
import torch
import argparse
import mirage as mi

from pkt_util import TorchRef, PersistentKernelTest

if __name__ == "__main__":
    max_batch_size = 16
    batch_size = 2
    parser = argparse.ArgumentParser()
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

    
    pkt = PersistentKernelTest(world_size, rank, args.trace_name, args.profiling)
    mpk = pkt.get_mpk()
    
    pkt.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    # (batch_size, 38, 19, 20)
    # (batch_size, 76, 38, 40)
    gridsize = [max_batch_size, 76, 38, 40]
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_rms = mpk.attach_input(torch_tensor=w_rms_torch, name="w_rms")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    rmsnorm_out = mpk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
    mpk.rmsnorm_layer(
        input=x,
        weight=w_rms,
        output=rmsnorm_out,
        grid_dim=(gridsize[0], 1, 1),
        block_dim=(128, 1, 1),
    )
    
    # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    mlp_mid = mpk.new_tensor(dims=(max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=rmsnorm_out,
        weight=w_gatedup,
        output=mlp_mid,
        grid_dim=(gridsize[1], 1, 1),
        block_dim=(128, 1, 1),
    )
    
    # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    # out_torch = silu_mul_out_torch
    silu_mul_out = mpk.new_tensor(dims=(max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mpk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(gridsize[2], 1, 1),
        block_dim=(128, 1, 1),
    )
    if splitk == 1:
        mpk.linear_with_residual_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj,
            residual=x,
            output=mlp_out,
            grid_dim=(gridsize[3], 1, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    else:
        mpk.linear_postfix_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj,
            output=mlp_out,
            grid_dim=(splitk, 16, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    
    pkt.compile_load(args.nc, args.output_dir)
    
    # pkt.memory_footprint_simulation(rank)
        
    ###

    def ref_run():
        return TorchRef.norm_mlp(x_torch[:batch_size], w_rms_torch, w_gatedup_torch, w_down_proj_torch) + x_torch[:batch_size]
    def mpk_run():
        mpk(batch_size)
        
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    mpk_output = out_torch[:batch_size]
    ###
    
    if (args.profiling):
        mpk(batch_size)
        print("Finish profiling.")
        exit()
        
    pkt.generate_report(mpk_run, mpk_output, splitk, 
                        graph.replay, ref_output, 
                        warnup_iter=100, test_iter=200, 
                        allclose_iter=5, print_all=False)

