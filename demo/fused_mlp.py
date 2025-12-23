
import torch
import argparse
import mirage as mi

from pkt_util import TorchRef, PersistentKernelTest

if __name__ == "__main__":
    batch_size = 8
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
    
    # pkt.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    # mlp_mid_torch = torch.zeros((batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
    mlp_mid = mpk.new_tensor(dims=(batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=x,
        weight=w_gatedup,
        output=mlp_mid,
        # grid_dim=(96, 1, 1),
        # grid_dim=(128, 1, 1),
        grid_dim=(38, 1, 1),    # (9728 * 2) / 128 = 152 / 76 / 38 / 19 / 8
        block_dim=(128, 1, 1),
    )
    
    # silu_mul_out_torch = torch.zeros((batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    # mlp_out_torch = silu_mul_out_torch
    silu_mul_out = mpk.new_tensor(dims=(batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mpk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(19, 1, 1),     # out: 512
        block_dim=(128, 1, 1),
    )
    if splitk == 1:
        mpk.linear_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj,
            output=mlp_out,
            grid_dim=(20, 1, 1),    # (2560) / 128 = 40 / 20
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
    
    pkt.compile_load(args.nc, args.output_dir)
    
    ###
    def ref_run():
        return TorchRef.mlp(x_torch, w_gatedup_torch, w_down_proj_torch)
    def mpk_run():
        mpk(batch_size)
        
    ref_output = ref_run()
    ###
    
    pkt.generate_report(mpk_run, out_torch, splitk, 
                        ref_run, ref_output, 
                        warnup_iter=100, test_iter=200, 
                        allclose_iter=5, print_all=False)