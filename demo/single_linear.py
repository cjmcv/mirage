
import torch
import argparse
import mirage as mi

from pkt_util import TestUtil, TorchRef, PersistentKernelTest

if __name__ == "__main__":
    batch_size = 128
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
    hidden_size = 2560        # K
    intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col/
    x_torch = torch.ones((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_torch = TestUtil.create_matrix_arange_row((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    linear_out_torch = torch.zeros((splitk, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w = mpk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mpk.attach_input(torch_tensor=linear_out_torch, name="linear_out")

    if splitk == 1:
        mpk.linear_layer(
            input=x,
            weight=w,
            output=linear_out,
            grid_dim=(1, 1, 1),  # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    else:
        mpk.linear_postfix_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=x,
            weight=w,
            output=linear_out,
            grid_dim=(splitk, 8, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    
    pkt.compile_load(args.nc, args.output_dir)
    
    mpk(batch_size)
    # ###
    # warnup_iter = 100
    # test_iter = 200
    
    def ref_run():
        return TorchRef.linear(x_torch, w_torch)
        
    def mpk_run():
        mpk(batch_size)
        
    # for _ in range(warnup_iter):
    #     ref_run()
    # ###
    
    # ################################################################
    print("mpk_out:", linear_out_torch)
    pkt.check_allclose(mpk_run, linear_out_torch, splitk, ref_run, 1)
        
    # pkt.time_event_record("mpk", mpk_run, test_iter)
    # pkt.time_event_record("torch_ref", ref_run, test_iter)

    # pkt.torch_profile(ref_run)
    # pkt.torch_profile(mpk_run)