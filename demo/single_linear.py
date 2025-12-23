
import torch
import argparse
import mirage as mi

from pkt_util import TestUtil, TorchRef, PersistentKernelTest

if __name__ == "__main__":
    # 只支持8的倍数，gridSize需要能N被整除。
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
    hidden_size = 2560        # K
    intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col/
    x_torch = torch.ones((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_torch = TestUtil.create_matrix_arange_row((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w = mpk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mpk.attach_input(torch_tensor=out_torch, name="linear_out")

    if splitk == 1:
        mpk.linear_layer(
            input=x,
            weight=w,
            output=linear_out,
            grid_dim=(38, 1, 1),  # (9728 * 2) / 128 / 4 = 38 / 19 / 8
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
    
    
    # ###
    def ref_run():
        return TorchRef.linear(x_torch, w_torch)
    def mpk_run():
        mpk(batch_size)
        
    ref_output = ref_run()
    # print("ref_output", ref_output)
    # mpk(batch_size)
    # print("out_torch", out_torch)
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    ###
    
    pkt.generate_report(mpk_run, out_torch, splitk, 
                        ref_run, ref_output, 
                        warnup_iter=100, test_iter=200, 
                        allclose_iter=5, print_all=False)