
import torch
import argparse
import mirage as mi

from common.pkt_util import TorchRef, MpkReporter, TestUtil
from common.mpk_layers import MpkLayers

if __name__ == "__main__":
    max_batch_size = 1
    batch_size = 1
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
    torch.set_default_dtype(torch.bfloat16)

    layers = MpkLayers(0, 2, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    reporter = MpkReporter() 
    # reporter.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560        # K
    intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col /
    x_torch = torch.ones((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_torch1 = TestUtil.create_matrix_arange_row((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_torch2 = TestUtil.create_matrix_arange_col((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch1.data_ptr(), "w2: ", w_torch2.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w1 = mpk.attach_input(torch_tensor=w_torch1, name="w1")
    w2 = mpk.attach_input(torch_tensor=w_torch2, name="w2")
    linear_out = mpk.attach_input(torch_tensor=out_torch, name="linear_out")
    mpk.linear_layer(
        input=x,
        weight=w1,
        output=linear_out,
        grid_dim=(38, 1, 1),  # (9728 * 2) / 8 = 2432 / ... / 76 / 38 / 19 / 8
        block_dim=(128, 1, 1),
    )
    
    mpk.mark_basic_weights(["w1"])
    mpk.append_replaceable_weights(1, ["w2"])

    layers.compile_load(args.nc, "./gen")
    
    ###########################################################
    
    
    
    # # ###
    def ref_run1():
        return TorchRef.linear(x_torch[:batch_size], w_torch1)
    def ref_run2():
        return TorchRef.linear(x_torch[:batch_size], w_torch2)
    
    def mpk_run1():
        mpk(batch_size, 0)
    def mpk_run2():
        mpk(batch_size, 1)         
    # for _ in range(10):
    #     print("Run1")
    #     mpk1(batch_size)
    #     print(out_torch[:batch_size])
        
    #     print("Run2")
    #     mpk2(batch_size)
    #     print(out_torch[:batch_size])    
            
    ref_output1 = ref_run1()
    ref_output2 = ref_run2()
    mpk_output = out_torch[:batch_size]
    
    # # print("ref_output", ref_output)
    # # mpk(batch_size)
    # # print("out_torch", out_torch)
    # # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    # #     print("allclose: True")
    # ###
    
    print("report1")
    reporter.generate_report(mpk_run1, mpk_output, splitk, 
                            ref_run1, ref_output1, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)
    print("report2")
    reporter.generate_report(mpk_run2, mpk_output, splitk, 
                            ref_run2, ref_output2, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)