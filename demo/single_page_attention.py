
import torch
import argparse
from torch import nn
import mirage as mi

from pkt_util import TorchRef, MpkReporter, TestUtil
from mpk_layers import MpkLayers

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
    # model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    layers = MpkLayers(0, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    reporter = MpkReporter() 
    model, tokenizer = reporter.memory_footprint_simulation(rank)
    
    i = 0
    num_q_heads, num_kv_heads, \
    w_q_norm_torch, w_k_norm_torch, \
    w_q_torch, w_k_torch, w_v_torch, \
    k_cache_torch, v_cache_torch = reporter.get_weight_qwen3_attention(layer_id=i)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    
    splitk = 1 # 8
    hidden_size = 1024        # K
    intermediate_size = 3072 # torch.randn / ones / TestUtil.create_matrix_arange_col/
    x_torch = torch.ones((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    out = mpk.attach_input(torch_tensor=out_torch, name="out")
    # print("x: ", x_torch.data_ptr(), x_torch.num_dims, "o: ", out_torch.data_ptr())

    cos_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[0][0, :4096, :],
        name="cos_position_embedding",
    )
    sin_pos_embed = mpk.attach_input(
        torch_tensor=position_embeddings[1][0, :4096, :],
        name="sin_position_embedding",
    )
    
    w_q_norm = mpk.attach_input(torch_tensor=w_q_norm_torch, name=f"layer_{i}_q_norm")
    w_k_norm = mpk.attach_input(torch_tensor=w_k_norm_torch, name=f"layer_{i}_k_norm")
    k_cache = mpk.attach_input(torch_tensor=k_cache_torch, name=f"layer_{i}_k_cache") 
    v_cache = mpk.attach_input(torch_tensor=v_cache_torch, name=f"layer_{i}_v_cache")
    
    w_q = mpk.attach_input(torch_tensor=w_q_torch, name=f"layer_{i}_q_proj")
    w_k = mpk.attach_input(torch_tensor=w_k_torch, name=f"layer_{i}_k_proj")
    w_v = mpk.attach_input(torch_tensor=w_v_torch, name=f"layer_{i}_v_proj")
    
    max_num_batched_requests = 1
    num_local_kv_heads = num_kv_heads // world_size
    
    mpk.paged_attention_layer(
        input=x,
        k_cache=k_cache,
        v_cache=v_cache,
        q_norm=w_q_norm,
        k_norm=w_k_norm,
        cos_pos_embed=cos_pos_embed,
        sin_pos_embed=sin_pos_embed,
        output=out,
        grid_dim=(max_num_batched_requests, num_local_kv_heads, 1),
        block_dim=(128, 1, 1),
    )
    
    layers.compile_load(args.nc, args.output_dir)
    
    
    # ###
    # def ref_run():
    #     return TorchRef.linear(x_torch[:batch_size], w_torch)
    def mpk_run():
        mpk(batch_size)
        
    # ref_output = ref_run()
    mpk_output = out_torch[:batch_size]
    mpk_run()
    print(mpk_output)
    # print("ref_output", ref_output)
    # mpk(batch_size)
    # print("out_torch", out_torch)
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    ###
    
    # reporter.generate_report(mpk_run, mpk_output, splitk, 
    #                         ref_run, ref_output, 
    #                         warnup_iter=100, test_iter=200, 
    #                         allclose_iter=5, print_all=False)