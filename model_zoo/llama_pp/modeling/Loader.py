import os
import sys
import torch
import time

from pathlib import Path
from typing import List

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelParams import ModelParams
from ModelUtils import __TextGenerator__
import ModelParallel


def load(
    ckpt_dir: str,
    model_params: ModelParams,
    friendly_gqa: bool, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool, # fuse qkv linear
    fused_kvcache: bool, # fuse key_value_cache and multi_head_attention
    fused_ffn_glu: bool, # fuse feed forward gate linear unit
    auto_causal: bool, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool, # 8bit kv cache quantization
    cache_layout: int, # change kv cache layout for hardware performance friendly
    cache_mode: int, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool, # use dynamic batching scheduling
    attn_wqkv_bias_term: bool,
    attn_wo_bias_term: bool,
    ffn_linear_bias_term: bool,
    load_to_cpu: bool,
    rotary_dim: int = 0,
    pp_size: int = 1,
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
) -> __TextGenerator__:
    start_time = time.time()

    if dynamic_batching:
        from llama_pp.modeling.dynamic_batching.Model_pp import TensorDumper, Transformer
        from llama_pp.modeling.dynamic_batching.Pipeline_pp import LLaMA
        if cache_layout != 3:
            print("Info: we suggest using cache_layout 3 for cuda inference performance")
    else:
        raise ValueError("we only support dynamic_batching == True")

    assert model_params.num_layers % pp_size == 0, \
        f"num_layers {model_params.num_layers} must be a multiple of pipeline parallelism size {pp_size}"
    
    local_rank, world_size = ModelParallel.setup(load_to_cpu)

    tp_size = world_size // pp_size

    assert tp_size == 1, \
        f"tensor parallelism size [{tp_size}] must equal to 1"

    dist_mapping = ModelParallel.DistMapping(world_size=world_size, rank=local_rank, tp_size=tp_size, pp_size=pp_size)

    dist_mapping.tp_proc_group = dist.new_group(ranks=dist_mapping.tp_group)
    dist_mapping.pp_proc_group = dist.new_group(ranks=dist_mapping.pp_group)

    if dist_mapping.tp_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model_params.dynamic_batching = bool(dynamic_batching)
    model_params.auto_causal = bool(auto_causal)
    model_params.cache_layout = cache_layout
    model_params.cache_mode = cache_mode
    if quantized_cache:
        model_params.cache_quant_bit = 8
        model_params.cache_quant_group = 8
    else:
        model_params.cache_quant_bit = 0
        model_params.cache_quant_group = 0

    if load_to_cpu:
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_params,
                        friendly_gqa,
                        fused_qkv,
                        fused_kvcache,
                        fused_ffn_glu,
                        attn_wqkv_bias_term,
                        attn_wo_bias_term,
                        ffn_linear_bias_term,
                        rotary_dim=rotary_dim,
                        dist_mapping=dist_mapping)
    
    torch.set_default_tensor_type(torch.FloatTensor)


    print("Loading")

    ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(ckpt_path) == 1, f"Expect only one checkpoint file in {ckpt_dir}"
    checkpoint = torch.load(ckpt_path[0], map_location="cpu")
    model.load_state_dict(checkpoint)
    del checkpoint
    # exit(0)

    generator = LLaMA(model)

    if dump_tensor_path is not None:
        dump_path = os.path.join(dump_tensor_path, "rank_{}".format(local_rank))
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        TensorDumper.dir = dump_path
        TensorDumper.enable_dump = True
        TensorDumper.dump_steps = dump_steps

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator
