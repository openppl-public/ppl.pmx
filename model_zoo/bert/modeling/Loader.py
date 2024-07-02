import os
import sys
import torch
import time

from pathlib import Path
from typing import List

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from bert.modeling.static_batching.Model import TensorDumper, BertTransformer
import bert.modeling.Params as Params
import ModelParallel


def load(
    ckpt_dir: str,
    model_params: Params.BertParams,
    with_proj_head: bool, # use projection head
    fused_qkv: bool, # fuse qkv linear
    attn_wqkv_bias_term: bool,
    attn_wo_bias_term: bool,
    ffn_linear_bias_term: bool,
    load_to_cpu: bool,
    dump_tensor_path: str = None,
):
    start_time = time.time()
    local_rank, world_size = ModelParallel.setup(load_to_cpu)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    ckpt_path = checkpoints[local_rank]

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    proc_group = dist.new_group(ranks=[_ for _ in range(world_size)], backend=
                                'gloo' if load_to_cpu else 'nccl')

    if load_to_cpu:
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = BertTransformer(
            model_params,
            with_proj_head,
            fused_qkv,
            attn_wqkv_bias_term,
            attn_wo_bias_term,
            ffn_linear_bias_term,
            proc_group=proc_group)
    torch.set_default_tensor_type(torch.FloatTensor)

    model.load_state_dict(checkpoint)

    if dump_tensor_path is not None:
        dump_path = os.path.join(dump_tensor_path, "rank_{}".format(local_rank))
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        TensorDumper.dir = dump_path
        TensorDumper.enable_dump = True
        TensorDumper.dump_steps = [0]

    del checkpoint

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model
