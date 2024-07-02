import fire
import sys
import os
import json
import torch

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import bert.modeling.Loader as Loader
import bert.modeling.Params as Params

def main(
    ckpt_dir: str,
    export_path: str,
    with_proj_head: bool = True, # use projection head to cls
    fused_qkv: bool = True, # fuse qkv linear
):

    with open(Path(ckpt_dir) / "opmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: Params.BertParams = Params.BertParams(**params)

    model = Loader.load(
        ckpt_dir=ckpt_dir,
        model_params=params,
        with_proj_head=with_proj_head,
        fused_qkv=fused_qkv,
        attn_wqkv_bias_term=True,
        attn_wo_bias_term=True,
        ffn_linear_bias_term=True,
        load_to_cpu=True
    )

    # export model
    input_ids = torch.tensor([[ 101, 3416,  891, 3144, 2945,  118,  122,  102   ],
                              [ 101, 3416,  891, 3144, 2945,  118,  123,  102  ]], dtype=torch.int64)

    input_shape = input_ids.shape
    seq_length = input_shape[1]
    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    position_ids = torch.arange(seq_length).expand((1, -1)).to(input_ids.device)

    attn_mask = torch.empty(0, dtype=torch.float16)

    local_rank = torch.distributed.get_rank(group=model.proc_group)
    model_path = os.path.join(export_path, "model_slice_{}".format(local_rank))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.onnx.export(
        model.cpu(),
        (input_ids, token_type_ids, position_ids, attn_mask),
        os.path.join(model_path, "model.onnx"),
        input_names=["input_ids", "token_type_ids", "position_ids", "attn_mask"],
        output_names=["bert_logits"],
        do_constant_folding=True,
        opset_version=11,
    )

    if local_rank == 0:
        with open(os.path.join(export_path, "params.json"), "w") as f:
            json.dump(model.params.__dict__, f)


if __name__ == "__main__":
    fire.Fire(main)
