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
    batch: int = 1,
    with_proj_head: bool = True, # use projection head to cls
    fused_qkv: bool = True, # fuse qkv linear
    dump_tensor_path: str = None,
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
        load_to_cpu=False,
        dump_tensor_path=dump_tensor_path
    )

    attn_mask = torch.empty(0, dtype=torch.float16).to('cuda')
    input_ids = torch.tensor([[ 101, 3416,  891, 3144, 2945,  118,  122,  102  ],
                              [ 101, 3416,  891, 3144, 2945,  118,  123,  102 ]], dtype=torch.int64).to('cuda')

    input_shape = input_ids.shape
    seq_length = input_shape[1]
    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    position_ids = torch.arange(seq_length).expand((1, -1)).to(input_ids.device)

    outputs = model.forward(input_ids, token_type_ids, position_ids, attn_mask)
    print(outputs)


if __name__ == "__main__":
    fire.Fire(main)
