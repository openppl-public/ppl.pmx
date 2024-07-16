import fire
import sys
import os
import json
import torch

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import intern_vit.modeling.Loader as Loader
import intern_vit.modeling.Params as Params

def main(
    ckpt_dir: str,
    export_path: str,
    with_proj_head: bool = True, # use projection head to cls
    fused_qkv: bool = True, # fuse qkv linear
):

    with open(Path(ckpt_dir) / "opmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: Params.ViTParams = Params.ViTParams(**params)

    model = Loader.load(
        ckpt_dir=ckpt_dir,
        model_params=params,
        with_proj_head=with_proj_head,
        fused_qkv=fused_qkv,
        attn_wqkv_bias_term=False,
        attn_wo_bias_term=True,
        ffn_linear_bias_term=True,
        load_to_cpu=True
    )

    # export model
    pixel_values = torch.ones([1, 3, params.image_size, params.image_size], dtype=torch.float16)
    attn_mask = torch.empty(0, dtype=torch.float16)

    local_rank = torch.distributed.get_rank(group=model.proc_group)
    model_path = os.path.join(export_path, "model_slice_{}".format(local_rank))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.onnx.export(
        model.cpu(),
        (pixel_values, attn_mask),
        os.path.join(model_path, "model.onnx"),
        input_names=["pixel_values", "attn_mask"],
        output_names=["vision_logits"],
        do_constant_folding=True,
        opset_version=11,
        dynamic_axes={
            'pixel_values': {
                0: 'batch',
                2: 'image_h',
                3: 'image_w',
            }
        }
    )

    if local_rank == 0:
        with open(os.path.join(export_path, "params.json"), "w") as f:
            json.dump(model.params.__dict__, f)


if __name__ == "__main__":
    fire.Fire(main)
