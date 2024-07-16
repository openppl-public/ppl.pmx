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
    batch: int = 1,
    with_proj_head: bool = True, # use projection head to cls
    fused_qkv: bool = True, # fuse qkv linear
    dump_tensor_path: str = None,
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
        load_to_cpu=False,
        dump_tensor_path=dump_tensor_path
    )

    attn_mask = torch.empty(0, dtype=torch.float16).to('cuda')
    pixel_values = torch.ones([batch,3,params.image_size,params.image_size], dtype=torch.float16).to('cuda')
    #from PIL import Image
    #import requests
    #from transformers import AutoProcessor, CLIPVisionModelWithProjection
    #processor = AutoProcessor.from_pretrained("~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb")
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    #inputs = processor(images=image, return_tensors="pt")
    outputs = model.forward(pixel_values, attn_mask)
    print(outputs)


if __name__ == "__main__":
    fire.Fire(main)
