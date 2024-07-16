# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os

import torch

from pathlib import Path
from safetensors import safe_open

"""
Sample usage:

```
python convert_hf_weights_to_pmx.py \
    --input_dir /path/to/downloaded/hf/weights/7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```
Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_pmx_model(model_path, input_base_path):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert pmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))
    import ipdb;ipdb.set_trace()

    # vision_config
    pmx_params_dict['hidden_dim'] = params['vision_config']['hidden_size']
    pmx_params_dict['num_heads'] = params['vision_config']['num_attention_heads']
    pmx_params_dict['num_layers'] = params['vision_config']['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['vision_config']['layer_norm_eps']
    pmx_params_dict['image_size'] = params['vision_config']['image_size']
    pmx_params_dict['patch_size'] = params['vision_config']['patch_size']
    #pmx_params_dict['projection_dim'] = params['vision_config']['projection_dim']
    pmx_params_dict['num_kv_heads'] = params['vision_config'].get('num_key_value_heads', params['vision_config']['num_attention_heads'])

    pmx_params_dict['llm_hidden_dim'] = params['llm_config']['hidden_size']
    pmx_params_dict['downsample_ratio'] = int(1 / params["downsample_ratio"])

    # TO DO: GQA / MQA, only test on llama
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']
    dims_per_head = pmx_params_dict['hidden_dim'] // num_heads
    key_value_dim = dims_per_head * num_kv_heads

    # compute intermediate_size
    pmx_params_dict['intermediate_dim'] = params['vision_config']['intermediate_size']
    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.pth")):
        hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for layer_i in range(pmx_params_dict['num_layers']):

        split_dim = [head * dims_per_head for head in [num_heads, num_kv_heads, num_kv_heads]]
        wq, wk, wv = hf_model_state_dict[f"vit.encoder.layers.{layer_i}.attn.qkv.weight"].split(split_dim, dim=0)

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wv.weight": wv,

            f"layers.{layer_i}.attention.wo.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.attn.proj.weight"],
            f"layers.{layer_i}.attention.wo.bias": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.attn.proj.bias"],

            # ls1 ls2 qk_norm
            f"layers.{layer_i}.attention.q_norm.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.attn.q_norm.weight"],
            f"layers.{layer_i}.attention.k_norm.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.attn.k_norm.weight"],
            f"layers.{layer_i}.ls1": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.ls1"],
            f"layers.{layer_i}.ls2": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.ls2"],

            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.mlp.fc1.weight"],
            f"layers.{layer_i}.feed_forward.w1.bias": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.mlp.fc1.bias"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.mlp.fc2.weight"],
            f"layers.{layer_i}.feed_forward.w2.bias": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.mlp.fc2.bias"],

            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.norm1.weight"],
            #f"layers.{layer_i}.attention_norm.bias": hf_model_state_dict[f"vision_model.encoder.layers.{layer_i}.layer_norm1.bias"],

            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"vit.encoder.layers.{layer_i}.norm2.weight"],
            #f"layers.{layer_i}.ffn_norm.bias": hf_model_state_dict[f"vision_model.encoder.layers.{layer_i}.layer_norm2.bias"],
        })

    # 'mlp1.0.bias', 'mlp1.0.weight', 'mlp1.1.bias', 'mlp1.1.weight', 'mlp1.3.bias', 'mlp1.3.weight'
    state_dict.update({
        "vision_embeddings.cls_emb_weight": hf_model_state_dict["vit.embeddings.class_embedding"],
        "vision_embeddings.patch_emb_weight": hf_model_state_dict["vit.embeddings.patch_embedding.weight"],
        "vision_embeddings.patch_emb_bias": hf_model_state_dict["vit.embeddings.patch_embedding.bias"],
        "vision_embeddings.pos_emb_weight": hf_model_state_dict["vit.embeddings.position_embedding"],
        "vision_projection.layernorm.weight": hf_model_state_dict["mlp1.0.weight"],
        "vision_projection.layernorm.bias": hf_model_state_dict["mlp1.0.bias"],
        "vision_projection.w1.weight": hf_model_state_dict["mlp1.1.weight"],
        "vision_projection.w1.bias": hf_model_state_dict["mlp1.1.bias"],
        "vision_projection.w2.weight": hf_model_state_dict["mlp1.3.weight"],
        "vision_projection.w2.bias": hf_model_state_dict["mlp1.3.bias"],
    })

    if "visual_projection.weight" in hf_model_state_dict.keys():
        state_dict.update({"vision_projection.weight": hf_model_state_dict["visual_projection.weight"]})

    torch.save(state_dict, os.path.join(model_path, "model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write PMX model",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir
    )

if __name__ == "__main__":
    main()
