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
    # vision_config
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['layer_norm_eps']

    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['type_vocab_size'] = params['type_vocab_size']
    pmx_params_dict['position_embedding_type'] = params['position_embedding_type']
    pmx_params_dict['max_position_embeddings'] = params['max_position_embeddings']

    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])

    # compute intermediate_size
    pmx_params_dict['intermediate_dim'] = params.get('intermediate_size')
    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.bin")):
        hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for layer_i in range(pmx_params_dict['num_layers']):

        wq = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.query.weight"]
        wk = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.key.weight"]
        wv = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.value.weight"]
        wq_bias = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.query.bias"]
        wk_bias = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.key.bias"]
        wv_bias = hf_model_state_dict[f"encoder.layer.{layer_i}.attention.self.value.bias"]

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wq.bias": wq_bias,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wk.bias": wk_bias,
            f"layers.{layer_i}.attention.wv.weight": wv,
            f"layers.{layer_i}.attention.wv.bias": wv_bias,

            f"layers.{layer_i}.attention.wo.weight": hf_model_state_dict[f"encoder.layer.{layer_i}.attention.output.dense.weight"],
            f"layers.{layer_i}.attention.wo.bias": hf_model_state_dict[f"encoder.layer.{layer_i}.attention.output.dense.bias"],

            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"encoder.layer.{layer_i}.intermediate.dense.weight"],
            f"layers.{layer_i}.feed_forward.w1.bias": hf_model_state_dict[f"encoder.layer.{layer_i}.intermediate.dense.bias"],

            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"encoder.layer.{layer_i}.output.dense.weight"],
            f"layers.{layer_i}.feed_forward.w2.bias": hf_model_state_dict[f"encoder.layer.{layer_i}.output.dense.bias"],
            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"encoder.layer.{layer_i}.attention.output.LayerNorm.weight"],
            f"layers.{layer_i}.attention_norm.bias": hf_model_state_dict[f"encoder.layer.{layer_i}.attention.output.LayerNorm.bias"],

            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"encoder.layer.{layer_i}.output.LayerNorm.weight"],
            f"layers.{layer_i}.ffn_norm.bias": hf_model_state_dict[f"encoder.layer.{layer_i}.output.LayerNorm.bias"],
        })

    state_dict.update({
        "input_embeddings.weight": hf_model_state_dict['embeddings.word_embeddings.weight'],
        "token_type_embeddings.weight": hf_model_state_dict['embeddings.token_type_embeddings.weight'],
        "position_embeddings.weight": hf_model_state_dict['embeddings.position_embeddings.weight'],

        "pre_layernorm.weight": hf_model_state_dict['embeddings.LayerNorm.weight'],
        "pre_layernorm.bias": hf_model_state_dict['embeddings.LayerNorm.bias'],
    })

    if "pooler.dense.weight" in hf_model_state_dict.keys():
        state_dict.update({"pool_projection.dense.weight": hf_model_state_dict["pooler.dense.weight"],
                           "pool_projection.dense.bias": hf_model_state_dict["pooler.dense.bias"]})


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
