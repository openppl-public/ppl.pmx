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
import gc
import json
import os
import shutil
import warnings

import torch

from tqdm import tqdm
from pathlib import Path
from torch_function.WeightOnlyQuantUtils import Int4QuantUtils, pseudo_quantize_linear_weight

"""
This method not only convert the weight, but also quant the weight

Sample usage:

```
python ConvertWeightToOpmx.py \
    --input_dir /path/to/downloaded/hf/weights/7B --output_dir /output/path --quant True
```

Thereafter, models can be quant to weight only:

```
Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_pmx_model(model_path, input_base_path, model_type, quant, group_size, n_bits, has_zeropoint, storage_bits):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert opmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['rms_norm_eps']
    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])

    # compute intermediate_size
    hidden_dim = pmx_params_dict['hidden_dim']
    multiple_of = params.get("multiple_of", 256)
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1)
    if "intermediate_size" in params.keys():
        pmx_params_dict['intermediate_dim'] = params.get("intermediate_size")
    else:
        pmx_params_dict['intermediate_dim'] = compute_intermediate_size(hidden_dim, ffn_dim_multiplier, multiple_of)

    for key in ('max_position_embeddings', 'rope_theta'):
        if key in params:
            pmx_params_dict[key] = params[key]

    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    # TO DO: GQA / MQA, only test on llama
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']
    dims_per_head = hidden_dim // num_heads
    key_value_dim = dims_per_head * num_kv_heads

    # load weights
    def unpermute(w, n_heads=num_heads, dim1=hidden_dim, dim2=hidden_dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    hf_model_state_dict, state_dict = {}, {}

    if model_type is None:
        if any(Path(input_base_path).glob("*.safetensors")):
            model_type = "safetensors"
        else:
            model_type = "bin"

    if model_type == "bin":
        for ckpt_path in sorted(Path(input_base_path).glob("*.bin")):
            hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))
    elif model_type == "safetensors":
        from safetensors import safe_open
        for ckpt_path in sorted(Path(input_base_path).glob("*.safetensors")):
            weights = safe_open(ckpt_path, 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
            hf_model_state_dict.update(weights)
    else:
        raise ValueError(f"Not support the model_type: {model_type}.")

    for layer_i in tqdm(range(pmx_params_dict['num_layers']), desc="Processing layers"):

        wq = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"])
        wk = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"], num_kv_heads, key_value_dim, hidden_dim)
        wv = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]
        wo = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]

        ffn1 = hf_model_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"]
        ffn2 = hf_model_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"]
        ffn3 = hf_model_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"]

        if quant:
            wq, wq_scale, wq_zero_point = pseudo_quantize_linear_weight(wq, n_bits, has_zeropoint, group_size)
            wk, wk_scale, wk_zero_point = pseudo_quantize_linear_weight(wk, n_bits, has_zeropoint, group_size)
            wv, wv_scale, wv_zero_point = pseudo_quantize_linear_weight(wv, n_bits, has_zeropoint, group_size)
            wo, wo_scale, wo_zero_point = pseudo_quantize_linear_weight(wo, n_bits, has_zeropoint, group_size)
            
            ffn1, ffn1_scale, ffn1_zero_point = pseudo_quantize_linear_weight(ffn1, n_bits, has_zeropoint, group_size)
            ffn2, ffn2_scale, ffn2_zero_point = pseudo_quantize_linear_weight(ffn2, n_bits, has_zeropoint, group_size)
            ffn3, ffn3_scale, ffn3_zero_point = pseudo_quantize_linear_weight(ffn3, n_bits, has_zeropoint, group_size)
            
            
            if has_zeropoint:
                # quant + zeropoint
                qwq = Int4QuantUtils.quantize_fp16_to_int4(wq, wq_scale, wq_zero_point, group_size, n_bits)
                packed_qwq = Int4QuantUtils.pack(qwq, storage_bits=storage_bits)
                qwk = Int4QuantUtils.quantize_fp16_to_int4(wk, wk_scale, wk_zero_point, group_size, n_bits)
                packed_qwk = Int4QuantUtils.pack(qwk, storage_bits=storage_bits)
                qwv = Int4QuantUtils.quantize_fp16_to_int4(wv, wv_scale, wv_zero_point, group_size, n_bits)
                packed_qwv = Int4QuantUtils.pack(qwv, storage_bits=storage_bits)
                qwo = Int4QuantUtils.quantize_fp16_to_int4(wo, wo_scale, wo_zero_point, group_size, n_bits)
                packed_qwo = Int4QuantUtils.pack(qwo, storage_bits=storage_bits)
                
                qffn1 = Int4QuantUtils.quantize_fp16_to_int4(ffn1, ffn1_scale, ffn1_zero_point, group_size, n_bits)
                packed_qffn1 = Int4QuantUtils.pack(qffn1, storage_bits=storage_bits)
                qffn2 = Int4QuantUtils.quantize_fp16_to_int4(ffn2, ffn2_scale, ffn2_zero_point, group_size, n_bits)
                packed_qffn2 = Int4QuantUtils.pack(qffn2, storage_bits=storage_bits)
                qffn3 = Int4QuantUtils.quantize_fp16_to_int4(ffn3, ffn3_scale, ffn3_zero_point, group_size, n_bits)
                packed_qffn3 = Int4QuantUtils.pack(qffn3, storage_bits=storage_bits)
                
                state_dict.update({
                #Q
                f"layers.{layer_i}.attention.wq.qweight": packed_qwq,
                f"layers.{layer_i}.attention.wq.scale": wq_scale,
                f"layers.{layer_i}.attention.wq.zeropoint": wq_zero_point,
                #K
                f"layers.{layer_i}.attention.wk.qweight": packed_qwk,
                f"layers.{layer_i}.attention.wk.scale": wk_scale,
                f"layers.{layer_i}.attention.wk.zeropoint": wk_zero_point,
                #V
                f"layers.{layer_i}.attention.wv.qweight": packed_qwv,
                f"layers.{layer_i}.attention.wv.scale": wv_scale,
                f"layers.{layer_i}.attention.wv.zeropoint": wv_zero_point,
                #O
                f"layers.{layer_i}.attention.wo.qweight": packed_qwo,
                f"layers.{layer_i}.attention.wo.scale": wo_scale,
                f"layers.{layer_i}.attention.wo.zeropoint": wo_zero_point,
                #ffn1
                f"layers.{layer_i}.feed_forward.w1.qweight": packed_qffn1,
                f"layers.{layer_i}.feed_forward.w1.scale": ffn1_scale,
                f"layers.{layer_i}.feed_forward.w1.zeropoint": ffn1_zero_point,
                #ffn2
                f"layers.{layer_i}.feed_forward.w2.qweight": packed_qffn2,
                f"layers.{layer_i}.feed_forward.w2.scale": ffn2_scale,
                f"layers.{layer_i}.feed_forward.w2.zeropoint": ffn2_zero_point,
                #ffn3
                f"layers.{layer_i}.feed_forward.w3.qweight": packed_qffn3,
                f"layers.{layer_i}.feed_forward.w3.scale": ffn3_scale,
                f"layers.{layer_i}.feed_forward.w3.zeropoint": ffn3_zero_point,
                #other
                f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
                f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
            })
                
            else:
                # only quant
                qwq = Int4QuantUtils.quantize_fp16_to_int4(wq, wq_scale, None, group_size, n_bits)
                packed_qwq = Int4QuantUtils.pack(qwq, storage_bits=32)
                qwk = Int4QuantUtils.quantize_fp16_to_int4(wk, wk_scale, None, group_size, n_bits)
                packed_qwk = Int4QuantUtils.pack(qwk, storage_bits=32)
                qwv = Int4QuantUtils.quantize_fp16_to_int4(wv, wv_scale, None, group_size, n_bits)
                packed_qwv = Int4QuantUtils.pack(qwv, storage_bits=32)
                qwo = Int4QuantUtils.quantize_fp16_to_int4(wo, wo_scale, None, group_size, n_bits)
                packed_qwo = Int4QuantUtils.pack(qwo, storage_bits=32)
                
                qffn1 = Int4QuantUtils.quantize_fp16_to_int4(ffn1, ffn1_scale, None, group_size, n_bits)
                packed_qffn1 = Int4QuantUtils.pack(qffn1, storage_bits=32)
                qffn2 = Int4QuantUtils.quantize_fp16_to_int4(ffn2, ffn2_scale, None, group_size, n_bits)
                packed_qffn2 = Int4QuantUtils.pack(qffn2, storage_bits=32)
                qffn3 = Int4QuantUtils.quantize_fp16_to_int4(ffn3, ffn3_scale, None, group_size, n_bits)
                packed_qffn3 = Int4QuantUtils.pack(qffn3, storage_bits=32)
                
                state_dict.update({
                    #Q
                    f"layers.{layer_i}.attention.wq.qweight": packed_qwq,
                    f"layers.{layer_i}.attention.wq.scale": wq_scale,
                    #K
                    f"layers.{layer_i}.attention.wk.qweight": packed_qwk,
                    f"layers.{layer_i}.attention.wk.scale": wk_scale,
                    #V
                    f"layers.{layer_i}.attention.wv.qweight": packed_qwv,
                    f"layers.{layer_i}.attention.wv.scale": wv_scale,
                    #O
                    f"layers.{layer_i}.attention.wo.qweight": packed_qwo,
                    f"layers.{layer_i}.attention.wo.scale": wo_scale,
                    #ffn1
                    f"layers.{layer_i}.feed_forward.w1.qweight": packed_qffn1,
                    f"layers.{layer_i}.feed_forward.w1.scale": ffn1_scale,
                    #ffn2
                    f"layers.{layer_i}.feed_forward.w2.qweight": packed_qffn2,
                    f"layers.{layer_i}.feed_forward.w2.scale": ffn2_scale,
                    #ffn3
                    f"layers.{layer_i}.feed_forward.w3.qweight": packed_qffn3,
                    f"layers.{layer_i}.feed_forward.w3.scale": ffn3_scale,
                    #other
                    f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
                    f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
                })

        else:
            # fp16
            state_dict.update({
                f"layers.{layer_i}.attention.wq.weight": wq,
                f"layers.{layer_i}.attention.wk.weight": wk,
                f"layers.{layer_i}.attention.wv.weight": wv,
                f"layers.{layer_i}.attention.wo.weight": wo,
                f"layers.{layer_i}.feed_forward.w1.weight": ffn1,
                f"layers.{layer_i}.feed_forward.w2.weight": ffn2,
                f"layers.{layer_i}.feed_forward.w3.weight": ffn3,
                f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
                f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
            })
    
    state_dict.update({
        "tok_embeddings.weight": hf_model_state_dict["model.embed_tokens.weight"],
        "norm.weight": hf_model_state_dict["model.norm.weight"],
        "output.weight": hf_model_state_dict["lm_head.weight"]
    })
    torch.save(state_dict, os.path.join(model_path, "model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write OPMX model",
    )
    parser.add_argument(
        "--model_type",
        choices=["bin", "safetensors"],
        default=None,
        help="Input model type",
    )
    parser.add_argument(
    "--quant",
    default=False,
    help="Enable quantization for the model. Set to True to quantize the model weights.",
    )
    parser.add_argument(
        "--group_size",
        default=128,
        help="Specify the size of groups for quantization. Determines how weights are grouped for quantization.",
    )
    parser.add_argument(
        "--n_bits",
        default=4,
        help="Set the number of bits for quantization. Determines the precision of the quantized weights.",
    )
    parser.add_argument(
        "--has_zeropoint",
        default=False,
        help="Include zero-point in quantization. Set to True to use zero-point quantization.",
    )
    parser.add_argument(
        "--storage_bits",
        default=32,
        help="Specify the number of bits for packing quantized values. Determines the storage size for quantized data.",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_type=args.model_type,
        quant=args.quant,
        group_size=args.group_size,
        n_bits=args.n_bits,
        has_zeropoint=args.has_zeropoint,
        storage_bits=args.storage_bits
    )

if __name__ == "__main__":
    main()
