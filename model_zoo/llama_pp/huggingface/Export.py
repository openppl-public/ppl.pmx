import fire
import sys
import os
import json

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama_pp.modeling.Loader as Loader
from ModelParams import ModelParams

def main(
    ckpt_dir: str,
    export_path: str,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    fused_ffn_glu: bool = True, # fuse feed forward gate linear unit
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 3, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = True, # use dynamic batching scheduling
    pp_size: int = 1,
):
    with open(Path(ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, fused_kvcache, fused_ffn_glu,
        auto_causal, quantized_cache, cache_layout,
        cache_mode, dynamic_batching,
        False, False, False, True,
        0, pp_size=pp_size
    )

    generator.export(export_path)

if __name__ == "__main__":
    fire.Fire(main)
