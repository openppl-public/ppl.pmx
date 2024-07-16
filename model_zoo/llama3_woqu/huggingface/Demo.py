import fire
import sys
import os
import json

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama3_woqu.modeling.Loader as Loader
from Tokenizer import Tokenizer
from ModelParams import ModelParams

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch: int = 4,
    seqlen_scale_up: int = 1,
    unaligned_batch: bool = False,
    max_gen_len: int = 256,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    fused_ffn_glu: bool = True, # fuse feed forward gate linear unit
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    # quant
    quant_data_type: str = "int4", # model quantization data type
    quant_method: str = "weight_only", # model quantization method
    quant_axis: int = 1, # model quantization axis
    group_size: int = 128, # model quantization group size
    storage_bits: int = 32, # storage bits for quantization
    has_zeropoint: bool = False, # model zeropoint
    float_zeropoint: bool = False, # model float zeropoint
    #
    dynamic_batching: bool = True, # use dynamic batching scheduling
    context_chunking: bool = True, # enable context chunking for dynamic batching
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):
    tokenizer = Tokenizer(model_path=tokenizer_path)

    with open(Path(ckpt_dir) / "opmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params,
        friendly_gqa=friendly_gqa,
        fused_qkv=fused_qkv,
        fused_kvcache=fused_kvcache,
        fused_ffn_glu=fused_ffn_glu,
        fused_alibi=False,
        auto_causal=auto_causal,
        with_rope=True,
        with_alibi=False,
        quantized_cache=quantized_cache,
        # quant
        quant_data_type=quant_data_type,
        quant_method=quant_method,
        quant_axis=quant_axis,
        group_size=group_size,
        storage_bits=storage_bits,
        has_zeropoint=has_zeropoint,
        float_zeropoint=float_zeropoint,
        #
        cache_layout=cache_layout,
        cache_mode=cache_mode,
        dynamic_batching=dynamic_batching,
        attn_wqkv_bias_term=False,
        attn_wo_bias_term=False,
        ffn_linear_bias_term=False,
        load_to_cpu=False,
        rotary_dim=0,
        dump_tensor_path=dump_tensor_path,
        dump_steps=dump_steps
    )

    generator.context_chunking = context_chunking if dynamic_batching else False

    if unaligned_batch:
        test_prompt = [        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        ]
        test_prompt = [tokenizer.encode(t, bos=True, eos=False) for t in test_prompt]

        prompt_tokens = test_prompt.copy()
        for _ in range((batch - 1) // len(test_prompt)):
            prompt_tokens.extend(test_prompt)
    else:
        test_prompt = "I believe the meaning of life is"
        test_prompt = tokenizer.encode(test_prompt, bos=True, eos=False)

        _scale_up_prompt = []
        for _ in range(seqlen_scale_up):
            _scale_up_prompt.extend(test_prompt)
        test_prompt = _scale_up_prompt

        prompt_tokens = [test_prompt for _ in range(batch)]

    print(f"prepared {len(prompt_tokens)} prompts")
    results = generator.generate(
        prompt_tokens[:batch], tokenizer.get_eos_id(), tokenizer.get_pad_id(),
        max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, top_k=0
    )

    for result in results:
        print(tokenizer.decode(result))
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
