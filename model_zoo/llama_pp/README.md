# Usage example

## Download model from Hugging Face

Download the model file from the [Hugging Face](https://huggingface.co/models).

## Convert model params

Due to the inconsistency with the implementation of Hugging Face's RotaryPositionEmbedding function, we need to convert the weight parameters.

```
cd ppl.pmx/model_zoo/llama/huggingface
python ConvertWeightToPmx.py --input_dir <hf_model_dir> --output_dir <pmx_model_dir>
```

you can find pmx model file in`<pmx_model_dir>` after the conversion.

## Spliting model
Here we support split model with tensor parallel and pipelien parallel in runtime, so we don't need extra split model script.

## Testing Model

The `Demo.py` script provides functionality to test the model for correctness before exporting.

```bash
OMP_NUM_THREADS=1 python Demo.py --nproc_per_node $pp_size \
--ckpt_dir <llama_dir> \
--tokenizer_path <llama_tokenizer_dir>/tokenizer.model \
--temperature 0 \
--top_p 0.95 \
--batch 4 \
--seqlen_scale_up 1 \
--unaligned_batch 0 \
--max_gen_len 16 \
--friendly_gqa 0 \
--fused_qkv 1 \
--fused_kvcache 0 \
--fused_ffn_glu 1 \
--auto_causal 1 \
--quantized_cache 1 \
--cache_layout 3 \
--cache_mode 0 \
--dynamic_batching 1 \
--pp_size $pp_size 
```

- `OMP_NUM_THREADS`: This parameter determines the number of OpenMP threads. It is set to 1 to prevent excessive CPU core usage. Each PyTorch process opens an OpenMP thread pool, and setting it to 1 avoids occupying too many CPU cores.
- `--nproc_per_node`: Specifies the number of model slices per node.

## Exporting Model

To export a model, you will use the `Export.py` script provided. Here's an example command for exporting a 13B model with 1 GPU:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $pp_size Export.py \
--ckpt_dir <llama_dir> \
--export_path <export_dir> \
--friendly_gqa 1 \
--fused_qkv 1 \
--fused_kvcache 1 \
--fused_ffn_glu 1 \
--auto_causal 1 \
--quantized_cache 1 \
--cache_layout 3 \
--cache_mode 0 \
--dynamic_batching 1 \
--pp_size $pp_size
```

Make sure to replace `$pp_size` with the actual number of GPUs you want to use.

## Generating Test Data

This script demonstrates how to generate test data for steps 0, 1, and 255 using the specified command.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Demo.py --ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --seqlen_scale_up 1 --max_gen_len 256 --dump_steps 0,1,255 --dump_tensor_path <dump_dir>  --batch 1
```

- `seqlen_scale_up`: Scale factor for input byte size (sequence length scaled up by 8).
- `max_gen_len`: Specifies the maximum generated output length in bytes.
- `dump_steps`: Steps at which to dump the test data.
- `dump_tensor_path`: Path to store the dumped test data.
- `batch`: Specifies the batch size for data processing.

Make sure to replace `<llama_dir>` , `<llama_tokenizer_dir>` and `<dump_tensor_path>`with the actual directory paths in your environment.
