# Usage example

## Download model from Hugging Face

Download the model file from the [Hugging Face](https://huggingface.co/models).

## Convert and quant model params

Add the PYTHONPATH of ppl.pmx
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ppl.pmx
```

To address inconsistencies with the implementation of Hugging Face's RotaryPositionEmbedding function, you can use the following script to convert the weight parameters:
```bash
python huggingface/ConvertWeightToOpmx.py --input_dir <hf_model_dir> --output_dir <pmx_model_dir>
```
#### Quantization
This script also handles the quantization process. If you want to quantize the LLaMA3 model with weight-only quantization, use the command below:

> Note: There are additional quantization configurations (group_size, n_bits, storage_bits) available in ConvertWeightToOpmx.py. Refer to the script for more details.

```bash
python huggingface/ConvertWeightToOpmx.py --input_dir <hf_model_dir> --output_dir <pmx_model_dir> --quant 1 
```
After the conversion, you will find the OPMX model file in <pmx_model_dir>.

## Spliting model

Quantize not support spliting model now

## Merging model

Quantize not support merging model now

## Testing Model

The `Demo.py` script provides functionality to test the model for correctness before exporting.

> Note: There are additional quantization configurations (group_size, n_bits, storage_bits) available in Demo.py. Refer to the script for more details.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu huggingface/Demo.py --ckpt_dir <convert_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --quant_data_type "int4" --quant_method "weight_only" --quant_axis 1 --group_size 128 --storage_bits 32
```

- `OMP_NUM_THREADS`: This parameter determines the number of OpenMP threads. It is set to 1 to prevent excessive CPU core usage. Each PyTorch process opens an OpenMP thread pool, and setting it to 1 avoids occupying too many CPU cores.
- `--nproc_per_node`: Specifies the number of model slices per node.

## Exporting Model

To export a model, you will use the `Export.py` script provided. Here's an example command for exporting a 13B model with 1 GPU:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu huggingface/Export.py --ckpt_dir <convert_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --quant_data_type "int4" --quant_method "weight_only" --quant_axis 1 --group_size 128 --storage_bits 32 --export_path <export_dir>
```

Make sure to replace `$num_gpu` with the actual number of GPUs you want to use.

## Generating Test Data

This script demonstrates how to generate test data for steps 0, 1, and 255 using the specified command.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu huggingface/Demo.py --ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --seqlen_scale_up 1 --max_gen_len 256 --dump_steps 0,1,255 --dump_tensor_path <dump_dir>  --batch 1 --quant_data_type "int4" --quant_method "weight_only" --quant_axis 1 --group_size 128 --storage_bits 32
```

- `seqlen_scale_up`: Scale factor for input byte size (sequence length scaled up by 8).
- `max_gen_len`: Specifies the maximum generated output length in bytes.
- `dump_steps`: Steps at which to dump the test data.
- `dump_tensor_path`: Path to store the dumped test data.
- `batch`: Specifies the batch size for data processing.

Make sure to replace `<llama_dir>` , `<llama_tokenizer_dir>` and `<dump_tensor_path>`with the actual directory paths in your environment.
