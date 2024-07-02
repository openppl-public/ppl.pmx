# Usage example

## Download model from Hugging Face

Download the model file from the [Hugging Face](https://huggingface.co/models).

## Convert model params

Due to the inconsistency with the implementation of Hugging Face's RotaryPositionEmbedding function, we need to convert the weight parameters.

```
python ConvertWeightToOPmx.py --input_dir <hf_model_dir> --output_dir <opmx_model_dir>
```

you can find opmx model file in`<opmx_model_dir>` after the conversion.

## Testing Model

The `Demo.py` script provides functionality to test the model for correctness before exporting.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Demo.py --ckpt_dir <vit_dir>
```

- `OMP_NUM_THREADS`: This parameter determines the number of OpenMP threads. It is set to 1 to prevent excessive CPU core usage. Each PyTorch process opens an OpenMP thread pool, and setting it to 1 avoids occupying too many CPU cores.
- `--nproc_per_node`: Specifies the number of model slices per node.

## Exporting Model

To export a model, you will use the `Export.py` script provided. Here's an example command for exporting a 13B model with 1 GPU:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Export.py --ckpt_dir <vit_dir> --export_path <export_dir>
```

Make sure to replace `$num_gpu` with the actual number of GPUs you want to use.

## Generating Test Data

This script demonstrates how to generate test data for steps 0, 1, and 255 using the specified command.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Demo.py --ckpt_dir <vit_dir> --dump_tensor_path <dump_dir> 
```

- `dump_tensor_path`: Path to store the dumped test data.

Make sure to replace `<vit_dir>` and `<dump_tensor_path>` with the actual directory paths in your environment.
