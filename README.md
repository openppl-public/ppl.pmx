# About OPMX

Open PPL Model Exchange (OPMX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. OPMX provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

Currently OPMX focus on the capabilities and hardware friendliness needed for Large Language Model(LLM) inferencing.

# Important Notice
- Support weight only quant model at [llama](model_zoo/llama3_woqu) at 16/07/2024.
- PMX has changed to OPMX at 25/04/2024. 
- And the domain of operators are also changed to `opmx`(refers to [TOC](docs/OperatorsTOC.md)).
- You can find the old code at [llm_v1](https://github.com/openppl-public/ppl.pmx/tree/llm_v1)

# Operator spec

Table of Contents: [Link](docs/OperatorsTOC.md)

About add new operator: [Link](docs/AddNewOp.md)

About update an operator's version: [Link](docs/UpdateOp.md)

# Use OPMX Python API

OPMX provides functional API based on `torch.autograd.Function`.

Clone the OPMX repo, and import `torch_function` like this:

```python
import pmx_llm.torch_function as OPMX
```

And then use it as Pytorch's functional API:

```python
norm, skip_out = OPMX.skip_rms_norm(x, weight, skip_in, -1, eps)
```

We can use these API in pytorch to custom your own model.

All OPMX function could be exported as custom operators by `torch.onnx.export`.

# Model Zoo

Some opensource model are provided in our model zoo.

Currently models:

- [LLaMA 1/2/3](model_zoo/llama)
    - [Exporting Facebook(Meta)'s LLaMA](model_zoo/llama/facebook)
    - [Exporting Huggingfaces's LLaMA](model_zoo/llama/huggingface)
- [Baichuan 1/2 7B](model_zoo/baichuan7b)
    - [Exporting Huggingfaces's Baichuan 7B](model_zoo/baichuan7b/huggingface)
- [Baichuan 1/2 13B](model_zoo/baichuan13b)
    - [Exporting Huggingfaces's Baichuan 13B](model_zoo/baichuan13b/huggingface)
- [InternLM 1](model_zoo/internlm)
    - [Exporting Huggingfaces's InternLM](model_zoo/internlm/huggingface)
- [InternLM 2](model_zoo/internlm2)
    - [Exporting Huggingfaces's InternLM](model_zoo/internlm2/huggingface)
- [ChatGLM 2/3](model_zoo/chatglm2)
    - [Exporting Huggingfaces's ChatGLM](model_zoo/chatglm2/huggingface)
- [Mixtral](model_zoo/mixtral)
    - [Exporting Huggingfaces's Mixtral](model_zoo/mixtral/huggingface)
- [Qwen 1/1.5](model_zoo/qwen)
    - [Exporting Huggingfaces's Qwen](model_zoo/qwen/huggingface)
- [Falcon](model_zoo/falcon)
    - [Exporting Huggingfaces's Falcon](model_zoo/falcon/huggingface)
- [Bigcode](model_zoo/bigcode)
    - [Exporting Huggingfaces's Bigcode](model_zoo/bigcode/huggingface)
