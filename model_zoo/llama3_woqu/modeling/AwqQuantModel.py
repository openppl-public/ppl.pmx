import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write OPMX model",
        required=True
    )
    parser.add_argument(
        "--pack",
        help="pack or not int4 quantize result",
        required=True
    )
    parser.add_argument(
        "--quant_type",
        help="how to quant the model",
        required=True
    )
    parser.add_argument(
        "--zero_point",
        help="how to quant the model",
        required=True
    )
    parser.add_argument(
        "--group_size",
        help="how to quant the model",
        required=True
    )
    parser.add_argument(
        "--quant_bit",
        help="how to quant the model",
        required=True
    )
    args = parser.parse_args()
    if args.quant_type == "awq":
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        quant_config = {"zero_point": args.zero_point,
                        "q_group_size": args.group_size,
                        "w_bit": args.quant_bit,
                        "version": "GEMM"}
        model = AutoAWQForCausalLM.from_pretrained(args.input_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.input_dir, trust_remote_code=True)
        model.quantize(tokenizer, quant_config=quant_config)
        if args.pack is True:
            model.pack()
        model.save_quantized(args.output_dir)

if __name__ == "__main__":
    main()