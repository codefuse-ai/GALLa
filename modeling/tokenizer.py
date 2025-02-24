from transformers import AutoTokenizer

def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [args.graph_pad_token]})
    # there is extra embedddings in deepseek coder. no need to resize the model
    args.graph_pad_id = tokenizer.convert_tokens_to_ids(args.graph_pad_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
