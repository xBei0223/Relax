import transformers
from transformers import Qwen2Tokenizer


def main() -> None:
    tokenizer = Qwen2Tokenizer.from_pretrained("/models/SenseNova-U1.5-8B-MoT", local_files_only=True)
    print("transformers", transformers.__version__, type(tokenizer).__mro__[:3])
    for name in (
        "vocab_files_names",
        "bpe_ranks",
        "get_vocab",
        "special_tokens_map",
        "get_added_vocab",
        "added_tokens_decoder",
        "padding_side",
        "truncation_side",
        "model_max_length",
        "split_special_tokens",
        "backend_tokenizer",
    ):
        print(name, hasattr(tokenizer, name))
    print("vocab_files_names", getattr(tokenizer, "vocab_files_names", None))
    gpt2 = __import__("transformers.models.gpt2.tokenization_gpt2", fromlist=["*"])
    print("gpt2 names", [name for name in dir(gpt2) if "unicode" in name.lower()])


if __name__ == "__main__":
    main()
