import inspect

import torch
import transformers
from transformers import Qwen2Tokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer


def main() -> None:
    print("transformers", transformers.__version__)
    print("lazy_initialization", inspect.signature(DynamicLayer.lazy_initialization))
    print("DynamicLayer.update", inspect.signature(DynamicLayer.update))
    key, value = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    cache = DynamicCache()
    layer = DynamicLayer()
    parameters = inspect.signature(DynamicLayer.lazy_initialization).parameters
    if "value_states" in parameters:
        layer.lazy_initialization(key, value)
    else:
        layer.lazy_initialization(key)
    layer.keys, layer.values = key, value
    cache.layers.append(layer)
    print("shared", cache.layers[0].keys is key, cache.layers[0].values is value, len(cache.layers), cache.get_seq_length())
    online = DynamicCache()
    online.update(key, value, 0)
    print("update copies", online.layers[0].keys.data_ptr() != key.data_ptr(), "getitem", hasattr(online, "__getitem__"))
    print("Qwen2Tokenizer.__init__", inspect.signature(Qwen2Tokenizer.__init__))
    print("to_str", hasattr(Qwen2Tokenizer, "backend_tokenizer"))


if __name__ == "__main__":
    main()
