import os
import sys

import flash_attn
import torch
import torch.distributed as dist
import transformers
from flash_attn import flash_attn_func


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    product = (x @ x).float().abs().mean()
    q = torch.randn(1, 512, 8, 128, device=device, dtype=torch.bfloat16)
    attention = flash_attn_func(q, q, q, causal=True).float().abs().mean()
    total = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(total)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product) & torch.isfinite(attention)):
        raise ValueError("GPU matrix multiplication or FlashAttention produced non-finite values")
    world = dist.get_world_size()
    if float(total) != world * (world + 1) / 2:
        raise ValueError("NCCL all_reduce produced an unexpected sum")
    print(
        rank,
        sys.version.split()[0],
        torch.__version__,
        torch.version.cuda,
        transformers.__version__,
        flash_attn.__version__,
        torch.cuda.get_device_name(device),
        float(product),
        float(attention),
        float(total),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
