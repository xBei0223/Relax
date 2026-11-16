import json

import torch.distributed as dist
from megatron.training.arguments import parse_args

from relax.utils.profile_utils import _should_profile_train_rank


def main() -> None:
    dist.init_process_group("gloo")
    args = parse_args(ignore_unknown_args=False)
    rank = dist.get_rank()
    selected = _should_profile_train_rank(args)
    if not args.use_pytorch_profiler or args.profile_ranks not in ([0], [1], [0, 1]):
        raise ValueError(f"Unexpected profiler selection on rank {rank}: {vars(args)}")
    if selected != (rank in args.profile_ranks):
        raise ValueError(f"Incorrect profiler selection on rank {rank}")
    if (args.profile_step_start, args.profile_step_end) not in ((62, 63), (63, 63)):
        raise ValueError("Unexpected profiler capture window")
    print(json.dumps({"rank": rank, "selected": selected, "profile_ranks": args.profile_ranks, "profile_step_start": args.profile_step_start, "profile_step_end": args.profile_step_end}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
