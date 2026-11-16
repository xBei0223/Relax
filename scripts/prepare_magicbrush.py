import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def write_image(data: bytes, path_stem: Path) -> str:
    # 保留原始编码字节，扩展名取自实际图像格式
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        extension = image.format.lower()
    path = path_stem.with_suffix(f".{extension}")
    path.write_bytes(data)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(args.shards, key=lambda path: path.name)
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=False)
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=["img_id", "turn_index", "source_img", "instruction", "target_img"])
        for record in table.to_pylist():
            if len(rows) == args.limit:
                break
            index = len(rows)
            source = write_image(record["source_img"]["bytes"], images / f"{index:05d}_source")
            target = write_image(record["target_img"]["bytes"], images / f"{index:05d}_target")
            rows.append(
                {
                    "image": f"images/{target}",
                    "edit_image": [f"images/{source}"],
                    "prompt": record["instruction"],
                    "img_id": record["img_id"],
                    "turn_index": record["turn_index"],
                }
            )
    if len(rows) != args.limit:
        raise ValueError(f"Selected shards contain {len(rows)} rows, fewer than {args.limit}")
    metadata = json.dumps(rows, ensure_ascii=False, indent=1).encode("utf-8")
    (args.output / "train.json").write_bytes(metadata)
    selection = {
        "dataset": "osunlp/MagicBrush",
        "revision": "6c254560879df7851233ea0fd2ae2a62764d0ef2",
        "rule": "按文件名顺序读取所列 train 分片，取前 limit 行，保留原始图像字节",
        "shards": [
            {"name": shard.name, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()} for shard in shards
        ],
        "limit": args.limit,
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "unique_conditions": len({(row["prompt"], row["edit_image"][0]) for row in rows}),
    }
    (args.output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
