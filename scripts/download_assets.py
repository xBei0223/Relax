import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def download(entry: dict, base: str, output: Path) -> dict:
    name = entry["rfilename"]
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != entry["size"] or path.with_name(path.name + ".aria2").exists():
        subprocess.run([
            "aria2c", "--continue=true", "--auto-file-renaming=false", "--file-allocation=none",
            "--max-connection-per-server=8", "--split=8", "--min-split-size=16M",
            "--connect-timeout=30", "--timeout=60", "--lowest-speed-limit=16K",
            "--max-tries=20", "--retry-wait=3", "--enable-color=false", "--show-console-readout=false",
            "--summary-interval=60", "--console-log-level=warn", "--download-result=hide",
            "--dir", str(path.parent), "--out", path.name, f"{base}/{name}",
        ], check=True)
    size = path.stat().st_size
    if size != entry["size"]:
        raise ValueError(f"Size mismatch: {name}: {size} != {entry['size']}")
    lfs = entry.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    expected = lfs["sha256"] if lfs else entry["blobId"]
    if digest.hexdigest() != expected:
        raise ValueError(f"Digest mismatch: {name}")
    result = {"name": name, "size": size, "digest": digest.hexdigest(), "algorithm": digest.name}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    configurations = [
        (
            "models", "sensenova/SenseNova-U1.5-8B-MoT", "a2462cc1b549bace2c0896143bbbcbb3fa217c76",
            "https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT/resolve/master",
            args.root / "models/SenseNova-U1.5-8B-MoT",
        ),
        (
            "datasets", "osunlp/MagicBrush", "6c254560879df7851233ea0fd2ae2a62764d0ef2",
            f"{args.endpoint}/datasets/osunlp/MagicBrush/resolve/6c254560879df7851233ea0fd2ae2a62764d0ef2",
            args.root / "data/MagicBrush",
        ),
    ]
    for kind, repository, revision, base, output in configurations:
        response = requests.get(
            f"{args.endpoint}/api/{kind}/{repository}/revision/{revision}",
            params={"blobs": "true"}, timeout=60,
        )
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != revision:
            raise ValueError(f"Unexpected revision: {metadata['sha']}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "hf-revision.json").write_text(json.dumps(metadata, indent=2) + "\n")
        files = [entry for entry in metadata["siblings"] if entry["rfilename"] not in {".gitattributes", "README.md", "README_CN.md"}]
        if kind == "datasets":
            files = sorted((entry for entry in files if entry["rfilename"].startswith("data/train-")), key=lambda entry: entry["rfilename"])[:4]
            if len(files) != 4:
                raise ValueError("Expected four training shards")
        with ThreadPoolExecutor(max_workers=3) as pool:
            verified = list(pool.map(lambda entry: download(entry, base, output), files))
        (output / "verified-files.json").write_text(json.dumps(verified, indent=2) + "\n")


if __name__ == "__main__":
    main()
