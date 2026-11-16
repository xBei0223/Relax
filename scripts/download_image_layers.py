import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import requests


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--aria2", default="aria2c")
    parser.add_argument("--base-url", default="https://ghcr.io/v2/redai-studio/relaxrl/blobs")
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    if hashlib.sha256(content).hexdigest() != "cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6":
        raise ValueError("Unexpected image manifest")
    manifest = json.loads(content)
    args.cache.mkdir(parents=True, exist_ok=True)
    header = ""
    if args.base_url == "https://ghcr.io/v2/redai-studio/relaxrl/blobs":
        response = requests.get("https://ghcr.io/token", params={"scope": "repository:redai-studio/relaxrl:pull", "service": "ghcr.io"}, timeout=30)
        response.raise_for_status()
        header = f"  header=Authorization: Bearer {response.json()['token']}\n"
    entries = []
    layers = {layer["digest"]: layer for layer in manifest["layers"]}
    for layer in sorted(layers.values(), key=lambda item: item["size"], reverse=True):
        digest = layer["digest"]
        path = args.cache / digest
        if path.exists() and path.stat().st_size == layer["size"] and not path.with_name(path.name + ".aria2").exists():
            if checksum(path) != digest:
                raise ValueError(f"Unexpected cached layer digest: {path}")
            continue
        entries.append(f"{args.base_url}/{digest}\n  out={digest}\n  checksum=sha-256={digest.removeprefix('sha256:')}\n{header}")
    inputs = args.cache / "download-inputs.txt"
    inputs.write_text("\n".join(entries))
    if entries:
        subprocess.run([
            args.aria2, "--input-file", str(inputs), "--dir", str(args.cache),
            "--continue=true", "--auto-file-renaming=false", "--check-integrity=true", "--file-allocation=none",
            "--max-concurrent-downloads=4", "--max-connection-per-server=8", "--split=8", "--min-split-size=16M",
            "--connect-timeout=30", "--timeout=60", "--max-tries=20", "--retry-wait=3",
            "--enable-color=false", "--show-console-readout=false", "--summary-interval=60",
            "--console-log-level=warn", "--download-result=hide",
        ], check=True)
    verified = []
    for layer in manifest["layers"]:
        path = args.cache / layer["digest"]
        if path.stat().st_size != layer["size"]:
            raise ValueError(f"Unexpected layer size: {path}")
        if checksum(path) != layer["digest"]:
            raise ValueError(f"Unexpected layer digest: {path}")
        verified.append({"digest": layer["digest"], "size": layer["size"]})
    (args.cache / "verified-layers.json").write_text(json.dumps(verified, indent=1) + "\n")


if __name__ == "__main__":
    main()
