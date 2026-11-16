import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sizes = {}
    for layer in manifest["layers"]:
        if not (args.cache / layer["digest"]).exists():
            sizes.setdefault(layer["size"], set()).add(layer["digest"])
    reused = []
    for source in args.sources:
        for path in source.iterdir():
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size not in sizes:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            key = f"sha256:{digest.hexdigest()}"
            target = args.cache / key
            if key in sizes[size] and not target.exists():
                os.link(path, target)
                reused.append({"digest": key, "source": str(path), "size": size})
    print(json.dumps({"bytes": sum(item["size"] for item in reused), "layers": reused}, indent=1))


if __name__ == "__main__":
    main()
