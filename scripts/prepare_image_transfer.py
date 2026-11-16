import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != "cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6":
        raise ValueError(f"Unexpected manifest: {digest}")
    manifest = json.loads(content)
    digests = [digest, manifest["config"]["digest"].removeprefix("sha256:")]
    digests.extend(layer["digest"].removeprefix("sha256:") for layer in manifest["layers"])
    files = ["oci-layout", "index.json", *(f"blobs/sha256/{value}" for value in digests)]
    args.output.write_text("\n".join(files) + "\n")
    print(json.dumps({"manifest": digest, "files": len(files)}))


if __name__ == "__main__":
    main()
