"""Merge downloaded PointQ-Bench byte parts and verify the full ZIP SHA-256."""

import argparse
import hashlib
from pathlib import Path


NAME = "PointQ-Bench_pointclouds_v1.0.zip"
SHA256 = "da2e1f58902811be130959bb3f3f43ae44c00a788558af7493eb5c06a969985b"
BYTES = 9959561286


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read and verify parts without creating the ZIP")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    parts = [root / f"{NAME}.{i:03d}" for i in range(1, 5)]
    checksums = root / "PointQ-Bench_PARTS_SHA256SUMS_v1.0.txt"
    expected = dict((line.split(maxsplit=1)[1].strip(), line.split(maxsplit=1)[0]) for line in checksums.read_text(encoding="ascii").splitlines() if line.strip())
    for path in parts:
        if not path.is_file() or path.name not in expected:
            raise FileNotFoundError("Missing part/checksum: " + path.name)
    destination = root / NAME
    temporary = root / (NAME + ".merge.part")
    if not args.verify_only and (destination.exists() or temporary.exists()):
        raise FileExistsError("Output already exists; use --verify-only or run in a new download folder")
    output = None if args.verify_only else temporary.open("xb")
    digest = hashlib.sha256()
    count = 0
    try:
        for path in parts:
            part_digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    part_digest.update(block)
                    count += len(block)
                    if output:
                        output.write(block)
            if part_digest.hexdigest() != expected[path.name]:
                raise ValueError("Part SHA-256 mismatch: " + path.name)
            print("Verified " + path.name, flush=True)
    finally:
        if output:
            output.close()
    if count != BYTES or digest.hexdigest() != SHA256:
        raise ValueError("Full archive size/SHA-256 mismatch")
    if not args.verify_only:
        temporary.rename(destination)
    print("OK: 3083-point-cloud archive SHA-256 matches the verified release package")


if __name__ == "__main__":
    main()
