from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcgs_sr3d.data import write_data_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CIGBench/Private layout and write the split manifest.")
    parser.add_argument("--cigbench-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--split-seed", type=int, default=20260813)
    args = parser.parse_args()

    manifest = write_data_manifest(
        args.output,
        args.cigbench_root,
        args.private_root,
        split_seed=args.split_seed,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
