from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path


def build_split(data_path: Path, seed: int) -> dict:
    records = json.loads(data_path.read_text())
    ids = [row["ID"] for row in records]
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    val_end = train_end + int(len(shuffled) * 0.15)
    splits = {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }
    return {
        "dataset": "OHR-Bench",
        "source": str(data_path),
        "seed": seed,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split_policy": "deterministic shuffle, 70/15/15",
        "counts": {name: len(values) for name, values in splits.items()},
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable FAAR OHR-Bench train/val/test split.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("OHR-Bench/data/qas_v2.json"))
    args = parser.parse_args()

    payload = build_split(args.data, args.seed)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at_utc"}
        comparable_new = {key: value for key, value in payload.items() if key != "created_at_utc"}
        if comparable_existing != comparable_new:
            raise SystemExit(f"Refusing to overwrite immutable split with different contents: {args.out}")
        print(f"Existing immutable split verified unchanged at {args.out}")
        return
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "counts": payload["counts"], "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
