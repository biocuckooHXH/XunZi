# scripts/preprocess_data.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.preprocessor import BiomedicalTextPreprocessor


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        # support multiple jsonl/json files
        records = []
        for fp in sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json"))):
            if fp.suffix == ".jsonl":
                records.extend(list(iter_jsonl(fp)))
            else:
                records.extend(json.loads(fp.read_text(encoding="utf-8")))
        return records

    # single file
    if p.suffix == ".jsonl":
        return list(iter_jsonl(p))
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError("input must be .jsonl/.json or a directory containing them")


def dump_jsonl(items, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="data/processed")
    ap.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--task", type=str, default="cpt", choices=["cpt", "it"])
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--min-length", type=int, default=128)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input_path)
    print(f"Loaded records: {len(records)}")

    pre = BiomedicalTextPreprocessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        min_length=args.min_length,
        task=args.task,
        seed=args.seed,
    )

    processed = pre.process_records(records)
    print(f"Processed kept: {len(processed)}")

    train, val = train_test_split(processed, test_size=args.val_ratio, random_state=args.seed)

    dump_jsonl(train, out_dir / "train.jsonl")
    dump_jsonl(val, out_dir / "val.jsonl")

    print(f"Saved: {out_dir/'train.jsonl'} ({len(train)})")
    print(f"Saved: {out_dir/'val.jsonl'} ({len(val)})")


if __name__ == "__main__":
    main()
