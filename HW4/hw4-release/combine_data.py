#!/usr/bin/env python3
"""Merge multiple JSONL datasets (email/safety classifier format: text + label per line)."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from pathlib import Path


def parse_record(raw: str, source: Path, line_no: int) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source}:{line_no}: invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{source}:{line_no}: expected JSON object, got {type(obj).__name__}")
    if "text" not in obj or "label" not in obj:
        raise ValueError(f"{source}:{line_no}: missing 'text' or 'label' keys")
    return {"text": obj["text"], "label": obj["label"]}


def _not_found_with_hint(path: Path) -> FileNotFoundError:
    parent = path.parent
    if parent.is_dir():
        names = [p.name for p in parent.iterdir() if p.is_file()]
        close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.5)
        if close:
            return FileNotFoundError(
                f"not found: {path}\n  (did you mean in {parent!s}: {', '.join(close)}?)"
            )
    return FileNotFoundError(f"not found: {path}")


def collect_records(inputs: list[Path], dedupe_on_text: bool) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for path in inputs:
        if not path.exists():
            raise _not_found_with_hint(path)
        if path.is_dir():
            raise IsADirectoryError(f"expected a file, got directory: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = parse_record(line, path, line_no)
                if dedupe_on_text:
                    t = rec["text"]
                    key = t if isinstance(t, str) else str(t)
                    if key in seen:
                        continue
                    seen.add(key)
                out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine JSONL datasets (one JSON object per line: text, label)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .jsonl files to merge (order preserved unless --shuffle).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .jsonl path.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate examples by exact string equality of 'text'.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle merged rows before writing (uses --seed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when --shuffle is set (default: 42).",
    )
    args = parser.parse_args()

    records = collect_records(args.inputs, dedupe_on_text=args.dedupe)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} lines -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
