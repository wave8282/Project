"""CSV pose dataset diagnostic utility.

Usage:
    python src/preprocessing/diagnose_csv.py fall_data.csv pose_data.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


def inspect_csv(path: Path) -> dict:
    digest = hashlib.sha256()
    rows = 0
    widths: dict[int, int] = {}
    invalid_numbers = 0
    non_finite = 0
    out_of_range = 0
    duplicate_rows = 0
    seen: set[tuple[str, ...]] = set()
    minimum = math.inf
    maximum = -math.inf

    with path.open("rb") as raw_file:
        for chunk in iter(lambda: raw_file.read(1024 * 1024), b""):
            digest.update(chunk)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            rows += 1
            widths[len(row)] = widths.get(len(row), 0) + 1
            key = tuple(row)
            if key in seen:
                duplicate_rows += 1
            else:
                seen.add(key)

            for value in row:
                try:
                    number = float(value)
                except ValueError:
                    invalid_numbers += 1
                    continue
                if not math.isfinite(number):
                    non_finite += 1
                    continue
                minimum = min(minimum, number)
                maximum = max(maximum, number)
                if number < 0 or number > 1:
                    out_of_range += 1

    return {
        "file": str(path),
        "rows": rows,
        "column_widths": widths,
        "invalid_numbers": invalid_numbers,
        "non_finite_values": non_finite,
        "out_of_range_0_1": out_of_range,
        "duplicate_rows": duplicate_rows,
        "min": None if minimum is math.inf else minimum,
        "max": None if maximum is -math.inf else maximum,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MediaPipe pose CSV files.")
    parser.add_argument("files", nargs="+", type=Path, help="CSV files to inspect")
    args = parser.parse_args()

    results = []
    for path in args.files:
        if not path.is_file():
            print(f"\n[NOT FOUND] {path}")
            continue
        result = inspect_csv(path)
        results.append(result)
        print(f"\n=== {path} ===")
        for key, value in result.items():
            if key != "file":
                print(f"{key}: {value}")

    if len(results) >= 2:
        print("\n=== FILE COMPARISON ===")
        for index, left in enumerate(results):
            for right in results[index + 1:]:
                same_hash = left["sha256"] == right["sha256"]
                print(
                    f"{left['file']} vs {right['file']}: "
                    f"identical_content={same_hash}"
                )


if __name__ == "__main__":
    main()
