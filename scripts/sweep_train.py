#!/usr/bin/env python3
"""
Run a simple grid sweep for YOLO training and summarize mAP50-95.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

from train_yolo import repo_root, run_training, to_repo_display


def parse_int_list(raw: str, flag_name: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{flag_name} must be comma-separated integers: {raw}") from exc
        if value <= 0:
            raise ValueError(f"{flag_name} values must be > 0: {raw}")
        values.append(value)

    if not values:
        raise ValueError(f"{flag_name} is empty: {raw}")

    return values


def make_unique_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    idx = 2
    while True:
        candidate = f"{base_name}_r{idx}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def extract_last_map5095(results_csv: Path) -> float:
    if not results_csv.exists():
        raise FileNotFoundError(f"[ERR] results.csv not found: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"[ERR] results.csv is empty: {results_csv}")

    last = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in rows[-1].items()}

    for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
        value = last.get(key)
        if value not in (None, ""):
            return float(value)

    raise KeyError(f"[ERR] mAP50-95 column not found in results.csv: {results_csv}")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid sweep runner for YOLO training")
    parser.add_argument("--epochs-list", type=str, default="3,10,30")
    parser.add_argument("--batch-list", type=str, default="8,16")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--data", type=str, default="data/processed/yolo/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    try:
        epochs_list = parse_int_list(args.epochs_list, "--epochs-list")
        batch_list = parse_int_list(args.batch_list, "--batch-list")
    except ValueError as exc:
        print(f"[ERR] {exc}")
        sys.exit(1)

    combinations = list(itertools.product(epochs_list, batch_list))
    print(f"[INFO] Sweep runs: {len(combinations)}")

    rows: list[list[str]] = []
    used_names: set[str] = set()
    failed = 0

    for idx, (epochs, batch) in enumerate(combinations, start=1):
        base_name = f"e{epochs}_b{batch}_img{args.imgsz}"
        run_name = make_unique_name(base_name, used_names)

        print(f"[INFO] ({idx}/{len(combinations)}) start: {run_name}")
        try:
            train_result = run_training(
                data=args.data,
                model=args.model,
                epochs=epochs,
                batch=batch,
                imgsz=args.imgsz,
                device=args.device,
                name=run_name,
                project=args.project,
                seed=args.seed,
                workers=args.workers,
                patience=args.patience,
            )
            map5095 = extract_last_map5095(Path(train_result["results_csv"]))
            map_text = f"{map5095:.6f}"
            result_path = to_repo_display(Path(train_result["results_csv"]), root)
            print(f"[OK] {run_name} mAP50-95(last): {map_text}")
        except Exception as exc:
            failed += 1
            map_text = "N/A"
            result_path = "-"
            print(exc)

        rows.append(
            [
                run_name,
                str(epochs),
                str(batch),
                str(args.imgsz),
                map_text,
                result_path,
            ]
        )

    print("\n[INFO] Sweep summary (last epoch)")
    print_table(
        headers=["name", "epochs", "batch", "imgsz", "mAP50-95", "results.csv"],
        rows=rows,
    )

    if failed > 0:
        print(f"\n[WARN] Failed runs: {failed}/{len(combinations)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
