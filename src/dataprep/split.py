from __future__ import annotations

import csv
import random
import re
from pathlib import Path


def _extract_group_id(file_name: str, regex: str, fallback: str) -> str:
    m = re.search(regex, file_name)
    if m:
        return m.group(0)
    if fallback == "prefix_before_underscore":
        return file_name.split("_")[0]
    return file_name


def add_group_id(records: list[dict], split_cfg: dict) -> None:
    regex = split_cfg.get("group_id", {}).get("regex", r"K-\d+-\d+-\d+")
    fallback = split_cfg.get("group_id", {}).get("fallback", "prefix_before_underscore")
    for r in records:
        r["group_id"] = _extract_group_id(r["file_name"], regex, fallback)


def make_group_split(records: list[dict], split_cfg: dict, seed: int) -> list[dict]:
    if not split_cfg.get("enabled", True):
        return []
    groups = sorted({r["group_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    n_train = int(len(groups) * train_ratio)
    train_groups = set(groups[:n_train])
    rows = []
    for g in groups:
        rows.append({"group_id": g, "split": "train" if g in train_groups else "val"})
    return rows


def write_splits(metadata_dir: Path, splits_name: str, splits: list[dict]) -> None:
    with (metadata_dir / splits_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "split"])
        writer.writeheader()
        writer.writerows(splits)
