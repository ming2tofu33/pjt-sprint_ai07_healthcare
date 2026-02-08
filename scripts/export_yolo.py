from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _repo_relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _build_image_index(root: Path) -> tuple[dict[str, Path], int]:
    index: dict[str, Path] = {}
    duplicate_name_count = 0
    for p in _iter_image_files(root):
        key = p.name
        if key in index:
            duplicate_name_count += 1
            continue
        index[key] = p
    return index, duplicate_name_count


def _link_or_copy(src: Path, dst: Path, *, link_mode: str, allow_fallback: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    if link_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if not allow_fallback:
            raise
        shutil.copy2(src, dst)
        return "copy_fallback"


def _load_split_map(splits_df: pd.DataFrame) -> tuple[dict[str, str], int]:
    split_map: dict[str, str] = {}
    conflict_count = 0
    for row in splits_df.itertuples(index=False):
        group_id = str(getattr(row, "group_id", "")).strip()
        split = str(getattr(row, "split", "")).strip().lower()
        if group_id == "":
            continue
        if split not in {"train", "val"}:
            continue
        prev = split_map.get(group_id)
        if prev is None:
            split_map[group_id] = split
        elif prev != split:
            conflict_count += 1
    return split_map, conflict_count


def _safe_float(value: object) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _write_data_yaml(out_dir: Path, class_ids_sorted: list[int]) -> None:
    data_yaml = out_dir / "data.yaml"
    rel_out = _repo_relative_str(out_dir)
    lines = [
        f"path: {rel_out}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_ids_sorted)}",
        "names:",
    ]
    for i, cid in enumerate(class_ids_sorted):
        lines.append(f'  {i}: "{cid}"')
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    # Force repo root as cwd so relative paths behave consistently across IDE/terminal.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="Export YOLO dataset from df_clean.csv + splits.csv (hardlink-first, repo-relative)."
    )
    parser.add_argument("--df", default="data/processed/df_clean.csv")
    parser.add_argument("--splits", default="data/metadata/splits.csv")
    parser.add_argument("--out", default="data/processed/yolo")
    parser.add_argument("--train-images", default="data/raw/train_images")
    parser.add_argument("--external-images", default="data/raw/external/combined/images")
    parser.add_argument("--class-map-out", default="data/metadata/class_map.csv")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--no-fallback", action="store_true", help="Disable hardlink->copy fallback.")
    parser.add_argument("--critical-missing-ratio", type=float, default=0.02)
    parser.add_argument("--critical-missing-count", type=int, default=20)
    args = parser.parse_args(argv)

    df_path = _resolve_repo_path(args.df)
    splits_path = _resolve_repo_path(args.splits)
    out_dir = _resolve_repo_path(args.out)
    train_images_dir = _resolve_repo_path(args.train_images)
    external_images_dir = _resolve_repo_path(args.external_images)
    class_map_path = _resolve_repo_path(args.class_map_out)

    if not df_path.exists():
        print(f"[ERR] df not found: {df_path}", file=sys.stderr)
        return 2
    if not splits_path.exists():
        print(f"[ERR] splits not found: {splits_path}", file=sys.stderr)
        return 2

    required_df_cols = {
        "file_name",
        "width",
        "height",
        "group_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "source",
    }
    required_split_cols = {"group_id", "split"}

    df = pd.read_csv(df_path)
    missing_df_cols = sorted(required_df_cols - set(df.columns))
    if missing_df_cols:
        print(f"[ERR] missing df columns: {missing_df_cols}", file=sys.stderr)
        return 2

    splits_df = pd.read_csv(splits_path)
    missing_split_cols = sorted(required_split_cols - set(splits_df.columns))
    if missing_split_cols:
        print(f"[ERR] missing splits columns: {missing_split_cols}", file=sys.stderr)
        return 2

    class_ids_sorted = sorted(int(x) for x in df["category_id"].dropna().astype(int).unique())
    class_map = {cid: i for i, cid in enumerate(class_ids_sorted)}

    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    with class_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_index", "category_id"])
        writer.writeheader()
        for cid, idx in class_map.items():
            writer.writerow({"class_index": idx, "category_id": cid})

    split_map, split_conflicts = _load_split_map(splits_df)

    train_index, train_dup_names = _build_image_index(train_images_dir)
    external_index, external_dup_names = _build_image_index(external_images_dir)

    images_train_dir = out_dir / "images" / "train"
    images_val_dir = out_dir / "images" / "val"
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"
    for p in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("file_name", sort=False)

    image_counts = {"train": 0, "val": 0}
    label_counts = {"train": 0, "val": 0}
    hardlink_count = 0
    copy_count = 0
    missing_images: list[dict[str, str]] = []
    missing_groups: list[dict[str, str]] = []
    source_fallback_count = 0
    invalid_bbox_rows = 0
    clamped_rows = 0
    classes_seen: set[int] = set()
    empty_label_files = 0

    for file_name, g in grouped:
        g_sources = set(g["source"].astype(str).str.lower().tolist())
        preferred = "train" if "train" in g_sources else "external"
        train_candidate = train_index.get(str(file_name))
        ext_candidate = external_index.get(str(file_name))

        source_path = None
        if preferred == "train":
            if train_candidate is not None:
                source_path = train_candidate
            elif ext_candidate is not None:
                source_path = ext_candidate
                source_fallback_count += 1
        else:
            if ext_candidate is not None:
                source_path = ext_candidate
            elif train_candidate is not None:
                source_path = train_candidate
                source_fallback_count += 1

        if source_path is None:
            if len(missing_images) < 20:
                missing_images.append(
                    {
                        "file_name": str(file_name),
                        "expected_train_root": str(train_images_dir),
                        "expected_external_root": str(external_images_dir),
                    }
                )
            continue

        group_ids = [str(x).strip() for x in g["group_id"].dropna().tolist() if str(x).strip()]
        group_id = group_ids[0] if group_ids else ""
        split = split_map.get(group_id, "train")
        if group_id == "" or group_id not in split_map:
            if len(missing_groups) < 20:
                missing_groups.append({"file_name": str(file_name), "group_id": group_id, "assigned_split": split})

        img_out_dir = images_train_dir if split == "train" else images_val_dir
        lbl_out_dir = labels_train_dir if split == "train" else labels_val_dir
        out_img_path = img_out_dir / str(file_name)
        mode = _link_or_copy(
            source_path,
            out_img_path,
            link_mode=args.link_mode,
            allow_fallback=not args.no_fallback,
        )
        if mode == "hardlink":
            hardlink_count += 1
        elif mode in {"copy", "copy_fallback"}:
            copy_count += 1

        label_lines: list[str] = []
        for row in g.itertuples(index=False):
            cid_raw = getattr(row, "category_id", None)
            try:
                cid = int(cid_raw)
            except Exception:
                invalid_bbox_rows += 1
                continue
            cls_idx = class_map.get(cid)
            if cls_idx is None:
                invalid_bbox_rows += 1
                continue

            width = _safe_float(getattr(row, "width", None))
            height = _safe_float(getattr(row, "height", None))
            bx = _safe_float(getattr(row, "bbox_x", None))
            by = _safe_float(getattr(row, "bbox_y", None))
            bw = _safe_float(getattr(row, "bbox_w", None))
            bh = _safe_float(getattr(row, "bbox_h", None))

            if None in (width, height, bx, by, bw, bh):
                invalid_bbox_rows += 1
                continue
            if width <= 0 or height <= 0 or bw <= 0 or bh <= 0:
                invalid_bbox_rows += 1
                continue

            xc = (bx + (bw / 2.0)) / width
            yc = (by + (bh / 2.0)) / height
            wn = bw / width
            hn = bh / height
            if not all(math.isfinite(v) for v in (xc, yc, wn, hn)):
                invalid_bbox_rows += 1
                continue

            xc_c = _clamp01(xc)
            yc_c = _clamp01(yc)
            wn_c = _clamp01(wn)
            hn_c = _clamp01(hn)
            if (xc_c, yc_c, wn_c, hn_c) != (xc, yc, wn, hn):
                clamped_rows += 1
            if wn_c <= 0.0 or hn_c <= 0.0:
                invalid_bbox_rows += 1
                continue

            label_lines.append(f"{cls_idx} {xc_c:.6f} {yc_c:.6f} {wn_c:.6f} {hn_c:.6f}")
            classes_seen.add(cls_idx)

        label_path = lbl_out_dir / (Path(str(file_name)).stem + ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8", newline="\n") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")
            else:
                empty_label_files += 1

        image_counts[split] += 1
        label_counts[split] += 1

    _write_data_yaml(out_dir, class_ids_sorted)

    all_min_cls = min(classes_seen) if classes_seen else None
    all_max_cls = max(classes_seen) if classes_seen else None
    nc = len(class_ids_sorted)
    total_images = int(df["file_name"].nunique())
    missing_image_count = total_images - (image_counts["train"] + image_counts["val"])
    missing_critical_threshold = max(
        int(total_images * float(args.critical_missing_ratio)),
        int(args.critical_missing_count),
    )
    critical = missing_image_count > missing_critical_threshold

    print("[SUMMARY] YOLO export done")
    print(f"  images train/val: {image_counts['train']} / {image_counts['val']}")
    print(f"  labels train/val: {label_counts['train']} / {label_counts['val']}")
    print(f"  nc: {nc}")
    print(f"  class_index min/max in labels: {all_min_cls} / {all_max_cls}")
    print(f"  missing images: {missing_image_count} (threshold>{missing_critical_threshold} => critical)")
    if missing_images:
        print("  missing image examples:")
        for ex in missing_images[:5]:
            print(f"    - {ex['file_name']}")
    if split_conflicts:
        print(f"  split conflicts detected: {split_conflicts} (kept first assignment)")
    if missing_groups:
        print(f"  missing group_id/split mapped to train: {len(missing_groups)}")
    print(f"  invalid bbox rows skipped: {invalid_bbox_rows}")
    print(f"  clamped bbox rows: {clamped_rows}")
    print(f"  empty label files: {empty_label_files}")
    print(f"  link stats hardlink/copy: {hardlink_count} / {copy_count}")
    print(f"  class map: {_repo_relative_str(class_map_path)}")
    print(f"  data yaml: {_repo_relative_str(out_dir / 'data.yaml')}")
    print(f"  image index (train/external): {len(train_index)} / {len(external_index)}")
    if train_dup_names or external_dup_names:
        print(f"  duplicate image names ignored (train/external): {train_dup_names} / {external_dup_names}")

    if all_min_cls is not None and (all_min_cls < 0 or all_max_cls >= nc):
        print("[ERR] class_index out of range in labels", file=sys.stderr)
        return 2

    if critical:
        print("[ERR] too many missing images; export marked as critical failure", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
