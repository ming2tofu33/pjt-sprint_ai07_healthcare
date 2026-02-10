"""src.dataprep.output.export_yolo — df_clean.csv + splits.csv → YOLO 데이터셋 변환.

STAGE 1 에서 사용한다.

두 가지 호출 방식을 지원한다:
- CLI: ``python -m src.dataprep.output.export_yolo --df ... --splits ... --out ...``
- 프로그래밍: ``from src.dataprep.output.export_yolo import run_export``

주요 특성:
- hardlink 우선, copy fallback (Windows 호환)
- train + external 이미지 양쪽 검색
- split conflict 감지
- bbox clamping 통계
- critical missing threshold
- class_map.csv 별도 출력
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 설치 여부는 환경 의존
    tqdm = None


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]  # src/dataprep/output/ → repo root


def _resolve_repo_path(path_str: str, repo_root: Path | None = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    root = repo_root or REPO_ROOT
    return (root / p).resolve()


def _repo_relative_str(path: Path, repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    try:
        return path.resolve().relative_to(root).as_posix()
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


def _write_data_yaml(
    out_dir: Path,
    class_ids_sorted: list[int],
    repo_root: Path | None = None,
) -> None:
    data_yaml = out_dir / "data.yaml"
    rel_out = _repo_relative_str(out_dir, repo_root)
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


def _iter_with_progress(
    iterable: Iterable[Any],
    *,
    total: int,
    desc: str,
    enabled: bool,
) -> Iterable[Any]:
    """환경에 따라 진행률을 표시하며 iterable을 순회한다.

    규칙:
    - enabled=False: 원본 iterable 그대로 순회
    - tqdm 설치 + TTY: tqdm progress bar 사용
    - 그 외: 주기적 heartbeat 로그 출력
    """
    if not enabled:
        yield from iterable
        return

    stderr = sys.stderr
    use_tqdm = bool(tqdm is not None and stderr is not None and hasattr(stderr, "isatty") and stderr.isatty())
    if use_tqdm:
        yield from tqdm(iterable, total=total, desc=desc, mininterval=0.5, dynamic_ncols=True)
        return

    count_step = max(500, total // 20) if total > 0 else 500
    time_step = 10.0

    last_logged_count = 0
    last_logged_time = perf_counter()
    for idx, item in enumerate(iterable, start=1):
        now = perf_counter()
        should_log = (
            idx == 1
            or idx == total
            or (idx - last_logged_count) >= count_step
            or (now - last_logged_time) >= time_step
        )
        if should_log:
            percent = (idx / total * 100.0) if total > 0 else 100.0
            print(f"[INFO] {desc}: {idx}/{total} ({percent:.1f}%)", flush=True)
            last_logged_count = idx
            last_logged_time = now
        yield item


# ─────────────────────────────────────────────
#  프로그래밍 방식 호출: run_export()
# ─────────────────────────────────────────────

def run_export(
    df_path: Path,
    splits_path: Path,
    out_dir: Path,
    train_images_dir: Path,
    external_images_dir: Path | None = None,
    class_map_path: Path | None = None,
    link_mode: str = "hardlink",
    allow_fallback: bool = True,
    critical_missing_ratio: float = 0.02,
    critical_missing_count: int = 20,
    repo_root: Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """df_clean.csv + splits.csv → YOLO 데이터셋 생성.

    Parameters
    ----------
    df_path : Path
        df_clean.csv 경로.
    splits_path : Path
        splits.csv 경로 (group_id, split 컬럼).
    out_dir : Path
        YOLO 데이터셋 출력 디렉터리.
    train_images_dir : Path
        train 이미지 루트 디렉터리.
    external_images_dir : Path | None
        external 이미지 루트 디렉터리 (없으면 빈 인덱스).
    class_map_path : Path | None
        class_map.csv 출력 경로 (None이면 생략).
    link_mode : str
        ``"hardlink"`` 또는 ``"copy"``.
    allow_fallback : bool
        hardlink 실패 시 copy fallback 허용 여부.
    critical_missing_ratio : float
        누락 이미지 비율 임계값.
    critical_missing_count : int
        누락 이미지 최소 건수 임계값.
    repo_root : Path | None
        repo root (data.yaml 상대경로 기준). None이면 자동 결정.
    progress : bool
        진행률 표시 활성화 여부. tqdm 미설치 환경에서는 heartbeat 로그로 대체된다.

    Returns
    -------
    dict
        변환 통계. 키: ``image_counts``, ``label_counts``, ``nc``,
        ``missing_image_count``, ``critical``, ``data_yaml`` 등.
    """
    root = repo_root or REPO_ROOT

    # ── 입력 검증 ──
    if not df_path.exists():
        raise FileNotFoundError(f"df 파일을 찾을 수 없습니다: {df_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits 파일을 찾을 수 없습니다: {splits_path}")

    required_df_cols = {
        "file_name", "width", "height", "group_id",
        "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "source",
    }
    required_split_cols = {"group_id", "split"}

    df = pd.read_csv(df_path)
    missing_df_cols = sorted(required_df_cols - set(df.columns))
    if missing_df_cols:
        raise ValueError(f"df에 필수 컬럼이 없습니다: {missing_df_cols}")

    splits_df = pd.read_csv(splits_path)
    missing_split_cols = sorted(required_split_cols - set(splits_df.columns))
    if missing_split_cols:
        raise ValueError(f"splits에 필수 컬럼이 없습니다: {missing_split_cols}")

    # ── class map 구축 ──
    class_ids_sorted = sorted(int(x) for x in df["category_id"].dropna().astype(int).unique())
    class_map = {cid: i for i, cid in enumerate(class_ids_sorted)}

    if class_map_path is not None:
        class_map_path.parent.mkdir(parents=True, exist_ok=True)
        with class_map_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_index", "category_id"])
            writer.writeheader()
            for cid, idx in class_map.items():
                writer.writerow({"class_index": idx, "category_id": cid})

    # ── split map 로드 ──
    split_map, split_conflicts = _load_split_map(splits_df)

    # ── 이미지 인덱스 구축 ──
    train_index, train_dup_names = _build_image_index(train_images_dir)
    if external_images_dir is not None and external_images_dir.exists():
        external_index, external_dup_names = _build_image_index(external_images_dir)
    else:
        external_index: dict[str, Path] = {}
        external_dup_names = 0

    # ── 출력 디렉터리 생성 ──
    images_train_dir = out_dir / "images" / "train"
    images_val_dir = out_dir / "images" / "val"
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"
    for p in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ── 메인 루프 ──
    grouped = df.groupby("file_name", sort=False)
    total_images = int(df["file_name"].nunique())

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

    for file_name, g in _iter_with_progress(
        grouped,
        total=total_images,
        desc="Converting images",
        enabled=progress,
    ):
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
                        "expected_external_root": str(external_images_dir or ""),
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
            link_mode=link_mode,
            allow_fallback=allow_fallback,
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

    # ── data.yaml 생성 ──
    _write_data_yaml(out_dir, class_ids_sorted, repo_root=root)

    # ── 통계 집계 ──
    all_min_cls = min(classes_seen) if classes_seen else None
    all_max_cls = max(classes_seen) if classes_seen else None
    nc = len(class_ids_sorted)
    missing_image_count = total_images - (image_counts["train"] + image_counts["val"])
    missing_critical_threshold = max(
        int(total_images * critical_missing_ratio),
        critical_missing_count,
    )
    critical = missing_image_count > missing_critical_threshold

    # class_index 범위 검증
    class_range_error = False
    if all_min_cls is not None and (all_min_cls < 0 or all_max_cls >= nc):
        class_range_error = True

    # ── convert_manifest.json 저장 ──
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "link_mode": link_mode,
        "source_df": str(df_path),
        "source_splits": str(splits_path),
        "output_dir": str(out_dir),
        "train_images": image_counts["train"],
        "val_images": image_counts["val"],
        "nc": nc,
        "missing_image_count": missing_image_count,
        "invalid_bbox_rows": invalid_bbox_rows,
        "clamped_rows": clamped_rows,
        "empty_label_files": empty_label_files,
    }
    manifest_path = out_dir / "convert_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "image_counts": image_counts,
        "label_counts": label_counts,
        "nc": nc,
        "class_ids_sorted": class_ids_sorted,
        "class_map": class_map,
        "all_min_cls": all_min_cls,
        "all_max_cls": all_max_cls,
        "missing_image_count": missing_image_count,
        "missing_critical_threshold": missing_critical_threshold,
        "critical": critical,
        "class_range_error": class_range_error,
        "split_conflicts": split_conflicts,
        "missing_images": missing_images,
        "missing_groups": missing_groups,
        "source_fallback_count": source_fallback_count,
        "invalid_bbox_rows": invalid_bbox_rows,
        "clamped_rows": clamped_rows,
        "empty_label_files": empty_label_files,
        "hardlink_count": hardlink_count,
        "copy_count": copy_count,
        "train_index_size": len(train_index),
        "external_index_size": len(external_index),
        "train_dup_names": train_dup_names,
        "external_dup_names": external_dup_names,
        "data_yaml": out_dir / "data.yaml",
        "out_dir": out_dir,
    }


# ─────────────────────────────────────────────
#  Label 검증
# ─────────────────────────────────────────────

def verify_labels(output_dir: Path, nc: int, progress: bool = False) -> dict:
    """생성된 YOLO label 파일을 재파싱하여 sanity check 한다.

    Parameters
    ----------
    output_dir : Path
        YOLO 데이터셋 루트 (labels/train, labels/val 포함).
    nc : int
        클래스 수 (class_index 범위 검증용).
    progress : bool
        진행률 표시 활성화 여부.

    Returns
    -------
    dict
        ``{total_files, total_lines, errors}``
    """
    total_files = 0
    total_lines = 0
    errors: list[str] = []

    label_files: list[Path] = []
    for split in ("train", "val"):
        label_dir = output_dir / "labels" / split
        if not label_dir.exists():
            continue
        label_files.extend(sorted(label_dir.glob("*.txt")))

    for txt_file in _iter_with_progress(
        label_files,
        total=len(label_files),
        desc="Verifying labels",
        enabled=progress,
    ):
        total_files += 1
        with txt_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{txt_file.name}:{line_no} 컬럼 개수 오류 ({len(parts)})")
                    continue
                try:
                    cls_idx = int(parts[0])
                    vals = [float(p) for p in parts[1:]]
                except ValueError:
                    errors.append(f"{txt_file.name}:{line_no} 숫자가 아닌 값 포함")
                    continue
                if cls_idx < 0 or cls_idx >= nc:
                    errors.append(
                        f"{txt_file.name}:{line_no} class {cls_idx} 범위 초과 [0, {nc})"
                    )
                for v in vals:
                    if v < 0.0 or v > 1.0:
                        errors.append(f"{txt_file.name}:{line_no} 값 {v} 범위 초과 [0,1]")
                        break

    return {"total_files": total_files, "total_lines": total_lines, "errors": errors}


# ─────────────────────────────────────────────
#  CLI 진입점
# ─────────────────────────────────────────────

def main(argv: list[str]) -> int:
    """CLI 진입점: argparse → run_export() 호출."""
    # 상대경로 해석 일관성을 위해 작업 디렉터리를 repo root로 고정한다.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="df_clean.csv + splits.csv에서 YOLO 데이터셋을 생성합니다 (hardlink 우선, repo 상대경로)."
    )
    parser.add_argument("--df", default="data/processed/df_clean.csv")
    parser.add_argument("--splits", default="data/metadata/splits.csv")
    parser.add_argument("--out", default="data/processed/yolo")
    parser.add_argument("--train-images", default="data/raw/train_images")
    parser.add_argument("--external-images", default="data/raw/external/combined/images")
    parser.add_argument("--class-map-out", default="data/metadata/class_map.csv")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--no-fallback", action="store_true", help="hardlink 실패 시 copy fallback을 비활성화합니다.")
    parser.add_argument("--critical-missing-ratio", type=float, default=0.02)
    parser.add_argument("--critical-missing-count", type=int, default=20)
    args = parser.parse_args(argv)

    df_path = _resolve_repo_path(args.df)
    splits_path = _resolve_repo_path(args.splits)
    out_dir = _resolve_repo_path(args.out)
    train_images_dir = _resolve_repo_path(args.train_images)
    external_images_dir = _resolve_repo_path(args.external_images)
    class_map_path = _resolve_repo_path(args.class_map_out)

    try:
        result = run_export(
            df_path=df_path,
            splits_path=splits_path,
            out_dir=out_dir,
            train_images_dir=train_images_dir,
            external_images_dir=external_images_dir,
            class_map_path=class_map_path,
            link_mode=args.link_mode,
            allow_fallback=not args.no_fallback,
            critical_missing_ratio=args.critical_missing_ratio,
            critical_missing_count=args.critical_missing_count,
            repo_root=REPO_ROOT,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2

    # ── summary 출력 ──
    image_counts = result["image_counts"]
    label_counts = result["label_counts"]
    nc = result["nc"]

    print("[요약] YOLO export 완료")
    print(f"  이미지 train/val: {image_counts['train']} / {image_counts['val']}")
    print(f"  라벨 train/val: {label_counts['train']} / {label_counts['val']}")
    print(f"  nc: {nc}")
    print(f"  라벨 내 class_index 최소/최대: {result['all_min_cls']} / {result['all_max_cls']}")
    print(
        f"  누락 이미지: {result['missing_image_count']} "
        f"(임계값>{result['missing_critical_threshold']} 이면 치명)"
    )
    if result["missing_images"]:
        print("  누락 이미지 예시:")
        for ex in result["missing_images"][:5]:
            print(f"    - {ex['file_name']}")
    if result["split_conflicts"]:
        print(f"  split 충돌 감지: {result['split_conflicts']} (첫 번째 할당 유지)")
    if result["missing_groups"]:
        print(f"  group_id/split 누락으로 train에 할당된 건수: {len(result['missing_groups'])}")
    print(f"  invalid bbox로 스킵된 행: {result['invalid_bbox_rows']}")
    print(f"  clamped bbox 행: {result['clamped_rows']}")
    print(f"  빈 라벨 파일 수: {result['empty_label_files']}")
    print(f"  링크 통계 hardlink/copy: {result['hardlink_count']} / {result['copy_count']}")
    if class_map_path:
        print(f"  클래스 맵: {_repo_relative_str(class_map_path)}")
    print(f"  data.yaml: {_repo_relative_str(result['data_yaml'])}")
    print(f"  이미지 인덱스(train/external): {result['train_index_size']} / {result['external_index_size']}")
    if result["train_dup_names"] or result["external_dup_names"]:
        print(
            "  중복 이미지 파일명으로 무시된 건수(train/external): "
            f"{result['train_dup_names']} / {result['external_dup_names']}"
        )

    if result["class_range_error"]:
        print("[ERR] 라벨 내 class_index 범위 오류", file=sys.stderr)
        return 2

    if result["critical"]:
        print("[ERR] 누락 이미지가 너무 많아 export를 치명적 실패로 처리합니다", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
