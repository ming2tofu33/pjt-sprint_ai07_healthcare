from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_output_path(yolo_root: Path, raw_value: object, default_name: str) -> Path:
    raw = str(raw_value).strip() if raw_value is not None else ""
    path = Path(raw or default_name)
    if not path.is_absolute():
        path = yolo_root / path
    return path


def _build_image_index(train_img_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for img_path in sorted(train_img_dir.glob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in _VALID_IMAGE_EXTS:
            continue
        stem_key = img_path.stem.lower()
        rel = (Path("images") / "train" / img_path.name).as_posix()
        prev = index.get(stem_key)
        if prev is None:
            index[stem_key] = rel
            continue
        if rel < prev:
            logger.warning(
                "동일 stem 이미지 중복 감지(stem=%s): %s 대신 %s 선택",
                stem_key,
                prev,
                rel,
            )
            index[stem_key] = rel
        else:
            logger.warning(
                "동일 stem 이미지 중복 감지(stem=%s): %s 유지, %s 제외",
                stem_key,
                prev,
                rel,
            )
    return index


def _to_data_yaml_ref(path: Path, yolo_root: Path) -> str:
    try:
        return path.resolve().relative_to(yolo_root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def build_balanced_train_manifest(config: dict, yolo_root: Path, base_data_yaml: Path) -> dict | None:
    """Build a balanced train list and derivative data yaml for lightweight oversampling."""
    train_cfg = config.get("train", {})
    balanced_cfg_raw = train_cfg.get("balanced_sampling", {})
    balanced_cfg = balanced_cfg_raw if isinstance(balanced_cfg_raw, dict) else {}
    if not bool(balanced_cfg.get("enabled", False)):
        return None

    min_threshold = int(balanced_cfg.get("min_threshold", 100))
    target_count = int(balanced_cfg.get("target_count", 300))
    max_repeat = int(balanced_cfg.get("max_repeat_per_image", 4))
    max_repeat = max(0, max_repeat)

    train_lbl_dir = yolo_root / "labels" / "train"
    train_img_dir = yolo_root / "images" / "train"
    if not train_lbl_dir.exists() or not train_img_dir.exists():
        raise FileNotFoundError(
            f"train labels/images 경로가 존재하지 않습니다: labels={train_lbl_dir}, images={train_img_dir}"
        )
    if not base_data_yaml.exists():
        raise FileNotFoundError(f"base data.yaml이 존재하지 않습니다: {base_data_yaml}")

    image_index = _build_image_index(train_img_dir)
    label_files = sorted(train_lbl_dir.glob("*.txt"))
    if not label_files:
        return {"generated": False, "reason": "no_train_labels"}

    class_to_images: dict[int, set[str]] = {}
    image_classes: dict[str, set[int]] = {}
    base_lines: list[str] = []
    seen_images: set[str] = set()

    for label_path in label_files:
        rel_img = image_index.get(label_path.stem.lower())
        if rel_img is None:
            logger.warning("라벨에 대응하는 학습 이미지를 찾지 못해 제외: %s", label_path.name)
            continue

        if rel_img not in seen_images:
            seen_images.add(rel_img)
            base_lines.append(rel_img)

        classes_in_image = image_classes.setdefault(rel_img, set())
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            classes_in_image.add(cls_id)

    if not base_lines:
        return {"generated": False, "reason": "no_mapped_train_images"}

    for rel_img, class_ids in image_classes.items():
        for cls_id in class_ids:
            class_to_images.setdefault(cls_id, set()).add(rel_img)

    current_counts = {cls_id: len(images) for cls_id, images in class_to_images.items()}
    minor_classes = sorted(cls_id for cls_id, count in current_counts.items() if count < min_threshold)
    if not minor_classes:
        return {
            "generated": False,
            "reason": "no_target_classes",
            "base_lines": len(base_lines),
            "minor_classes": 0,
        }

    exposure_counts = Counter(current_counts)
    repeat_counts: Counter[str] = Counter()
    extra_lines: list[str] = []
    unmet_classes: list[int] = []

    for cls_id in minor_classes:
        candidates = sorted(class_to_images.get(cls_id, set()))
        if not candidates:
            unmet_classes.append(cls_id)
            logger.warning("소수 클래스 보강 후보 없음: class=%s", cls_id)
            continue

        cursor = 0
        while exposure_counts[cls_id] < target_count:
            selected: str | None = None
            for _ in range(len(candidates)):
                candidate = candidates[cursor % len(candidates)]
                cursor += 1
                if repeat_counts[candidate] < max_repeat:
                    selected = candidate
                    break

            if selected is None:
                unmet_classes.append(cls_id)
                logger.warning(
                    "반복 제한으로 목표치 미달: class=%s, current=%d, target=%d, max_repeat=%d",
                    cls_id,
                    exposure_counts[cls_id],
                    target_count,
                    max_repeat,
                )
                break

            extra_lines.append(selected)
            repeat_counts[selected] += 1
            for affected_cls in image_classes.get(selected, set()):
                exposure_counts[affected_cls] += 1

    output_train_list = _resolve_output_path(
        yolo_root,
        balanced_cfg.get("output_train_list"),
        "train_balanced.txt",
    )
    output_data_yaml = _resolve_output_path(
        yolo_root,
        balanced_cfg.get("output_data_yaml"),
        "data_balanced.yaml",
    )
    output_train_list.parent.mkdir(parents=True, exist_ok=True)
    output_data_yaml.parent.mkdir(parents=True, exist_ok=True)

    final_lines = base_lines + extra_lines
    with output_train_list.open("w", encoding="utf-8", newline="\n") as f:
        for line in final_lines:
            f.write(f"{line}\n")

    base_yaml = yaml.safe_load(base_data_yaml.read_text(encoding="utf-8"))
    if not isinstance(base_yaml, dict):
        raise ValueError(f"base data.yaml 형식이 올바르지 않습니다: {base_data_yaml}")
    base_yaml["train"] = _to_data_yaml_ref(output_train_list, yolo_root)
    with output_data_yaml.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(base_yaml, f, sort_keys=False, allow_unicode=True)

    return {
        "generated": True,
        "minor_classes": len(minor_classes),
        "minor_class_ids": minor_classes,
        "added_lines": len(extra_lines),
        "base_lines": len(base_lines),
        "final_train_lines": len(final_lines),
        "unmet_classes": unmet_classes,
        "output_train_list": str(output_train_list),
        "output_data_yaml": str(output_data_yaml),
    }

