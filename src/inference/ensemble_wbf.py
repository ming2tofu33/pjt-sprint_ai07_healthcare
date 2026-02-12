"""WBF-based multi-model inference helpers for STAGE 4 submission."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ensemble_boxes import weighted_boxes_fusion as _weighted_boxes_fusion
except ImportError:  # pragma: no cover - environment-dependent
    _weighted_boxes_fusion = None

from src.inference.predictor import batch_predict
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_WEIGHT_TAGS = {"competition_best", "best", "last"}
_ALLOWED_CONF_TYPES = {"avg", "max", "box_and_model_avg", "absent_model_aware_avg"}


@dataclass(frozen=True)
class ModelSpec:
    run_name: str
    weight_tag: str
    conf: float
    nms_iou: float
    imgsz: int
    augment: bool
    model_weight: float
    weight_path: Path


def resolve_ensemble_model_specs(
    config: dict[str, Any],
    cli_args: Any,
    repo_root: Path,
) -> list[ModelSpec]:
    """Resolve model specs for ensemble inference."""
    sub_cfg = config.get("submission", {})
    train_cfg = config.get("train", {})
    ensemble_cfg = sub_cfg.get("ensemble", {})
    if not isinstance(ensemble_cfg, dict):
        ensemble_cfg = {}

    method = str(ensemble_cfg.get("method", "wbf")).strip().lower()
    if method != "wbf":
        raise ValueError(f"Unsupported ensemble method: {method} (only 'wbf' is supported)")

    paths_cfg = config.get("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError("Invalid config: paths section is missing or not a mapping")

    default_conf = float(sub_cfg.get("conf", 0.25))
    default_nms_iou = float(sub_cfg.get("nms_iou", 0.5))
    default_imgsz = int(train_cfg.get("imgsz", 640))
    default_augment = bool(sub_cfg.get("augment", False))

    cfg_specs = _parse_specs_from_config(
        ensemble_cfg.get("runs", []),
        paths_cfg=paths_cfg,
        repo_root=repo_root,
        default_conf=default_conf,
        default_nms_iou=default_nms_iou,
        default_imgsz=default_imgsz,
        default_augment=default_augment,
    )
    cfg_specs_by_run = {spec.run_name: spec for spec in cfg_specs}

    cli_runs = _parse_cli_ensemble_runs(getattr(cli_args, "ensemble_runs", ""))
    if cli_runs:
        resolved: list[ModelSpec] = []
        for run_name in cli_runs:
            if run_name in cfg_specs_by_run:
                resolved.append(cfg_specs_by_run[run_name])
                continue

            resolved.append(
                _build_model_spec(
                    run_name=run_name,
                    weight_tag="competition_best",
                    conf=default_conf,
                    nms_iou=default_nms_iou,
                    imgsz=default_imgsz,
                    augment=default_augment,
                    model_weight=1.0,
                    paths_cfg=paths_cfg,
                    repo_root=repo_root,
                )
            )
        specs = resolved
    else:
        specs = cfg_specs

    if len(specs) < 2:
        raise ValueError("submission.ensemble.runs must contain at least 2 models")

    return specs


def load_idx2id_for_run(
    run_name: str,
    paths_cfg: dict[str, Any],
    repo_root: Path,
) -> dict[int, int]:
    """Load class index -> category_id mapping from label_map_full.json."""
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()

    label_map_path = processed_base / run_name / "label_map_full.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"label_map_full.json not found for run '{run_name}': {label_map_path}"
        )

    with label_map_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    idx2id_raw = payload.get("idx2id", {})
    if not isinstance(idx2id_raw, dict) or not idx2id_raw:
        raise ValueError(f"Invalid idx2id in label map: {label_map_path}")

    return {int(k): int(v) for k, v in idx2id_raw.items()}


def validate_category_map_compatibility(idx2id_list: list[dict[int, int]]) -> set[int]:
    """Validate category_id set compatibility across models (strict mode)."""
    if not idx2id_list:
        raise ValueError("No idx2id mappings provided for validation")

    category_sets = [set(mapping.values()) for mapping in idx2id_list]
    base = category_sets[0]

    for idx, current in enumerate(category_sets[1:], start=1):
        if current != base:
            missing = sorted(base - current)
            extra = sorted(current - base)
            raise ValueError(
                "category_id mapping mismatch across ensemble models. "
                f"model[0]-only={missing[:20]} model[{idx}]-only={extra[:20]}"
            )

    return set(base)


def predict_for_spec(
    spec: ModelSpec,
    test_images_dir: Path,
    device: Any,
    verbose: bool,
) -> list[dict]:
    """Run batch prediction for one model spec."""
    logger.info(
        "ensemble model predict | run=%s | weight=%s | conf=%.4f | iou=%.3f | imgsz=%d | augment=%s",
        spec.run_name,
        spec.weight_path,
        spec.conf,
        spec.nms_iou,
        spec.imgsz,
        spec.augment,
    )
    return batch_predict(
        weights_path=spec.weight_path,
        source=test_images_dir,
        conf=spec.conf,
        iou=spec.nms_iou,
        max_det=300,
        device=device,
        imgsz=spec.imgsz,
        verbose=verbose,
        augment=spec.augment,
    )


def fuse_predictions_wbf(
    predictions_by_model: list[list[dict]],
    idx2id_by_model: list[dict[int, int]],
    wbf_cfg: dict[str, Any] | None,
    model_weights: list[float] | None,
) -> list[dict]:
    """Fuse multi-model predictions by class-aware WBF."""
    if _weighted_boxes_fusion is None:
        raise ImportError(
            "ensemble-boxes is required for WBF ensemble. "
            "Install with: pip install ensemble-boxes"
        )
    if len(predictions_by_model) != len(idx2id_by_model):
        raise ValueError("predictions_by_model and idx2id_by_model length mismatch")

    n_models = len(predictions_by_model)
    if n_models == 0:
        return []

    if model_weights is None:
        model_weights = [1.0] * n_models
    if len(model_weights) != n_models:
        raise ValueError("model_weights length must match number of models")
    _validate_model_weights(model_weights)

    total_weight = sum(float(w) for w in model_weights)
    weight_ratios = [round(float(w) / total_weight, 6) for w in model_weights]
    logger.debug(
        "WBF model weights | raw=%s | ratio=%s",
        [float(w) for w in model_weights],
        weight_ratios,
    )

    cfg = wbf_cfg or {}
    iou_thr = float(cfg.get("iou_thr", 0.55))
    skip_box_thr = float(cfg.get("skip_box_thr", 0.0001))
    conf_type = str(cfg.get("conf_type", "avg")).strip().lower()
    if conf_type not in _ALLOWED_CONF_TYPES:
        raise ValueError(
            f"Unsupported wbf.conf_type={conf_type}. "
            f"Allowed: {sorted(_ALLOWED_CONF_TYPES)}"
        )

    by_stem: list[dict[str, dict]] = []
    all_stems: set[str] = set()
    for detections in predictions_by_model:
        det_map: dict[str, dict] = {}
        for det in detections:
            stem = str(det.get("image_stem", "")).strip()
            if not stem:
                continue
            det_map[stem] = det
        by_stem.append(det_map)
        all_stems.update(det_map.keys())

    fused_rows: list[dict] = []
    input_count = 0
    output_count = 0

    for stem in sorted(all_stems):
        image_id = _parse_image_id(stem)
        ref = _pick_reference_det(stem, by_stem)
        if ref is None:
            continue

        image_path = str(ref.get("image_path", ""))
        h, w = _pick_image_shape(stem, by_stem)
        if h <= 0 or w <= 0:
            logger.warning("skip image with invalid shape | stem=%s | shape=(%d,%d)", stem, h, w)
            continue

        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[int]] = []

        for model_idx, det_map in enumerate(by_stem):
            det = det_map.get(stem)
            model_boxes: list[list[float]] = []
            model_scores: list[float] = []
            model_labels: list[int] = []

            if det is not None:
                for box in det.get("boxes", []):
                    xyxy = box.get("xyxy")
                    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    x1 = min(max(x1, 0.0), float(w))
                    y1 = min(max(y1, 0.0), float(h))
                    x2 = min(max(x2, 0.0), float(w))
                    y2 = min(max(y2, 0.0), float(h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_idx = int(box.get("class_idx", -1))
                    category_id = int(idx2id_by_model[model_idx].get(class_idx, class_idx))
                    score = float(box.get("conf", 0.0))

                    # Normalize by original image size so WBF is consistent across mixed imgsz.
                    model_boxes.append(
                        [
                            x1 / float(w),
                            y1 / float(h),
                            x2 / float(w),
                            y2 / float(h),
                        ]
                    )
                    model_scores.append(score)
                    model_labels.append(category_id)
                    input_count += 1

            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)

        if not any(scores_list):
            continue

        fused_boxes, fused_scores, fused_labels = _weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=model_weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type=conf_type,
        )

        for idx in range(len(fused_scores)):
            x1n, y1n, x2n, y2n = [float(v) for v in fused_boxes[idx]]
            x1 = min(max(x1n, 0.0), 1.0) * float(w)
            y1 = min(max(y1n, 0.0), 1.0) * float(h)
            x2 = min(max(x2n, 0.0), 1.0) * float(w)
            y2 = min(max(y2n, 0.0), 1.0) * float(h)
            if x2 <= x1 or y2 <= y1:
                continue

            score = float(fused_scores[idx])
            category_id = int(round(float(fused_labels[idx])))
            output_count += 1

            fused_rows.append(
                {
                    "image_id": image_id,
                    "image_stem": stem,
                    "image_path": image_path,
                    "orig_shape": (h, w),
                    "category_id": category_id,
                    "bbox_x": x1,
                    "bbox_y": y1,
                    "bbox_w": x2 - x1,
                    "bbox_h": y2 - y1,
                    "score": score,
                }
            )

    logger.info(
        "WBF fused | models=%d | images=%d | input_boxes=%d | output_boxes=%d",
        n_models,
        len(all_stems),
        input_count,
        output_count,
    )
    return fused_rows


def build_debug_detections_from_rows(rows: list[dict]) -> list[dict]:
    """Convert row-style outputs into visualization-friendly detections."""
    by_image: dict[int, dict] = {}
    for row in rows:
        image_id = int(row["image_id"])
        entry = by_image.setdefault(
            image_id,
            {
                "image_path": str(row.get("image_path", "")),
                "image_stem": str(row.get("image_stem", image_id)),
                "orig_shape": tuple(row.get("orig_shape", (0, 0))),
                "boxes": [],
            },
        )
        x = float(row["bbox_x"])
        y = float(row["bbox_y"])
        w = float(row["bbox_w"])
        h = float(row["bbox_h"])
        entry["boxes"].append(
            {
                "class_idx": int(row["category_id"]),
                "conf": float(row["score"]),
                "xyxy": [x, y, x + w, y + h],
                "xywh": [x, y, w, h],
            }
        )
    return [by_image[k] for k in sorted(by_image.keys())]


def _parse_specs_from_config(
    runs_cfg: Any,
    *,
    paths_cfg: dict[str, Any],
    repo_root: Path,
    default_conf: float,
    default_nms_iou: float,
    default_imgsz: int,
    default_augment: bool,
) -> list[ModelSpec]:
    if not isinstance(runs_cfg, list):
        return []

    specs: list[ModelSpec] = []
    for item in runs_cfg:
        if not isinstance(item, dict):
            continue
        run_name = str(item.get("run_name", "")).strip()
        if not run_name:
            continue

        spec = _build_model_spec(
            run_name=run_name,
            weight_tag=str(item.get("weight_tag", "competition_best")).strip().lower(),
            conf=float(item.get("conf", default_conf)),
            nms_iou=float(item.get("nms_iou", default_nms_iou)),
            imgsz=int(item.get("imgsz", default_imgsz)),
            augment=bool(item.get("augment", default_augment)),
            model_weight=float(item.get("model_weight", 1.0)),
            paths_cfg=paths_cfg,
            repo_root=repo_root,
        )
        specs.append(spec)
    return specs


def _build_model_spec(
    *,
    run_name: str,
    weight_tag: str,
    conf: float,
    nms_iou: float,
    imgsz: int,
    augment: bool,
    model_weight: float,
    paths_cfg: dict[str, Any],
    repo_root: Path,
) -> ModelSpec:
    if weight_tag not in _ALLOWED_WEIGHT_TAGS:
        raise ValueError(
            f"Unsupported weight_tag='{weight_tag}'. Allowed: {sorted(_ALLOWED_WEIGHT_TAGS)}"
        )

    weight_path = _resolve_weight_path_for_run(
        run_name=run_name,
        weight_tag=weight_tag,
        paths_cfg=paths_cfg,
        repo_root=repo_root,
    )
    return ModelSpec(
        run_name=run_name,
        weight_tag=weight_tag,
        conf=float(conf),
        nms_iou=float(nms_iou),
        imgsz=int(imgsz),
        augment=bool(augment),
        model_weight=float(model_weight),
        weight_path=weight_path,
    )


def _resolve_weight_path_for_run(
    *,
    run_name: str,
    weight_tag: str,
    paths_cfg: dict[str, Any],
    repo_root: Path,
) -> Path:
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()

    weights_dir = runs_base / run_name / "weights"
    if weight_tag == "competition_best":
        competition_pt = weights_dir / "competition_best.pt"
        best_pt = weights_dir / "best.pt"
        if competition_pt.exists():
            return competition_pt
        if best_pt.exists():
            return best_pt
        raise FileNotFoundError(
            f"weights not found for run='{run_name}' (checked: {competition_pt}, {best_pt})"
        )

    weight_path = weights_dir / f"{weight_tag}.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"weight file not found for run='{run_name}': {weight_path}"
        )
    return weight_path


def _parse_cli_ensemble_runs(raw: str) -> list[str]:
    if not raw:
        return []
    names = [token.strip() for token in raw.split(",") if token.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _validate_model_weights(weights: list[float]) -> None:
    for idx, weight in enumerate(weights):
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"Invalid model_weight at index={idx}: {weight} (must be finite)")
        if value <= 0.0:
            raise ValueError(f"Invalid model_weight at index={idx}: {weight} (must be > 0)")


def _pick_reference_det(stem: str, by_stem: list[dict[str, dict]]) -> dict | None:
    for det_map in by_stem:
        det = det_map.get(stem)
        if det is not None:
            return det
    return None


def _pick_image_shape(stem: str, by_stem: list[dict[str, dict]]) -> tuple[int, int]:
    for det_map in by_stem:
        det = det_map.get(stem)
        if det is None:
            continue
        shape = det.get("orig_shape", (0, 0))
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            continue
        h = int(shape[0])
        w = int(shape[1])
        if h > 0 and w > 0:
            return h, w
    return 0, 0


def _parse_image_id(stem: str) -> int:
    try:
        return int(stem)
    except ValueError:
        return abs(hash(stem)) % (10**9)
