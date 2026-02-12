"""STAGE 4: Build submission CSV from single-model or ensemble inference."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from shutil import copy2
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.ensemble_wbf import (  # noqa: E402
    build_debug_detections_from_rows,
    fuse_predictions_wbf,
    load_idx2id_for_run,
    predict_for_spec,
    resolve_ensemble_model_specs,
    validate_category_map_compatibility,
)
from src.inference.postprocess import (  # noqa: E402
    apply_submission_filters_and_topk,
    postprocess_detections,
)
from src.inference.predictor import batch_predict  # noqa: E402
from src.inference.submission import (  # noqa: E402
    validate_submission,
    write_submission,
    write_submission_manifest,
)
from src.inference.submission_debug import save_submission_debug_images  # noqa: E402
from src.utils.config_loader import load_experiment_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _resolve_cli_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _release_cuda_cache_if_possible(device: Any) -> None:
    """Best-effort memory cleanup between ensemble model runs."""
    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gc.collect() skipped: %s", exc)

    try:
        import torch
    except Exception:
        logger.debug("torch not available; skip cuda cache release")
        return

    try:
        if isinstance(device, str) and device.strip().lower() == "cpu":
            logger.debug("device=cpu; skip cuda cache release")
            return
        if not torch.cuda.is_available():
            logger.debug("cuda unavailable; skip cuda cache release")
            return
        torch.cuda.empty_cache()
        logger.debug("cuda cache released")
    except Exception as exc:  # noqa: BLE001
        logger.debug("cuda cache release skipped: %s", exc)


def _parse_int_csv(raw: str) -> set[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return {int(v) for v in values}


def _load_class_map_csv(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"class_map.csv is empty: {path}")
        missing = {"class_index", "category_id"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"class_map.csv missing columns: {sorted(missing)}")
        for row in reader:
            mapping[int(row["class_index"])] = int(row["category_id"])
    if not mapping:
        raise ValueError(f"class_map.csv has no rows: {path}")
    return mapping


def _load_class_min_conf_csv(path: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"class_min_conf csv is empty: {path}")
        missing = {"category_id", "min_conf"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"class_min_conf csv missing columns: {sorted(missing)}")
        for row in reader:
            mapping[int(row["category_id"])] = float(row["min_conf"])
    return mapping


def _load_label_map_for_run(run_name: str, paths_cfg: dict, repo_root: Path) -> dict:
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()
    label_map_path = processed_base / run_name / "label_map_full.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map_full.json not found: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_category_name_map(label_map: dict) -> dict[int, str]:
    idx2id_raw = label_map.get("idx2id", {})
    names = label_map.get("names", [])
    out: dict[int, str] = {}
    if not isinstance(idx2id_raw, dict) or not isinstance(names, list):
        return out
    for idx_str, cid in idx2id_raw.items():
        try:
            idx = int(idx_str)
            category_id = int(cid)
        except Exception:
            continue
        if 0 <= idx < len(names):
            name = names[idx]
            if isinstance(name, (str, int, float)):
                out[category_id] = str(name)
    return out


def _resolve_submission_output_path(
    *,
    out_arg: str,
    run_name: str,
    conf: float,
    nms_iou: float,
    imgsz: int,
    min_conf: float,
    max_det_per_image: int,
    submissions_dir: Path,
    repo_root: Path,
) -> Path:
    if out_arg:
        out_path = _resolve_cli_path(out_arg, repo_root)
        if out_path.suffix.lower() == ".csv":
            return out_path
        out_path.mkdir(parents=True, exist_ok=True)
        csv_filename = (
            f"{run_name}_conf{conf:.2f}_iou{nms_iou:.2f}_img{imgsz}_"
            f"min{min_conf:.2f}_top{max_det_per_image}.csv"
        )
        return out_path / csv_filename
    return submissions_dir / f"{run_name}_conf{conf}.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 4: 제출 파일 생성")
    parser.add_argument("--run-name", required=True, help="실험 이름")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="confidence threshold (기본: config 의 submission.conf)",
    )
    parser.add_argument("--device", default=None, help="GPU 디바이스 (예: 0, cpu)")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    # Existing compatibility options
    parser.add_argument("--weights", type=str, default=None, help="가중치(.pt) 경로 직접 지정")
    parser.add_argument("--test-dir", type=str, default=None, help="테스트 이미지 디렉터리 직접 지정")
    parser.add_argument("--class-map", type=str, default=None, help="class_map.csv 경로 직접 지정")
    parser.add_argument("--imgsz", type=int, default=None, help="추론 이미지 크기 오버라이드")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 임계값 오버라이드")
    parser.add_argument("--min-conf", type=float, default=None, help="후처리 최소 score 임계값")
    parser.add_argument(
        "--class-min-conf-csv",
        type=str,
        default=None,
        help="카테고리별 최소 score csv(category_id,min_conf)",
    )
    parser.add_argument("--topk", type=int, default=None, help="이미지당 최대 박스 수 오버라이드")
    parser.add_argument("--out", type=str, default="", help="출력 CSV 경로(또는 디렉터리)")
    parser.add_argument("--keep-category-ids", type=str, default="", help="허용 category_id csv")
    parser.add_argument("--keep-category-ids-file", type=str, default="", help="허용 category_id 파일")

    # New minimal ensemble overrides
    parser.add_argument(
        "--ensemble-enable",
        action="store_true",
        help="submission.ensemble을 강제로 활성화",
    )
    parser.add_argument(
        "--ensemble-runs",
        type=str,
        default="",
        help="submission.ensemble.runs를 대체할 run_name csv",
    )

    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()
    logger.info("STAGE 4 시작 | run_name=%s | config=%s", run_name, config_path)

    config, repo_root = load_experiment_config(config_path, script_path)

    paths_cfg = config.get("paths", {})
    sub_cfg = config.get("submission", {})
    ensemble_cfg = sub_cfg.get("ensemble", {}) if isinstance(sub_cfg.get("ensemble", {}), dict) else {}
    debug_cfg = sub_cfg.get("debug", {}) if isinstance(sub_cfg.get("debug", {}), dict) else {}
    train_cfg = config.get("train", {})

    artifact_layout = str(paths_cfg.get("artifact_layout", "legacy")).strip().lower()
    if artifact_layout not in {"legacy", "compact"}:
        logger.warning("unknown paths.artifact_layout=%s, fallback=legacy", artifact_layout)
        artifact_layout = "legacy"

    conf = args.conf if args.conf is not None else float(sub_cfg.get("conf", 0.25))
    nms_iou = args.iou if args.iou is not None else float(sub_cfg.get("nms_iou", 0.5))
    max_det_per_image = args.topk if args.topk is not None else int(sub_cfg.get("max_det_per_image", 4))
    min_conf = args.min_conf if args.min_conf is not None else float(sub_cfg.get("min_conf", 0.0))
    augment = bool(sub_cfg.get("augment", False))
    debug_enabled = bool(debug_cfg.get("enabled", True))
    debug_sample_size = int(debug_cfg.get("sample_size", 12))
    debug_seed = int(debug_cfg.get("seed", 42))
    device = args.device if args.device is not None else sub_cfg.get("device")
    if device is not None and isinstance(device, str) and device.isdigit():
        device = int(device)
    imgsz = args.imgsz if args.imgsz is not None else int(train_cfg.get("imgsz", 640))

    ensemble_enabled = bool(ensemble_cfg.get("enabled", False)) or bool(args.ensemble_enable)

    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name

    if artifact_layout == "compact":
        debug_output_dir = run_dir / "submit" / "debug"
        manifest_output_path = run_dir / "submit" / "submission_manifest.json"
        root_manifest_shortcut = run_dir / "submission_manifest.json"
    else:
        debug_output_dir = run_dir / "submission_debug"
        manifest_output_path = run_dir / "submission_manifest.json"
        root_manifest_shortcut = None

    if args.test_dir:
        test_images_dir = _resolve_cli_path(args.test_dir, repo_root)
    else:
        test_images_dir = Path(paths_cfg.get("test_images_dir", "data/raw/test_images"))
        if not test_images_dir.is_absolute():
            test_images_dir = (repo_root / test_images_dir).resolve()
    if not test_images_dir.exists():
        logger.error("테스트 이미지 디렉터리가 존재하지 않습니다: %s", test_images_dir)
        sys.exit(1)

    test_images = (
        sorted(test_images_dir.glob("*.png"))
        + sorted(test_images_dir.glob("*.jpg"))
        + sorted(test_images_dir.glob("*.jpeg"))
    )
    n_test_images = len(test_images)
    if n_test_images == 0:
        logger.error("테스트 이미지가 없습니다: %s", test_images_dir)
        sys.exit(1)

    submissions_dir = Path(paths_cfg.get("submissions_dir", "artifacts/submissions"))
    if not submissions_dir.is_absolute():
        submissions_dir = (repo_root / submissions_dir).resolve()
    submissions_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _resolve_submission_output_path(
        out_arg=args.out,
        run_name=run_name,
        conf=conf,
        nms_iou=nms_iou,
        imgsz=imgsz,
        min_conf=min_conf,
        max_det_per_image=max_det_per_image,
        submissions_dir=submissions_dir,
        repo_root=repo_root,
    )

    class_min_conf_by_category: dict[int, float] = {}
    class_min_conf_csv = args.class_min_conf_csv
    if class_min_conf_csv is None:
        cfg_csv = sub_cfg.get("class_min_conf_csv")
        class_min_conf_csv = cfg_csv if isinstance(cfg_csv, str) and cfg_csv.strip() else None
    if class_min_conf_csv:
        class_min_conf_path = _resolve_cli_path(class_min_conf_csv, repo_root)
        if not class_min_conf_path.exists():
            logger.error("class_min_conf csv 가 존재하지 않습니다: %s", class_min_conf_path)
            sys.exit(1)
        class_min_conf_by_category = _load_class_min_conf_csv(class_min_conf_path)

    keep_ids: set[int] = set()
    cfg_keep_ids = sub_cfg.get("keep_category_ids")
    if isinstance(cfg_keep_ids, list):
        keep_ids.update(int(v) for v in cfg_keep_ids)
    if args.keep_category_ids:
        keep_ids.update(_parse_int_csv(args.keep_category_ids))
    if args.keep_category_ids_file:
        keep_file = _resolve_cli_path(args.keep_category_ids_file, repo_root)
        if not keep_file.exists():
            logger.error("keep_category_ids 파일이 존재하지 않습니다: %s", keep_file)
            sys.exit(1)
        for line in keep_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            if "," in text:
                keep_ids.update(_parse_int_csv(text))
            else:
                keep_ids.add(int(text))
    keep_category_ids = keep_ids if keep_ids else None

    logger.info(
        "추론 시작 | ensemble=%s | conf=%.3f | nms_iou=%.3f | max_det=%d | min_conf=%.3f | augment=%s",
        ensemble_enabled,
        conf,
        nms_iou,
        max_det_per_image,
        min_conf,
        augment,
    )

    rows: list[dict]
    debug_detections: list[dict]
    idx2name_for_debug: dict[int, str]
    valid_category_ids: set[int]
    selected_weight_label: str
    ensemble_meta: dict[str, Any] | None = None

    if ensemble_enabled:
        if args.class_map:
            logger.warning("--class-map is ignored in ensemble mode (run label_map is required)")
        if args.weights:
            logger.warning("--weights is ignored in ensemble mode (spec-defined run weights are used)")

        try:
            specs = resolve_ensemble_model_specs(config, args, repo_root)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to resolve ensemble specs: %s", exc)
            sys.exit(1)
        logger.info("ensemble enabled | models=%d", len(specs))
        for spec in specs:
            logger.info(
                "  run=%s | weight_tag=%s | weight=%s | model_weight=%.3f",
                spec.run_name,
                spec.weight_tag,
                spec.weight_path,
                spec.model_weight,
            )
        logger.info("ensemble pipeline order | predict -> wbf -> filters(topk)")

        idx2id_by_model: list[dict[int, int]] = []
        predictions_by_model: list[list[dict]] = []
        model_weights: list[float] = []
        category_name_map: dict[int, str] = {}
        ensemble_models_meta: list[dict[str, Any]] = []
        ensemble_started_at = time.perf_counter()
        n_specs = len(specs)

        for spec_idx, spec in enumerate(specs, start=1):
            try:
                idx2id_run = load_idx2id_for_run(spec.run_name, paths_cfg, repo_root)
            except Exception as exc:  # noqa: BLE001
                logger.error("failed to load label_map for run=%s: %s", spec.run_name, exc)
                sys.exit(1)
            idx2id_by_model.append(idx2id_run)
            model_weights.append(spec.model_weight)

            logger.info(
                "ensemble progress | model=%d/%d | run=%s | stage=predict_start",
                spec_idx,
                n_specs,
                spec.run_name,
            )
            predict_started_at = time.perf_counter()
            try:
                model_predictions = predict_for_spec(
                    spec=spec,
                    test_images_dir=test_images_dir,
                    device=device,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("ensemble predict failed | run=%s | error=%s", spec.run_name, exc)
                sys.exit(1)
            predict_elapsed = time.perf_counter() - predict_started_at

            pred_images = len(model_predictions)
            pred_boxes = sum(len(det.get("boxes", [])) for det in model_predictions)
            predictions_by_model.append(model_predictions)
            ensemble_models_meta.append(
                {
                    "index": spec_idx,
                    "run_name": spec.run_name,
                    "weight_tag": spec.weight_tag,
                    "weight_path": str(spec.weight_path),
                    "model_weight": float(spec.model_weight),
                    "conf": float(spec.conf),
                    "nms_iou": float(spec.nms_iou),
                    "imgsz": int(spec.imgsz),
                    "augment": bool(spec.augment),
                    "pred_images": int(pred_images),
                    "pred_boxes": int(pred_boxes),
                    "elapsed_sec": round(float(predict_elapsed), 3),
                }
            )
            logger.info(
                "ensemble progress | model=%d/%d | run=%s | stage=predict_done | seconds=%.3f | images=%d | boxes=%d",
                spec_idx,
                n_specs,
                spec.run_name,
                predict_elapsed,
                pred_images,
                pred_boxes,
            )
            _release_cuda_cache_if_possible(device)

            try:
                label_map = _load_label_map_for_run(spec.run_name, paths_cfg, repo_root)
                category_name_map.update(_build_category_name_map(label_map))
            except Exception as exc:  # noqa: BLE001
                logger.warning("label names unavailable for run=%s: %s", spec.run_name, exc)

        strict_category_map = bool(ensemble_cfg.get("strict_category_map", True))
        if strict_category_map:
            try:
                valid_category_ids = validate_category_map_compatibility(idx2id_by_model)
                logger.info("ensemble category-map check: strict PASS | categories=%d", len(valid_category_ids))
            except Exception as exc:  # noqa: BLE001
                logger.error("ensemble category-map check failed: %s", exc)
                sys.exit(1)
        else:
            sets = [set(m.values()) for m in idx2id_by_model]
            valid_category_ids = set.intersection(*sets) if sets else set()
            logger.warning(
                "ensemble strict_category_map=false | using intersection category_ids=%d",
                len(valid_category_ids),
            )

        if keep_category_ids is not None:
            unknown_ids = sorted(keep_category_ids - valid_category_ids)
            if unknown_ids:
                logger.warning("unknown keep_category_ids (not in mapping): %s", unknown_ids)

        wbf_cfg = ensemble_cfg.get("wbf", {})
        wbf_cfg_norm = wbf_cfg if isinstance(wbf_cfg, dict) else {}
        wbf_iou_thr = float(wbf_cfg_norm.get("iou_thr", 0.55))
        wbf_skip_box_thr = float(wbf_cfg_norm.get("skip_box_thr", 0.0001))
        wbf_conf_type = str(wbf_cfg_norm.get("conf_type", "avg")).strip().lower()
        logger.info("ensemble progress | stage=fuse_start")
        fuse_started_at = time.perf_counter()
        try:
            fused_rows = fuse_predictions_wbf(
                predictions_by_model=predictions_by_model,
                idx2id_by_model=idx2id_by_model,
                wbf_cfg=wbf_cfg_norm,
                model_weights=model_weights,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ensemble WBF fusion failed: %s", exc)
            sys.exit(1)
        fuse_elapsed_sec = time.perf_counter() - fuse_started_at
        rows = apply_submission_filters_and_topk(
            fused_rows,
            max_det_per_image=max_det_per_image,
            min_conf=min_conf,
            class_min_conf_by_category=class_min_conf_by_category,
            keep_category_ids=keep_category_ids,
        )
        debug_detections = build_debug_detections_from_rows(rows)
        idx2name_for_debug = {int(k): str(v) for k, v in category_name_map.items()}

        input_boxes = sum(int(m["pred_boxes"]) for m in ensemble_models_meta)
        total_elapsed_sec = time.perf_counter() - ensemble_started_at
        logger.info(
            "ensemble progress | stage=fuse_done | seconds=%.3f | input_boxes=%d | fused_boxes=%d | final_rows=%d",
            fuse_elapsed_sec,
            input_boxes,
            len(fused_rows),
            len(rows),
        )
        logger.info(
            "ensemble fusion stats | input_boxes=%d | fused_boxes=%d | final_rows=%d | total_seconds=%.3f",
            input_boxes,
            len(fused_rows),
            len(rows),
            total_elapsed_sec,
        )
        selected_weight_label = "ensemble(" + ", ".join(spec.run_name for spec in specs) + ")"
        ensemble_meta = {
            "ensemble_enabled": True,
            "ensemble_method": "wbf",
            "ensemble_runs": [spec.run_name for spec in specs],
            "ensemble_strict_category_map": strict_category_map,
            "ensemble_weight_policy": {
                "validation": "strict_positive_finite",
                "normalization": "none",
            },
            "ensemble_wbf": {
                "iou_thr": wbf_iou_thr,
                "skip_box_thr": wbf_skip_box_thr,
                "conf_type": wbf_conf_type,
            },
            "ensemble_models": ensemble_models_meta,
            "ensemble_stats": {
                "input_boxes": int(input_boxes),
                "fused_boxes": int(len(fused_rows)),
                "final_rows": int(len(rows)),
                "fuse_elapsed_sec": round(float(fuse_elapsed_sec), 3),
                "total_elapsed_sec": round(float(total_elapsed_sec), 3),
            },
        }
    else:
        if args.weights:
            selected_weight = _resolve_cli_path(args.weights, repo_root)
            best_pt = selected_weight
        else:
            weights_dir = run_dir / "weights"
            competition_pt = weights_dir / "competition_best.pt"
            best_pt = weights_dir / "best.pt"
            selected_weight = competition_pt if competition_pt.exists() else best_pt

        if not selected_weight.exists():
            logger.error("가중치 파일이 존재하지 않습니다: %s", selected_weight)
            logger.error("확인 경로: best=%s", best_pt)
            logger.error("STAGE 2 를 먼저 실행하거나 --weights 를 직접 지정하세요.")
            sys.exit(1)

        if args.class_map:
            class_map_path = _resolve_cli_path(args.class_map, repo_root)
            if not class_map_path.exists():
                logger.error("class_map.csv 가 존재하지 않습니다: %s", class_map_path)
                sys.exit(1)
            idx2id = _load_class_map_csv(class_map_path)
            idx2name = {}
            logger.info("class_map.csv 로드 | rows=%d", len(idx2id))
        else:
            try:
                label_map = _load_label_map_for_run(run_name, paths_cfg, repo_root)
            except FileNotFoundError as exc:
                logger.error(str(exc))
                logger.error("STAGE 0 를 먼저 실행하거나 --class-map 을 직접 지정하세요.")
                sys.exit(1)
            idx2id = {int(k): int(v) for k, v in label_map.get("idx2id", {}).items()}
            names = label_map.get("names", [])
            idx2name = {
                i: str(name)
                for i, name in enumerate(names)
                if isinstance(name, (str, int, float))
            }
            logger.info("label_map 로드 | num_classes=%d", int(label_map.get("num_classes", 0)))

        valid_category_ids = set(int(v) for v in idx2id.values())
        if keep_category_ids is not None:
            unknown_ids = sorted(keep_category_ids - valid_category_ids)
            if unknown_ids:
                logger.warning("unknown keep_category_ids (not in mapping): %s", unknown_ids)

        logger.info("제출 가중치 선택 | selected=%s", selected_weight)
        detections = batch_predict(
            weights_path=selected_weight,
            source=test_images_dir,
            conf=conf,
            iou=nms_iou,
            max_det=max_det_per_image * 5,
            device=device,
            imgsz=imgsz,
            verbose=args.verbose,
            augment=augment,
        )
        rows = postprocess_detections(
            detections,
            idx2id=idx2id,
            max_det_per_image=max_det_per_image,
            min_conf=min_conf,
            class_min_conf_by_category=class_min_conf_by_category,
            keep_category_ids=keep_category_ids,
        )
        debug_detections = detections
        idx2name_for_debug = idx2name
        selected_weight_label = str(selected_weight)
        ensemble_meta = {
            "ensemble_enabled": False,
        }

    debug_report = save_submission_debug_images(
        run_dir=run_dir,
        detections=debug_detections,
        idx2name=idx2name_for_debug,
        max_det_per_image=max_det_per_image,
        output_dir=debug_output_dir,
        sample_size=debug_sample_size,
        seed=debug_seed,
        enabled=debug_enabled,
    )
    logger.info(
        "submission debug | enabled=%s | requested=%d | saved=%d | dir=%s",
        debug_report.get("debug_enabled", False),
        debug_report.get("debug_sample_requested", 0),
        debug_report.get("debug_sample_saved", 0),
        debug_report.get("debug_output_dir", ""),
    )
    if debug_report.get("debug_skipped_reason"):
        logger.warning("submission debug skip/fallback: %s", debug_report["debug_skipped_reason"])

    write_submission(rows, csv_path)
    logger.info("표준 제출 포맷(8컬럼) CSV 생성 완료")

    report = validate_submission(
        csv_path,
        max_det_per_image=max_det_per_image,
        valid_category_ids=valid_category_ids,
    )

    manifest_path = write_submission_manifest(
        report,
        run_dir=run_dir,
        conf=conf,
        n_test_images=n_test_images,
        csv_path=csv_path,
        debug_report=debug_report,
        output_path=manifest_output_path,
    )
    if ensemble_meta is not None:
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            manifest_payload.update(ensemble_meta)
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest_payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to append ensemble metadata to manifest: %s", exc)

    if root_manifest_shortcut is not None and manifest_path.exists():
        if manifest_path.resolve() != root_manifest_shortcut.resolve():
            root_manifest_shortcut.parent.mkdir(parents=True, exist_ok=True)
            copy2(str(manifest_path), str(root_manifest_shortcut))
            logger.info("submission_manifest shortcut 갱신 | %s -> %s", manifest_path, root_manifest_shortcut)

    logger.info("=" * 60)
    logger.info("STAGE 4 완료")
    logger.info("  run_name       : %s", run_name)
    logger.info("  artifact_layout: %s", artifact_layout)
    logger.info("  weight_file    : %s", selected_weight_label)
    logger.info("  test_images    : %d", n_test_images)
    logger.info("  ensemble       : %s", ensemble_enabled)
    logger.info("  conf_threshold : %.3f", conf)
    logger.info("  min_conf       : %.3f", min_conf)
    logger.info("  max_det/image  : %d", max_det_per_image)
    logger.info("  submission CSV : %s", csv_path)
    logger.info("  manifest       : %s", manifest_path)
    logger.info("  debug_dir      : %s", debug_report.get("debug_output_dir", ""))
    logger.info("  total rows     : %d", report["total_rows"])
    logger.info("  n_images       : %d", report["n_images"])
    logger.info("  validation     : %s", "PASS" if report["valid"] else "FAIL")
    if class_min_conf_by_category:
        logger.info("  class min conf : %d overrides", len(class_min_conf_by_category))
    if keep_category_ids is not None:
        logger.info("  keep categories: %d", len(keep_category_ids))
    if report["errors"]:
        logger.warning("  errors         : %d", len(report["errors"]))
    if report["warnings"]:
        logger.warning("  warnings       : %d", len(report["warnings"]))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
