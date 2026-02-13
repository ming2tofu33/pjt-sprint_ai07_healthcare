"""STAGE 4: 제출 파일 생성.

STAGE 2 산출물(``competition_best.pt`` 우선, 없으면 ``best.pt``)과 테스트 이미지,
label_map 을 사용하여 추론 → 후처리(Top-4) → 제출 CSV 생성 → 검증까지 수행한다.

사용법::

    python scripts/4_submission.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--conf 0.25] [--device 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ── 프로젝트 루트를 sys.path 에 추가 ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_experiment_config
from src.utils.logger import get_logger
from src.inference.predictor import batch_predict
from src.inference.postprocess import postprocess_detections
from src.inference.submission_debug import save_submission_debug_images
from src.inference.submission import (
    write_submission,
    validate_submission,
    write_submission_manifest,
)

logger = get_logger(__name__)


def _resolve_cli_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _parse_int_csv(raw: str) -> set[int]:
    values = [x.strip().lstrip("\ufeff") for x in raw.split(",") if x.strip()]
    return {int(v) for v in values}


def _load_class_map_csv(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"class_map.csv 가 비어 있습니다: {path}")
        missing = {"class_index", "category_id"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"class_map.csv 필수 컬럼 누락: {sorted(missing)}")
        for row in reader:
            mapping[int(row["class_index"])] = int(row["category_id"])
    if not mapping:
        raise ValueError(f"class_map.csv 데이터가 없습니다: {path}")
    return mapping


def _load_class_min_conf_csv(path: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"class_min_conf csv 가 비어 있습니다: {path}")
        missing = {"category_id", "min_conf"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"class_min_conf csv 필수 컬럼 누락: {sorted(missing)}")
        for row in reader:
            mapping[int(row["category_id"])] = float(row["min_conf"])
    return mapping


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

    # make_submit_csv.py 호환을 위한 선택 옵션
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

    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 4 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    config, repo_root = load_experiment_config(config_path, script_path)

    paths_cfg = config.get("paths", {})
    sub_cfg = config.get("submission", {})
    debug_cfg = sub_cfg.get("debug", {}) if isinstance(sub_cfg.get("debug", {}), dict) else {}
    train_cfg = config.get("train", {})

    # confidence threshold: CLI > config
    conf = args.conf if args.conf is not None else float(sub_cfg.get("conf", 0.25))
    nms_iou = args.iou if args.iou is not None else float(sub_cfg.get("nms_iou", 0.5))
    max_det_per_image = args.topk if args.topk is not None else int(sub_cfg.get("max_det_per_image", 4))
    min_conf = args.min_conf if args.min_conf is not None else float(sub_cfg.get("min_conf", 0.0))
    augment = bool(sub_cfg.get("augment", False))
    debug_enabled = bool(debug_cfg.get("enabled", True))
    debug_sample_size = int(debug_cfg.get("sample_size", 12))
    debug_seed = int(debug_cfg.get("seed", 42))
    device = args.device
    if device is None:
        device = sub_cfg.get("device")
    if device is not None and isinstance(device, str) and device.isdigit():
        device = int(device)
    imgsz = args.imgsz if args.imgsz is not None else int(train_cfg.get("imgsz", 640))

    # ── 2) 경로 결정 ────────────────────────────────────────
    # runs/<run_name>/weights/{competition_best.pt|best.pt}
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name

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

    # 테스트 이미지 디렉터리
    if args.test_dir:
        test_images_dir = _resolve_cli_path(args.test_dir, repo_root)
    else:
        test_images_dir = Path(paths_cfg.get("test_images_dir", "data/raw/test_images"))
        if not test_images_dir.is_absolute():
            test_images_dir = (repo_root / test_images_dir).resolve()

    if not test_images_dir.exists():
        logger.error("테스트 이미지 디렉터리가 존재하지 않습니다: %s", test_images_dir)
        sys.exit(1)

    # 제출 CSV 출력 경로
    submissions_dir = Path(paths_cfg.get("submissions_dir", "artifacts/submissions"))
    if not submissions_dir.is_absolute():
        submissions_dir = (repo_root / submissions_dir).resolve()
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = _resolve_cli_path(args.out, repo_root)
        if out_path.suffix.lower() == ".csv":
            csv_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            csv_filename = (
                f"{run_name}_conf{conf:.2f}_iou{nms_iou:.2f}_img{imgsz}_"
                f"min{min_conf:.2f}_top{max_det_per_image}.csv"
            )
            csv_path = out_path / csv_filename
    else:
        csv_filename = f"{run_name}_conf{conf}.csv"
        csv_path = submissions_dir / csv_filename

    # ── 3) label_map 로드 ──────────────────────────────────
    if args.class_map:
        class_map_path = _resolve_cli_path(args.class_map, repo_root)
        if not class_map_path.exists():
            logger.error("class_map.csv 가 존재하지 않습니다: %s", class_map_path)
            sys.exit(1)
        idx2id = _load_class_map_csv(class_map_path)
        idx2name: dict[int, str] = {}
        logger.info("class_map.csv 로드 | rows=%d", len(idx2id))
    else:
        # label_map_full.json (idx2id 매핑 필요)
        processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
        if not processed_base.is_absolute():
            processed_base = (repo_root / processed_base).resolve()
        cache_dir = processed_base / run_name

        label_map_path = cache_dir / "label_map_full.json"
        if not label_map_path.exists():
            logger.error("label_map_full.json 이 존재하지 않습니다: %s", label_map_path)
            logger.error("STAGE 0 을 먼저 실행하세요: python scripts/0_split_data.py --run-name %s ...", run_name)
            logger.error("또는 --class-map 을 직접 지정하세요.")
            sys.exit(1)

        with label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)

        idx2id = {int(k): int(v) for k, v in label_map.get("idx2id", {}).items()}
        names = label_map.get("names", [])
        idx2name = {
            i: str(name) for i, name in enumerate(names)
            if isinstance(name, (str, int, float))
        }
        nc = label_map.get("num_classes", 0)
        logger.info("label_map 로드 | num_classes=%d", nc)

    valid_category_ids = set(int(v) for v in idx2id.values())

    # 카테고리별 최소 score 설정 (추가 기능)
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

    # keep_category_ids 설정 (config + CLI 합집합, 추가 기능)
    keep_category_ids: set[int] | None = None
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
        for line in keep_file.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text:
                continue
            if "," in text:
                keep_ids.update(_parse_int_csv(text))
            else:
                keep_ids.add(int(text))

    if keep_ids:
        keep_category_ids = keep_ids
        unknown_ids = sorted(keep_category_ids - valid_category_ids)
        if unknown_ids:
            logger.warning("매핑에 없는 keep_category_ids 가 있습니다: %s", unknown_ids)

    # ── 4) 테스트 이미지 스캔 ──────────────────────────────
    test_images = sorted(test_images_dir.glob("*.png")) + \
                  sorted(test_images_dir.glob("*.jpg")) + \
                  sorted(test_images_dir.glob("*.jpeg"))
    n_test_images = len(test_images)

    if n_test_images == 0:
        logger.error("테스트 이미지가 없습니다: %s", test_images_dir)
        sys.exit(1)

    logger.info("테스트 이미지 %d 장 발견", n_test_images)

    # ── 5) 배치 추론 ─────────────────────────────────────────
    logger.info(
        "추론 시작 | conf=%.3f | nms_iou=%.3f | max_det_per_image=%d | min_conf=%.3f | augment(TTA)=%s",
        conf,
        nms_iou,
        max_det_per_image,
        min_conf,
        augment,
    )

    logger.info("제출 가중치 선택 | selected=%s | fallback=%s", selected_weight, best_pt)
    detections = batch_predict(
        weights_path=selected_weight,
        source=test_images_dir,
        conf=conf,
        iou=nms_iou,
        max_det=max_det_per_image * 5,  # Ultralytics 에는 여유있게 전달
        device=device,
        imgsz=imgsz,
        verbose=args.verbose,
        augment=augment,
    )

    # ── 6) 제출 전 시각 sanity check 저장 (실패해도 계속 진행) ───────
    debug_report = save_submission_debug_images(
        run_dir=run_dir,
        detections=detections,
        idx2name=idx2name,
        max_det_per_image=max_det_per_image,
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

    # ── 7) 후처리: Top-K + class→category_id + score/카테고리 필터(추가 기능) ──
    rows = postprocess_detections(
        detections,
        idx2id=idx2id,
        max_det_per_image=max_det_per_image,
        min_conf=min_conf,
        class_min_conf_by_category=class_min_conf_by_category,
        keep_category_ids=keep_category_ids,
    )

    # ── 8) 제출 CSV 저장 ────────────────────────────────────
    write_submission(rows, csv_path)
    logger.info("대회 제출 포맷(8컬럼) CSV 생성 완료")

    # ── 9) 제출 검증 ────────────────────────────────────────
    report = validate_submission(
        csv_path,
        max_det_per_image=max_det_per_image,
        valid_category_ids=valid_category_ids,
    )

    # ── 10) 매니페스트 저장 ─────────────────────────────────
    write_submission_manifest(
        report,
        run_dir=run_dir,
        conf=conf,
        n_test_images=n_test_images,
        csv_path=csv_path,
        debug_report=debug_report,
    )

    # ── 11) 요약 출력 ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 완료")
    logger.info("  run_name       : %s", run_name)
    logger.info("  weight_file    : %s", selected_weight)
    logger.info("  test_images    : %d", n_test_images)
    logger.info("  conf_threshold : %.3f", conf)
    logger.info("  min_conf       : %.3f", min_conf)
    logger.info("  max_det/image  : %d", max_det_per_image)
    logger.info("  submission CSV : %s", csv_path)
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
