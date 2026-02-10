#!/usr/bin/env python
"""STAGE 4: 제출 파일 생성.

STAGE 2 산출물(``best.pt``)과 테스트 이미지, label_map 을 사용하여
추론 → 후처리(Top-4) → 제출 CSV 생성 → 검증까지 수행한다.

사용법::

    python scripts/4_submission.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--conf 0.25] [--device 0]
"""
from __future__ import annotations

import argparse
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
from src.inference.submission import (
    write_submission,
    validate_submission,
    write_submission_manifest,
)

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 4: 제출 파일 생성")
    parser.add_argument("--run-name", required=True, help="실험 이름")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--conf", type=float, default=None,
                        help="confidence threshold (기본: config 의 submission.conf)")
    parser.add_argument("--device", default=None, help="GPU 디바이스 (예: 0, cpu)")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 억제")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 4 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    config, repo_root = load_experiment_config(config_path, script_path)

    paths_cfg = config.get("paths", {})
    sub_cfg = config.get("submission", {})
    train_cfg = config.get("train", {})

    # confidence threshold: CLI > config
    conf = args.conf if args.conf is not None else float(sub_cfg.get("conf", 0.25))
    nms_iou = float(sub_cfg.get("nms_iou", 0.5))
    max_det_per_image = int(sub_cfg.get("max_det_per_image", 4))
    device = args.device
    if device is None:
        device = sub_cfg.get("device")
    if device is not None and isinstance(device, str) and device.isdigit():
        device = int(device)
    imgsz = int(train_cfg.get("imgsz", 640))

    # ── 2) 경로 결정 ────────────────────────────────────────
    # best.pt
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        logger.error("best.pt 가 존재하지 않습니다: %s", best_pt)
        logger.error("STAGE 2 를 먼저 실행하세요: python scripts/2_train.py --run-name %s ...", run_name)
        sys.exit(1)

    # 테스트 이미지 디렉터리
    test_images_dir = Path(paths_cfg.get("test_images_dir", "data/raw/test_images"))
    if not test_images_dir.is_absolute():
        test_images_dir = (repo_root / test_images_dir).resolve()

    if not test_images_dir.exists():
        logger.error("테스트 이미지 디렉터리가 존재하지 않습니다: %s", test_images_dir)
        sys.exit(1)

    # label_map_full.json (idx2id 매핑 필요)
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()
    cache_dir = processed_base / run_name

    label_map_path = cache_dir / "label_map_full.json"
    if not label_map_path.exists():
        logger.error("label_map_full.json 이 존재하지 않습니다: %s", label_map_path)
        logger.error("STAGE 0 을 먼저 실행하세요: python scripts/0_split_data.py --run-name %s ...", run_name)
        sys.exit(1)

    # 제출 CSV 출력 경로
    submissions_dir = Path(paths_cfg.get("submissions_dir", "artifacts/submissions"))
    if not submissions_dir.is_absolute():
        submissions_dir = (repo_root / submissions_dir).resolve()
    submissions_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{run_name}_conf{conf}.csv"
    csv_path = submissions_dir / csv_filename

    # ── 3) label_map 로드 ──────────────────────────────────
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx2id = label_map.get("idx2id", {})
    valid_category_ids = set(int(v) for v in idx2id.values())
    nc = label_map.get("num_classes", 0)

    logger.info("label_map 로드 | num_classes=%d", nc)

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
    logger.info("추론 시작 | conf=%.3f | nms_iou=%.3f | max_det_per_image=%d",
                conf, nms_iou, max_det_per_image)

    detections = batch_predict(
        weights_path=best_pt,
        source=test_images_dir,
        conf=conf,
        iou=nms_iou,
        max_det=max_det_per_image * 5,  # Ultralytics 에는 여유있게 전달
        device=device,
        imgsz=imgsz,
        verbose=not args.quiet,
    )

    # ── 6) 후처리: Top-K + class→category_id ─────────────────
    rows = postprocess_detections(
        detections,
        idx2id=idx2id,
        max_det_per_image=max_det_per_image,
    )

    # ── 7) 제출 CSV 저장 ────────────────────────────────────
    write_submission(rows, csv_path)

    # ── 8) 제출 검증 ────────────────────────────────────────
    report = validate_submission(
        csv_path,
        max_det_per_image=max_det_per_image,
        valid_category_ids=valid_category_ids,
    )

    # ── 9) 매니페스트 저장 ──────────────────────────────────
    write_submission_manifest(
        report,
        run_dir=run_dir,
        conf=conf,
        n_test_images=n_test_images,
        csv_path=csv_path,
    )

    # ── 10) 요약 출력 ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 완료")
    logger.info("  run_name       : %s", run_name)
    logger.info("  best.pt        : %s", best_pt)
    logger.info("  test_images    : %d", n_test_images)
    logger.info("  conf_threshold : %.3f", conf)
    logger.info("  max_det/image  : %d", max_det_per_image)
    logger.info("  submission CSV : %s", csv_path)
    logger.info("  total rows     : %d", report["total_rows"])
    logger.info("  n_images       : %d", report["n_images"])
    logger.info("  validation     : %s", "PASS" if report["valid"] else "FAIL")
    if report["errors"]:
        logger.warning("  errors         : %d", len(report["errors"]))
    if report["warnings"]:
        logger.warning("  warnings       : %d", len(report["warnings"]))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
