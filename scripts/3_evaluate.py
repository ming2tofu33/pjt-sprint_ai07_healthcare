"""STAGE 3: 모델 평가.

STAGE 2 산출물(``best.pt``)과 YOLO 데이터셋(``data.yaml``)을 입력받아
Ultralytics validation 을 실행하고 메트릭을 저장한다.

사용법::

    python scripts/3_evaluate.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--device 0]
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
from src.utils.registry import update_run
from src.models.detector import PillDetector, save_metrics

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 3: 모델 평가")
    parser.add_argument("--run-name", required=True, help="실험 이름")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--device", default=None, help="GPU 디바이스 (예: 0, cpu)")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 억제")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 3 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    config, repo_root = load_experiment_config(config_path, script_path)

    # CLI --device 가 있으면 config 오버라이드
    if args.device is not None:
        config.setdefault("evaluate", {})["device"] = (
            int(args.device) if args.device.isdigit() else args.device
        )

    paths_cfg = config.get("paths", {})

    # ── 2) 경로 결정 ────────────────────────────────────────
    # runs/<run_name>/weights/best.pt
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        logger.error("best.pt 가 존재하지 않습니다: %s", best_pt)
        logger.error("STAGE 2 를 먼저 실행하세요: python scripts/2_train.py --run-name %s ...", run_name)
        sys.exit(1)

    # YOLO 데이터셋 data.yaml
    datasets_base = Path(paths_cfg.get("datasets_dir", "data/processed/datasets"))
    if not datasets_base.is_absolute():
        datasets_base = (repo_root / datasets_base).resolve()
    dataset_prefix = config.get("yolo_convert", {}).get("dataset_prefix", "pill_od_yolo")
    dataset_dir = datasets_base / f"{dataset_prefix}_{run_name}"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error("data.yaml 이 존재하지 않습니다: %s", data_yaml)
        logger.error("STAGE 1 을 먼저 실행하세요: python scripts/1_preprocess.py --run-name %s ...", run_name)
        sys.exit(1)

    # _registry.csv
    registry_path = runs_base / "_registry.csv"

    # ── 3) 모델 로드 ────────────────────────────────────────
    logger.info("모델 로드 | %s", best_pt)
    detector = PillDetector.from_weights(best_pt)

    # ── 4) 평가 실행 ────────────────────────────────────────
    eval_cfg = config.get("evaluate", {})
    logger.info("평가 시작 | conf=%.4f | iou=%.2f | device=%s",
                eval_cfg.get("conf", 0.001),
                eval_cfg.get("iou", 0.75),
                eval_cfg.get("device", "auto"))

    metrics = detector.validate(data_yaml=data_yaml, config=config)

    # ── 5) 메트릭 저장 ──────────────────────────────────────
    # 기존 metrics.json 이 있으면 병합 (학습 메트릭 유지 + eval 추가)
    metrics_path = run_dir / "metrics.json"
    existing_metrics: dict = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # eval 결과를 eval_ prefix 로 저장하여 학습 메트릭과 구분
    eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
    merged = {**existing_metrics, **eval_metrics}

    save_metrics(merged, run_dir, "metrics.json")
    logger.info("metrics.json 갱신 | eval_mAP50=%.4f | eval_mAP50-95=%.4f",
                metrics.get("mAP50", 0), metrics.get("mAP50_95", 0))

    # ── 6) registry 갱신 ────────────────────────────────────
    updated = update_run(
        registry_path,
        run_name,
        best_map50=f"{metrics.get('mAP50', 0):.4f}",
        best_map50_95=f"{metrics.get('mAP50_95', 0):.4f}",
        notes="eval_complete",
    )
    if updated:
        logger.info("_registry.csv 갱신 완료")
    else:
        logger.warning("_registry.csv 에서 run_name=%s 행을 찾지 못함 (STAGE 2 미실행?)", run_name)

    # ── 7) 요약 출력 ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 완료")
    logger.info("  run_name     : %s", run_name)
    logger.info("  best.pt      : %s", best_pt)
    logger.info("  data.yaml    : %s", data_yaml)
    logger.info("  eval_mAP50   : %.4f", metrics.get("mAP50", 0))
    logger.info("  eval_mAP50-95: %.4f", metrics.get("mAP50_95", 0))
    logger.info("  precision    : %.4f", metrics.get("precision", 0))
    logger.info("  recall       : %.4f", metrics.get("recall", 0))

    # 클래스별 AP (있으면)
    if "per_class_ap50" in metrics:
        n_classes = len(metrics["per_class_ap50"])
        logger.info("  per_class_ap50: %d classes", n_classes)
        if n_classes <= 20:
            for idx, ap in enumerate(metrics["per_class_ap50"]):
                logger.info("    class %d: %.4f", idx, ap)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
