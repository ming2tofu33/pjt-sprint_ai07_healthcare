#!/usr/bin/env python
"""STAGE 2: YOLO 모델 학습.

STAGE 1 산출물(``data.yaml``)과 실험 config 를 입력받아
Ultralytics YOLO 학습을 실행하고 가중치/메트릭/레지스트리를 갱신한다.

사용법::

    python scripts/2_train.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--device 0] [--resume]
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
from src.utils.registry import append_run, update_run
from src.models.detector import (
    PillDetector,
    save_config_resolved,
    save_metrics,
    copy_best_weights,
)

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 2: YOLO 학습")
    parser.add_argument("--run-name", required=True, help="실험 이름")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--device", default=None, help="GPU 디바이스 (예: 0, cpu)")
    parser.add_argument("--resume", action="store_true", help="last.pt 에서 학습 재개")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 억제")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 2 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    config, repo_root = load_experiment_config(config_path, script_path)

    # CLI --device 가 있으면 config 오버라이드
    if args.device is not None:
        config.setdefault("train", {})["device"] = (
            int(args.device) if args.device.isdigit() else args.device
        )

    paths_cfg = config.get("paths", {})

    # ── 2) 경로 결정 ────────────────────────────────────────
    # STAGE 1 산출물
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

    # runs/<run_name>/
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # artifacts/best_models/
    best_models_dir = Path(paths_cfg.get("best_models_dir", "artifacts/best_models"))
    if not best_models_dir.is_absolute():
        best_models_dir = (repo_root / best_models_dir).resolve()

    # _registry.csv
    registry_path = runs_base / "_registry.csv"

    # ── 3) config_resolved.yaml 저장 ────────────────────────
    save_config_resolved(config, run_dir)
    logger.info("config_resolved.yaml 저장 | %s", run_dir / "config_resolved.yaml")

    # ── 4) 모델 로드 ────────────────────────────────────────
    if args.resume:
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            logger.error("last.pt 가 존재하지 않습니다: %s", last_pt)
            logger.error("--resume 없이 처음부터 학습하세요.")
            sys.exit(1)
        logger.info("학습 재개 | last.pt=%s", last_pt)
        detector = PillDetector.from_weights(last_pt)
    else:
        detector = PillDetector.from_config(config)
        logger.info("모델 로드 | %s", config.get("model", {}).get("pretrained", "?"))

    # ── 5) 학습 실행 ────────────────────────────────────────
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    logger.info("학습 시작 | epochs=%s | imgsz=%s | batch=%s",
                train_cfg.get("epochs", "?"),
                train_cfg.get("imgsz", "?"),
                train_cfg.get("batch", "?"))

    # Ultralytics 는 project/name 하위에 결과를 저장한다.
    # project=runs_base, name=run_name 으로 지정하면
    # runs/<run_name>/ 에 저장된다.
    train_results = detector.train(
        data_yaml=data_yaml,
        project=str(runs_base),
        name=run_name,
        config=config,
    )

    # ── 6) 학습 결과 후처리 ──────────────────────────────────
    # Ultralytics 가 실제 저장한 디렉터리 (run_dir 와 동일할 수 있음)
    train_output_dir = run_dir

    # best.pt 복사
    run_best, artifact_best = copy_best_weights(
        train_output_dir,
        run_dir=run_dir,
        best_models_dir=best_models_dir,
        run_name=run_name,
    )

    if run_best:
        logger.info("best.pt 복사 완료 | %s", run_best)
        logger.info("artifact 복사 완료 | %s", artifact_best)
    else:
        logger.warning("best.pt 를 찾을 수 없습니다. 학습이 정상 완료되었는지 확인하세요.")

    # ── 7) 메트릭 추출 + 저장 ────────────────────────────────
    metrics = _extract_train_metrics(train_results, run_dir)
    save_metrics(metrics, run_dir, "metrics.json")
    logger.info("metrics.json 저장 | mAP50=%.4f | mAP50-95=%.4f",
                metrics.get("mAP50", 0), metrics.get("mAP50_95", 0))

    # ── 8) registry 갱신 ────────────────────────────────────
    append_run(
        registry_path,
        run_name=run_name,
        model=model_cfg.get("architecture", ""),
        epochs=int(train_cfg.get("epochs", 0)),
        imgsz=int(train_cfg.get("imgsz", 0)),
        best_map50=metrics.get("mAP50"),
        best_map50_95=metrics.get("mAP50_95"),
        weights_path=str(run_best) if run_best else "",
        config_path=str(config_path),
        notes="train_complete",
    )
    logger.info("_registry.csv 갱신 완료")

    # ── 9) 요약 출력 ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 완료")
    logger.info("  run_name     : %s", run_name)
    logger.info("  run_dir      : %s", run_dir)
    logger.info("  best.pt      : %s", run_best or "N/A")
    logger.info("  mAP50        : %.4f", metrics.get("mAP50", 0))
    logger.info("  mAP50-95     : %.4f", metrics.get("mAP50_95", 0))
    logger.info("=" * 60)


def _extract_train_metrics(train_results: object, run_dir: Path) -> dict:
    """학습 결과 객체 또는 results.csv 에서 최종 메트릭을 추출한다."""
    metrics: dict = {}

    # 방법 1: Ultralytics results 객체에서 직접 추출
    try:
        box = train_results.box
        metrics["mAP50"] = float(box.map50)
        metrics["mAP50_95"] = float(box.map)
        metrics["precision"] = float(box.mp)
        metrics["recall"] = float(box.mr)
        return metrics
    except Exception:
        pass

    # 방법 2: results_dict 에서 추출
    try:
        rd = train_results.results_dict
        metrics["mAP50"] = float(rd.get("metrics/mAP50(B)", 0))
        metrics["mAP50_95"] = float(rd.get("metrics/mAP50-95(B)", 0))
        metrics["precision"] = float(rd.get("metrics/precision(B)", 0))
        metrics["recall"] = float(rd.get("metrics/recall(B)", 0))
        return metrics
    except Exception:
        pass

    # 방법 3: results.csv 파싱 (마지막 행)
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        try:
            import csv
            with results_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                for col in last:
                    col_stripped = col.strip()
                    if "mAP50(B)" in col_stripped and "95" not in col_stripped:
                        metrics["mAP50"] = float(last[col])
                    elif "mAP50-95(B)" in col_stripped:
                        metrics["mAP50_95"] = float(last[col])
                    elif "precision(B)" in col_stripped:
                        metrics["precision"] = float(last[col])
                    elif "recall(B)" in col_stripped:
                        metrics["recall"] = float(last[col])
        except Exception:
            pass

    return metrics


if __name__ == "__main__":
    main()
