"""STAGE 2: YOLO 모델 학습."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from shutil import copy2
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.detector import (  # noqa: E402
    PillDetector,
    copy_best_weights,
    extract_metrics,
    save_config_resolved,
    save_metrics,
)
from src.utils.config_loader import load_experiment_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.registry import append_run  # noqa: E402

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 2: YOLO 모델 학습")
    parser.add_argument("--run-name", required=True, help="실험 이름")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--device", default=None, help="GPU 디바이스 (예: 0, cpu)")
    parser.add_argument("--resume", action="store_true", help="last.pt에서 학습 재개")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="last.pt가 있으면 자동으로 재개 (없으면 새로 시작)",
    )
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 2 시작 | run_name=%s | config=%s", run_name, config_path)

    config, repo_root = load_experiment_config(config_path, script_path)

    if args.device is not None:
        config.setdefault("train", {})["device"] = (
            int(args.device) if args.device.isdigit() else args.device
        )

    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    datasets_base = Path(paths_cfg.get("datasets_dir", "data/processed/datasets"))
    if not datasets_base.is_absolute():
        datasets_base = (repo_root / datasets_base).resolve()
    dataset_prefix = config.get("yolo_convert", {}).get("dataset_prefix", "pill_od_yolo")
    dataset_dir = datasets_base / f"{dataset_prefix}_{run_name}"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error("data.yaml이 존재하지 않습니다: %s", data_yaml)
        logger.error("STAGE 1을 먼저 실행하세요: python scripts/1_preprocess.py --run-name %s ...", run_name)
        sys.exit(1)

    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_artifacts_dir = run_dir / "train"

    best_models_dir = Path(paths_cfg.get("best_models_dir", "artifacts/best_models"))
    if not best_models_dir.is_absolute():
        best_models_dir = (repo_root / best_models_dir).resolve()
    best_models_dir.mkdir(parents=True, exist_ok=True)

    registry_path = runs_base / "_registry.csv"

    save_config_resolved(config, run_dir)
    logger.info("config_resolved.yaml 저장 | %s", run_dir / "config_resolved.yaml")

    try:
        resume_enabled, resume_reason, last_pt = _resolve_resume_mode(
            run_dir,
            resume_flag=args.resume,
            auto_resume_flag=args.auto_resume,
        )
    except FileNotFoundError:
        logger.error("last.pt가 존재하지 않습니다: %s", run_dir / "weights" / "last.pt")
        logger.error("--resume 없이 처음부터 학습하세요.")
        sys.exit(1)

    if resume_enabled:
        logger.info("학습 재개 | resume=%s | last.pt=%s", resume_reason, last_pt)
        detector = PillDetector.from_weights(last_pt)
    else:
        if args.auto_resume:
            logger.info("학습 시작 | resume=off | checkpoint not found: %s", last_pt)
        else:
            logger.info("학습 시작 | resume=off")
        detector = PillDetector.from_config(config)
        logger.info("모델 로드 | %s", model_cfg.get("pretrained", "?"))

    logger.info(
        "학습 시작 | epochs=%s | imgsz=%s | batch=%s",
        train_cfg.get("epochs", "?"),
        train_cfg.get("imgsz", "?"),
        train_cfg.get("batch", "?"),
    )
    train_results = detector.train(
        data_yaml=data_yaml,
        project=str(run_dir),
        name="train",
        config=config,
    )

    train_output_dir = train_artifacts_dir
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
        logger.warning("best.pt를 찾을 수 없습니다. 학습이 정상 완료되었는지 확인하세요.")

    train_results_csv = train_artifacts_dir / "results.csv"
    metrics = extract_metrics(train_results, results_csv_path=train_results_csv)
    _copy_train_results_shortcut(train_results_csv, run_dir)
    snapshot_summary = _load_snapshot_summary(train_cfg, train_artifacts_dir)
    metrics["debug_snapshots"] = snapshot_summary
    logger.info(
        "debug snapshots | enabled=%s | saved_epochs=%d | saved_files=%d | dir=%s",
        snapshot_summary.get("enabled", False),
        len(snapshot_summary.get("saved_epochs", [])),
        int(snapshot_summary.get("saved_files", 0)),
        snapshot_summary.get("snapshot_dir", ""),
    )
    if metrics.get("mAP75_95") is None:
        logger.warning("train 결과에서 mAP75_95를 계산하지 못했습니다. all_ap 미제공.")

    competition_best_path, competition_report = _select_competition_best_weight(
        config=config,
        run_dir=run_dir,
        run_name=run_name,
        data_yaml=data_yaml,
        best_models_dir=best_models_dir,
    )
    metrics["competition_select"] = competition_report
    if competition_best_path is not None:
        metrics["competition_best_path"] = str(competition_best_path)

    save_metrics(metrics, run_dir, "metrics.json")

    map75_95 = metrics.get("mAP75_95")
    map75_95_str = f"{map75_95:.4f}" if map75_95 is not None else "N/A"
    logger.info(
        "metrics.json 저장 | mAP50=%.4f | mAP50-95=%.4f | mAP75-95=%s (대회지표)",
        metrics.get("mAP50", 0),
        metrics.get("mAP50_95", 0),
        map75_95_str,
    )

    selected_weight_for_registry = (
        competition_best_path if competition_best_path is not None else run_best
    )

    append_run(
        registry_path,
        run_name=run_name,
        model=model_cfg.get("architecture", ""),
        epochs=int(train_cfg.get("epochs", 0)),
        imgsz=int(train_cfg.get("imgsz", 0)),
        best_map50=metrics.get("mAP50"),
        best_map50_95=metrics.get("mAP50_95"),
        best_map75_95=metrics.get("mAP75_95"),
        weights_path=str(selected_weight_for_registry) if selected_weight_for_registry else "",
        config_path=str(config_path),
        notes="train_complete",
    )
    logger.info("_registry.csv 갱신 완료")

    logger.info("=" * 60)
    logger.info("STAGE 2 완료")
    logger.info("  run_name               : %s", run_name)
    logger.info("  run_dir                : %s", run_dir)
    logger.info("  train_artifacts_dir    : %s", train_artifacts_dir)
    logger.info("  best.pt                : %s", run_best or "N/A")
    logger.info("  competition_best.pt    : %s", competition_best_path or "N/A")
    logger.info("  mAP50                  : %.4f", metrics.get("mAP50", 0))
    logger.info("  mAP50-95               : %.4f", metrics.get("mAP50_95", 0))
    logger.info("  mAP75-95 (지표)        : %s", map75_95_str)
    logger.info("=" * 60)


def _select_competition_best_weight(
    *,
    config: dict,
    run_dir: Path,
    run_name: str,
    data_yaml: Path,
    best_models_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    train_cfg = config.get("train", {})
    comp_cfg = train_cfg.get("competition_select", {})
    if not isinstance(comp_cfg, dict):
        comp_cfg = {}

    enabled = bool(comp_cfg.get("enabled", True))
    candidate_tags = _normalize_competition_candidate_tags(comp_cfg.get("candidates", ["best", "last"]))
    output_name = str(comp_cfg.get("output_name", "competition_best.pt")).strip() or "competition_best.pt"
    output_name = Path(output_name).name
    use_tta = bool(comp_cfg.get("use_tta", False))

    report: dict[str, Any] = {
        "enabled": enabled,
        "candidate_tags": candidate_tags,
        "output_name": output_name,
        "use_tta": use_tta,
        "candidates": [],
        "selected": None,
        "status": "not_run",
    }

    if not enabled:
        report["status"] = "disabled"
        return None, report

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    project_dir = run_dir / "competition_select"
    project_dir.mkdir(parents=True, exist_ok=True)

    evaluated_candidates: list[dict[str, Any]] = []
    for tag in candidate_tags:
        weight_path = weights_dir / f"{tag}.pt"
        candidate_row: dict[str, Any] = {
            "tag": tag,
            "weight_path": str(weight_path),
            "exists": weight_path.exists(),
            "mAP75_95": None,
            "mAP50": None,
            "mAP50_95": None,
            "status": "missing",
        }
        if not weight_path.exists():
            report["candidates"].append(candidate_row)
            continue

        try:
            detector = PillDetector.from_weights(weight_path)
            metrics = detector.validate(
                data_yaml=data_yaml,
                config=config,
                eval_overrides={"augment": use_tta},
                project=project_dir,
                name=f"val_{tag}",
                exist_ok=True,
            )
            candidate_row["mAP75_95"] = metrics.get("mAP75_95")
            candidate_row["mAP50"] = metrics.get("mAP50")
            candidate_row["mAP50_95"] = metrics.get("mAP50_95")
            candidate_row["status"] = "ok"
            evaluated_candidates.append(candidate_row)
        except Exception as exc:
            candidate_row["status"] = "eval_failed"
            candidate_row["error"] = str(exc)
            logger.warning("competition_select 후보 평가 실패 | tag=%s | error=%s", tag, exc)

        report["candidates"].append(candidate_row)

    if not evaluated_candidates:
        report["status"] = "no_candidate"
        return None, report

    metric_ready = [c for c in evaluated_candidates if c.get("mAP75_95") is not None]
    if metric_ready:
        order_map = {tag: idx for idx, tag in enumerate(candidate_tags)}
        selected = sorted(
            metric_ready,
            key=lambda c: (-float(c["mAP75_95"]), order_map.get(str(c["tag"]), 999)),
        )[0]
        report["status"] = "selected_by_map75_95"
    else:
        order_map = {tag: idx for idx, tag in enumerate(candidate_tags)}
        selected = sorted(
            evaluated_candidates,
            key=lambda c: order_map.get(str(c["tag"]), 999),
        )[0]
        report["status"] = "selected_without_map75_95"
        logger.warning("competition_select: 모든 후보에서 mAP75_95를 계산하지 못해 순서 기반 선택을 사용합니다.")

    selected_src = Path(str(selected["weight_path"]))
    selected_dst = weights_dir / output_name
    if selected_src.resolve() != selected_dst.resolve():
        copy2(str(selected_src), str(selected_dst))

    artifact_path = best_models_dir / f"{run_name}_competition_best.pt"
    copy2(str(selected_dst), str(artifact_path))

    report["selected"] = {
        "tag": selected.get("tag"),
        "source_weight": str(selected_src),
        "output_weight": str(selected_dst),
        "artifact_weight": str(artifact_path),
        "mAP75_95": selected.get("mAP75_95"),
        "mAP50": selected.get("mAP50"),
        "mAP50_95": selected.get("mAP50_95"),
    }

    report_path = run_dir / "competition_best.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "competition_select 완료 | selected=%s | mAP75_95=%s | out=%s",
        selected.get("tag"),
        "N/A" if selected.get("mAP75_95") is None else f"{float(selected['mAP75_95']):.4f}",
        selected_dst,
    )
    return selected_dst, report


def _normalize_competition_candidate_tags(raw_value: Any) -> list[str]:
    allowed = {"best", "last"}
    if not isinstance(raw_value, list):
        return ["best", "last"]

    out: list[str] = []
    for item in raw_value:
        tag = str(item).strip().lower()
        if tag in allowed and tag not in out:
            out.append(tag)
    return out or ["best", "last"]


def _resolve_resume_mode(
    run_dir: Path,
    *,
    resume_flag: bool,
    auto_resume_flag: bool,
) -> tuple[bool, str, Path]:
    """STAGE 2 학습 재개 여부를 계산한다."""
    last_pt = run_dir / "weights" / "last.pt"

    if resume_flag:
        if not last_pt.exists():
            raise FileNotFoundError(last_pt)
        return True, "explicit", last_pt

    if auto_resume_flag and last_pt.exists():
        return True, "auto", last_pt

    return False, "off", last_pt


def _copy_train_results_shortcut(train_results_csv: Path, run_dir: Path) -> Path | None:
    if not train_results_csv.exists():
        logger.warning("train/results.csv가 없어 루트 shortcut 복사를 건너뜁니다: %s", train_results_csv)
        return None

    shortcut_path = run_dir / "results.csv"
    if train_results_csv.resolve() != shortcut_path.resolve():
        copy2(str(train_results_csv), str(shortcut_path))
    return shortcut_path


def _load_snapshot_summary(train_cfg: dict, train_artifacts_dir: Path) -> dict[str, Any]:
    debug_cfg = train_cfg.get("debug_snapshots", {})
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    try:
        max_epoch_gap = int(debug_cfg.get("max_epoch_gap", 20))
    except Exception:
        max_epoch_gap = 20

    summary: dict[str, Any] = {
        "enabled": bool(debug_cfg.get("enabled", True)),
        "scope": str(debug_cfg.get("scope", "val_only")),
        "percent_points": list(debug_cfg.get("percent_points", [10, 30, 50, 70, 90, 100])),
        "max_epoch_gap": max_epoch_gap,
        "saved_epochs": [],
        "saved_files": 0,
        "snapshot_dir": "",
        "index_file": "",
    }

    output_dirname = str(debug_cfg.get("output_dirname", "snapshots")).strip() or "snapshots"
    manifest_path = train_artifacts_dir / output_dirname / "snapshot_manifest.json"
    if not manifest_path.exists():
        summary["status"] = "manifest_not_found"
        return summary

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            summary.update(manifest)
            summary["status"] = "ok"
        else:
            summary["status"] = "invalid_manifest_type"
    except Exception as exc:
        summary["status"] = "manifest_load_failed"
        summary["error"] = str(exc)

    return summary


if __name__ == "__main__":
    main()
