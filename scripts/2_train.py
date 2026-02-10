"""STAGE 2: YOLO 紐⑤뜽 ?숈뒿.

STAGE 1 ?곗텧臾?``data.yaml``)怨??ㅽ뿕 config 瑜??낅젰諛쏆븘
Ultralytics YOLO ?숈뒿???ㅽ뻾?섍퀬 媛以묒튂/硫뷀듃由??덉??ㅽ듃由щ? 媛깆떊?쒕떎.

?ъ슜踰?:

    python scripts/2_train.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--device 0] [--resume] [--auto-resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ?? ?꾨줈?앺듃 猷⑦듃瑜?sys.path ??異붽? ????????????????????????????
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
    parser = argparse.ArgumentParser(description="STAGE 2: YOLO ?숈뒿")
    parser.add_argument("--run-name", required=True, help="?ㅽ뿕 ?대쫫")
    parser.add_argument("--config", required=True, help="?ㅽ뿕 config YAML 寃쎈줈")
    parser.add_argument("--device", default=None, help="GPU ?붾컮?댁뒪 (?? 0, cpu)")
    parser.add_argument("--resume", action="store_true", help="last.pt ?먯꽌 ?숈뒿 ?ш컻")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="last.pt가 있으면 자동으로 학습 재개 (없으면 새로 시작)",
    )
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 2 ?쒖옉 | run_name=%s | config=%s", run_name, config_path)

    # ?? 1) config 濡쒕뱶 ??????????????????????????????????????
    config, repo_root = load_experiment_config(config_path, script_path)

    # CLI --device 가 있으면 config 오버라이드
    if args.device is not None:
        config.setdefault("train", {})["device"] = (
            int(args.device) if args.device.isdigit() else args.device
        )

    paths_cfg = config.get("paths", {})

    # ?? 2) 寃쎈줈 寃곗젙 ????????????????????????????????????????
    # STAGE 1 산출물
    datasets_base = Path(paths_cfg.get("datasets_dir", "data/processed/datasets"))
    if not datasets_base.is_absolute():
        datasets_base = (repo_root / datasets_base).resolve()
    dataset_prefix = config.get("yolo_convert", {}).get("dataset_prefix", "pill_od_yolo")
    dataset_dir = datasets_base / f"{dataset_prefix}_{run_name}"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error("data.yaml ??議댁옱?섏? ?딆뒿?덈떎: %s", data_yaml)
        logger.error("STAGE 1 ??癒쇱? ?ㅽ뻾?섏꽭?? python scripts/1_preprocess.py --run-name %s ...", run_name)
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

    # ?? 3) config_resolved.yaml ???????????????????????????
    save_config_resolved(config, run_dir)
    logger.info("config_resolved.yaml ???| %s", run_dir / "config_resolved.yaml")

    # ?? 4) 紐⑤뜽 濡쒕뱶 ????????????????????????????????????????
    try:
        resume_enabled, resume_reason, last_pt = _resolve_resume_mode(
            run_dir,
            resume_flag=args.resume,
            auto_resume_flag=args.auto_resume,
        )
    except FileNotFoundError:
        logger.error("last.pt 가 존재하지 않습니다: %s", run_dir / "weights" / "last.pt")
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
        logger.info("모델 로드 | %s", config.get("model", {}).get("pretrained", "?"))

    # ?? 5) ?숈뒿 ?ㅽ뻾 ????????????????????????????????????????
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    logger.info("?숈뒿 ?쒖옉 | epochs=%s | imgsz=%s | batch=%s",
                train_cfg.get("epochs", "?"),
                train_cfg.get("imgsz", "?"),
                train_cfg.get("batch", "?"))

    # Ultralytics ??project/name ?섏쐞??寃곌낵瑜???ν븳??
    # project=runs_base, name=run_name ?쇰줈 吏?뺥븯硫?    # runs/<run_name>/ ????λ맂??
    train_results = detector.train(
        data_yaml=data_yaml,
        project=str(runs_base),
        name=run_name,
        config=config,
    )

    # ?? 6) ?숈뒿 寃곌낵 ?꾩쿂由???????????????????????????????????
    # Ultralytics 媛 ?ㅼ젣 ??ν븳 ?붾젆?곕━ (run_dir ? ?숈씪?????덉쓬)
    train_output_dir = run_dir

    # best.pt 蹂듭궗
    run_best, artifact_best = copy_best_weights(
        train_output_dir,
        run_dir=run_dir,
        best_models_dir=best_models_dir,
        run_name=run_name,
    )

    if run_best:
        logger.info("best.pt 蹂듭궗 ?꾨즺 | %s", run_best)
        logger.info("artifact 蹂듭궗 ?꾨즺 | %s", artifact_best)
    else:
        logger.warning("best.pt 瑜?李얠쓣 ???놁뒿?덈떎. ?숈뒿???뺤긽 ?꾨즺?섏뿀?붿? ?뺤씤?섏꽭??")

    # ?? 7) 硫뷀듃由?異붿텧 + ???????????????????????????????????
    metrics = _extract_train_metrics(train_results, run_dir)
    save_metrics(metrics, run_dir, "metrics.json")

    map75_95 = metrics.get("mAP75_95")
    map75_95_str = f"{map75_95:.4f}" if map75_95 is not None else "N/A"
    logger.info("metrics.json ???| mAP50=%.4f | mAP50-95=%.4f | mAP75-95=%s (??뚯???",
                metrics.get("mAP50", 0), metrics.get("mAP50_95", 0), map75_95_str)

    # ?? 8) registry 媛깆떊 ????????????????????????????????????
    append_run(
        registry_path,
        run_name=run_name,
        model=model_cfg.get("architecture", ""),
        epochs=int(train_cfg.get("epochs", 0)),
        imgsz=int(train_cfg.get("imgsz", 0)),
        best_map50=metrics.get("mAP50"),
        best_map50_95=metrics.get("mAP50_95"),
        best_map75_95=metrics.get("mAP75_95"),
        weights_path=str(run_best) if run_best else "",
        config_path=str(config_path),
        notes="train_complete",
    )
    logger.info("_registry.csv 媛깆떊 ?꾨즺")

    # ?? 9) ?붿빟 異쒕젰 ????????????????????????????????????????
    logger.info("=" * 60)
    logger.info("STAGE 2 ?꾨즺")
    logger.info("  run_name       : %s", run_name)
    logger.info("  run_dir        : %s", run_dir)
    logger.info("  best.pt        : %s", run_best or "N/A")
    logger.info("  mAP50          : %.4f", metrics.get("mAP50", 0))
    logger.info("  mAP50-95       : %.4f", metrics.get("mAP50_95", 0))
    logger.info("  mAP75-95 (???: %s", map75_95_str)
    logger.info("=" * 60)


def _extract_train_metrics(train_results: object, run_dir: Path) -> dict:
    """?숈뒿 寃곌낵 媛앹껜 ?먮뒗 results.csv ?먯꽌 理쒖쥌 硫뷀듃由?쓣 異붿텧?쒕떎.

    ????됯? 吏??``mAP@[0.75:0.95]`` 瑜?``mAP75_95`` ?ㅻ줈 異붿텧?쒕떎.
    """
    metrics: dict = {}

    # 諛⑸쾿 1: Ultralytics results 媛앹껜?먯꽌 吏곸젒 異붿텧
    try:
        box = train_results.box
        metrics["mAP50"] = float(box.map50)
        metrics["mAP50_95"] = float(box.map)
        metrics["precision"] = float(box.mp)
        metrics["recall"] = float(box.mr)

        # ???吏?? mAP@[0.75:0.95]
        all_ap = None
        if hasattr(box, "all_ap") and box.all_ap is not None:
            all_ap = np.array(box.all_ap)
        elif hasattr(box, "ap") and box.ap is not None:
            ap_arr = np.array(box.ap)
            if ap_arr.ndim == 2 and ap_arr.shape[1] == 10:
                all_ap = ap_arr

        if all_ap is not None and all_ap.ndim == 2 and all_ap.shape[1] >= 10:
            metrics["mAP75_95"] = float(all_ap[:, 5:].mean())
        else:
            metrics["mAP75_95"] = None

        return metrics
    except Exception:
        pass

    # 諛⑸쾿 2: results_dict ?먯꽌 異붿텧
    try:
        rd = train_results.results_dict
        metrics["mAP50"] = float(rd.get("metrics/mAP50(B)", 0))
        metrics["mAP50_95"] = float(rd.get("metrics/mAP50-95(B)", 0))
        metrics["precision"] = float(rd.get("metrics/precision(B)", 0))
        metrics["recall"] = float(rd.get("metrics/recall(B)", 0))
        metrics["mAP75_95"] = None  # results_dict ?먯꽌??怨꾩궛 遺덇?
        return metrics
    except Exception:
        pass

    # 諛⑸쾿 3: results.csv ?뚯떛 (留덉?留???
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
                metrics["mAP75_95"] = None  # CSV ?먯꽌??怨꾩궛 遺덇?
        except Exception:
            pass

    return metrics




def _resolve_resume_mode(
    run_dir: Path,
    *,
    resume_flag: bool,
    auto_resume_flag: bool,
) -> tuple[bool, str, Path]:
    """STAGE 2 학습 재개 여부를 계산한다.

    Returns
    -------
    tuple[bool, str, Path]
        (resume_enabled, resume_reason, last_pt)
        resume_reason은 ``explicit`` | ``auto`` | ``off``.
    """
    last_pt = run_dir / "weights" / "last.pt"

    if resume_flag:
        if not last_pt.exists():
            raise FileNotFoundError(last_pt)
        return True, "explicit", last_pt

    if auto_resume_flag and last_pt.exists():
        return True, "auto", last_pt

    return False, "off", last_pt

if __name__ == "__main__":
    main()

