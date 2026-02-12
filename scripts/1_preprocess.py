"""STAGE 1: df_clean.csv + splits.csv → YOLO 데이터셋 변환.

STAGE 0 산출물(``df_clean.csv``, ``splits.csv``)과 원본 이미지를 입력받아
Ultralytics YOLO 학습용 디렉터리 구조를 생성한다.

내부적으로 ``export_yolo.run_export()`` 를 호출하여
hardlink-first, external 이미지 검색, split conflict 감지,
clamping 통계, critical missing threshold 등의 기능을 사용한다.

사용법::

    python scripts/1_preprocess.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--copy]
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
from src.dataprep.output.export_yolo import run_export, verify_labels

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 1: df_clean → YOLO 변환")
    parser.add_argument("--run-name", required=True, help="STAGE 0 에서 사용한 run_name")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--copy", action="store_true", default=False,
                        help="이미지를 복사 (기본: hardlink)")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 1 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    config, repo_root = load_experiment_config(config_path, script_path)

    # ── 2) 경로 결정 ────────────────────────────────────────
    paths_cfg = config.get("paths", {})
    yolo_cfg = config.get("yolo_convert", {})

    # metadata 디렉터리 (splits.csv 위치)
    metadata_dir = Path(paths_cfg.get("metadata_dir", "data/metadata"))
    if not metadata_dir.is_absolute():
        metadata_dir = (repo_root / metadata_dir).resolve()

    # cache 디렉터리 (df_clean.csv 위치 — STAGE 0 이 여기에 저장)
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()
    cache_dir = processed_base / run_name

    # 원본 이미지 디렉터리
    train_images_dir = Path(paths_cfg.get("train_images_dir", "data/raw/train_images"))
    if not train_images_dir.is_absolute():
        train_images_dir = (repo_root / train_images_dir).resolve()

    # external 이미지 디렉터리 (선택)
    external_images_str = paths_cfg.get("external_images_dir", "data/raw/external/combined/images")
    external_images_dir = Path(external_images_str)
    if not external_images_dir.is_absolute():
        external_images_dir = (repo_root / external_images_dir).resolve()

    # YOLO 데이터셋 출력 디렉터리
    datasets_base = Path(paths_cfg.get("datasets_dir", "data/processed/datasets"))
    if not datasets_base.is_absolute():
        datasets_base = (repo_root / datasets_base).resolve()
    dataset_prefix = yolo_cfg.get("dataset_prefix", "pill_od_yolo")
    output_dir = datasets_base / f"{dataset_prefix}_{run_name}"

    # class_map.csv 출력 경로
    class_map_path = metadata_dir / "class_map.csv"

    # ── 3) STAGE 0 산출물 검증 ───────────────────────────────
    df_clean_name = config.get("outputs", {}).get("df_clean_name", "df_clean.csv")
    splits_name = config.get("split", {}).get("splits_name", "splits.csv")

    # df_clean.csv: cache/<run_name>/ 에서 찾음
    df_path = cache_dir / df_clean_name
    # splits.csv: metadata/ 에서 찾음
    splits_path = metadata_dir / splits_name

    for p, label in [(df_path, df_clean_name), (splits_path, splits_name)]:
        if not p.exists():
            logger.error("%s 이 존재하지 않습니다: %s", label, p)
            logger.error("STAGE 0 을 먼저 실행하세요: python scripts/0_split_data.py --run-name %s ...", run_name)
            sys.exit(1)

    logger.info("STAGE 0 산출물 확인 완료")
    logger.info("  df_clean  : %s", df_path)
    logger.info("  splits    : %s", splits_path)

    # ── 4) link_mode 결정 ─────────────────────────────────────
    if args.copy:
        link_mode = "copy"
    else:
        link_mode = yolo_cfg.get("link_mode", "hardlink")

    allow_fallback = yolo_cfg.get("allow_fallback", True)
    critical_missing_ratio = yolo_cfg.get("critical_missing_ratio", 0.02)
    critical_missing_count = yolo_cfg.get("critical_missing_count", 20)

    # ── 5) YOLO 변환 실행 ─────────────────────────────────────
    logger.info("YOLO 변환 시작 | output=%s | link_mode=%s", output_dir, link_mode)

    try:
        result = run_export(
            df_path=df_path,
            splits_path=splits_path,
            out_dir=output_dir,
            train_images_dir=train_images_dir,
            external_images_dir=external_images_dir,
            class_map_path=class_map_path,
            link_mode=link_mode,
            allow_fallback=allow_fallback,
            critical_missing_ratio=critical_missing_ratio,
            critical_missing_count=critical_missing_count,
            repo_root=repo_root,
            progress=args.verbose,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("YOLO 변환 실패: %s", e)
        sys.exit(1)

    image_counts = result["image_counts"]
    nc = result["nc"]

    logger.info("변환 완료 | train_images=%d | val_images=%d",
                image_counts["train"], image_counts["val"])
    logger.info("  nc=%d | class_index range=[%s, %s]",
                nc, result["all_min_cls"], result["all_max_cls"])

    if result["missing_image_count"] > 0:
        logger.warning("누락 이미지 %d 건 (threshold=%d)",
                       result["missing_image_count"], result["missing_critical_threshold"])
    if result["split_conflicts"]:
        logger.warning("split conflicts: %d (첫 번째 할당 유지)", result["split_conflicts"])
    if result["invalid_bbox_rows"] > 0:
        logger.warning("invalid bbox rows: %d", result["invalid_bbox_rows"])
    if result["clamped_rows"] > 0:
        logger.warning("clamped bbox rows: %d", result["clamped_rows"])
    if result["empty_label_files"] > 0:
        logger.warning("빈 label 파일: %d", result["empty_label_files"])

    logger.info("  link stats: hardlink=%d / copy=%d",
                result["hardlink_count"], result["copy_count"])

    if result["class_range_error"]:
        logger.error("class_index 범위 이상! labels 에서 [0, %d) 밖의 인덱스 발견", nc)
        sys.exit(1)

    if result["critical"]:
        logger.error("누락 이미지가 임계값 초과 — critical failure")
        sys.exit(1)

    # ── 6) label 검증 (선택) ─────────────────────────────────
    do_verify = yolo_cfg.get("verify_labels", True)
    if do_verify:
        logger.info("label 검증 중 ...")
        vresult = verify_labels(output_dir, nc=nc, progress=args.verbose)
        logger.info("  검증: files=%d, lines=%d, errors=%d",
                    vresult["total_files"], vresult["total_lines"], len(vresult["errors"]))
        if vresult["errors"]:
            for err in vresult["errors"][:20]:
                logger.warning("  [VERIFY] %s", err)
            if len(vresult["errors"]) > 20:
                logger.warning("  ... 외 %d 건", len(vresult["errors"]) - 20)

            # 에러 비율이 임계값을 초과하면 파이프라인 중단
            n_errors = len(vresult["errors"])
            n_total = max(vresult["total_files"], 1)
            error_ratio = n_errors / n_total
            critical_ratio = float(yolo_cfg.get("critical_missing_ratio", 0.02))
            if error_ratio > critical_ratio:
                logger.error(
                    "label 검증 에러 비율(%.1f%%)이 임계값(%.1f%%)을 초과합니다.",
                    error_ratio * 100, critical_ratio * 100,
                )
                sys.exit(1)
            else:
                logger.warning(
                    "label 검증 에러 %d건 (%.1f%%) — 임계값 이하이므로 계속 진행합니다.",
                    n_errors, error_ratio * 100,
                )

    # ── 7) 요약 출력 ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 완료")
    logger.info("  run_name     : %s", run_name)
    logger.info("  data.yaml    : %s", result["data_yaml"])
    logger.info("  output_dir   : %s", output_dir)
    logger.info("  train images : %d", image_counts["train"])
    logger.info("  val images   : %d", image_counts["val"])
    logger.info("  nc           : %d", nc)
    logger.info("  class_map    : %s", class_map_path)
    logger.info("=" * 60)
    


if __name__ == "__main__":
    main()
