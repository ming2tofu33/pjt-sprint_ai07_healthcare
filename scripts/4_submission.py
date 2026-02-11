"""STAGE 4: ?쒖텧 ?뚯씪 ?앹꽦.

STAGE 2 ?곗텧臾?``competition_best.pt`` ?곗꽑, ?놁쑝硫?``best.pt``)怨??뚯뒪???대?吏,
label_map ???ъ슜?섏뿬 異붾줎 ???꾩쿂由?Top-4) ???쒖텧 CSV ?앹꽦 ??寃利앷퉴吏 ?섑뻾?쒕떎.

?ъ슜踰?:

    python scripts/4_submission.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml [--conf 0.25] [--device 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from shutil import copy2

# ?? ?꾨줈?앺듃 猷⑦듃瑜?sys.path ??異붽? ????????????????????????????
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 4: ?쒖텧 ?뚯씪 ?앹꽦")
    parser.add_argument("--run-name", required=True, help="?ㅽ뿕 ?대쫫")
    parser.add_argument("--config", required=True, help="?ㅽ뿕 config YAML 寃쎈줈")
    parser.add_argument("--conf", type=float, default=None,
                        help="confidence threshold (湲곕낯: config ??submission.conf)")
    parser.add_argument("--device", default=None, help="GPU ?붾컮?댁뒪 (?? 0, cpu)")
    parser.add_argument("--verbose", action="store_true", help="?곸꽭 濡쒓렇 異쒕젰")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 4 ?쒖옉 | run_name=%s | config=%s", run_name, config_path)

    # ?? 1) config 濡쒕뱶 ??????????????????????????????????????
    config, repo_root = load_experiment_config(config_path, script_path)

    paths_cfg = config.get("paths", {})
    sub_cfg = config.get("submission", {})
    debug_cfg = sub_cfg.get("debug", {}) if isinstance(sub_cfg.get("debug", {}), dict) else {}
    train_cfg = config.get("train", {})

    # confidence threshold: CLI > config
    conf = args.conf if args.conf is not None else float(sub_cfg.get("conf", 0.25))
    nms_iou = float(sub_cfg.get("nms_iou", 0.5))
    max_det_per_image = int(sub_cfg.get("max_det_per_image", 4))
    augment = bool(sub_cfg.get("augment", False))
    debug_enabled = bool(debug_cfg.get("enabled", True))
    debug_sample_size = int(debug_cfg.get("sample_size", 12))
    debug_seed = int(debug_cfg.get("seed", 42))
    device = args.device
    if device is None:
        device = sub_cfg.get("device")
    if device is not None and isinstance(device, str) and device.isdigit():
        device = int(device)
    imgsz = int(train_cfg.get("imgsz", 640))

    # ?? 2) 寃쎈줈 寃곗젙 ????????????????????????????????????????
    # runs/<run_name>/weights/{competition_best.pt|best.pt}
    runs_base = Path(paths_cfg.get("runs_dir", "runs"))
    if not runs_base.is_absolute():
        runs_base = (repo_root / runs_base).resolve()
    run_dir = runs_base / run_name
    submit_dir = run_dir / "submit"
    submit_debug_dir = submit_dir / "debug"
    submit_manifest_path = submit_dir / "submission_manifest.json"
    root_manifest_shortcut = run_dir / "submission_manifest.json"
    submit_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    competition_pt = weights_dir / "competition_best.pt"
    best_pt = weights_dir / "best.pt"
    selected_weight = competition_pt if competition_pt.exists() else best_pt
    if not selected_weight.exists():
        logger.error("媛以묒튂 ?뚯씪??議댁옱?섏? ?딆뒿?덈떎: %s", selected_weight)
        logger.error("?뺤씤 寃쎈줈: competition=%s | best=%s", competition_pt, best_pt)
        logger.error("STAGE 2 瑜?癒쇱? ?ㅽ뻾?섏꽭?? python scripts/2_train.py --run-name %s ...", run_name)
        sys.exit(1)

    # ?뚯뒪???대?吏 ?붾젆?곕━
    test_images_dir = Path(paths_cfg.get("test_images_dir", "data/raw/test_images"))
    if not test_images_dir.is_absolute():
        test_images_dir = (repo_root / test_images_dir).resolve()

    if not test_images_dir.exists():
        logger.error("?뚯뒪???대?吏 ?붾젆?곕━媛 議댁옱?섏? ?딆뒿?덈떎: %s", test_images_dir)
        sys.exit(1)

    # label_map_full.json (idx2id 留ㅽ븨 ?꾩슂)
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()
    cache_dir = processed_base / run_name

    label_map_path = cache_dir / "label_map_full.json"
    if not label_map_path.exists():
        logger.error("label_map_full.json ??議댁옱?섏? ?딆뒿?덈떎: %s", label_map_path)
        logger.error("STAGE 0 ??癒쇱? ?ㅽ뻾?섏꽭?? python scripts/0_split_data.py --run-name %s ...", run_name)
        sys.exit(1)

    # ?쒖텧 CSV 異쒕젰 寃쎈줈
    submissions_dir = Path(paths_cfg.get("submissions_dir", "artifacts/submissions"))
    if not submissions_dir.is_absolute():
        submissions_dir = (repo_root / submissions_dir).resolve()
    submissions_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{run_name}_conf{conf}.csv"
    csv_path = submissions_dir / csv_filename

    # ?? 3) label_map 濡쒕뱶 ??????????????????????????????????
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx2id = label_map.get("idx2id", {})
    names = label_map.get("names", [])
    idx2name = {
        i: str(name) for i, name in enumerate(names)
        if isinstance(name, (str, int, float))
    }
    valid_category_ids = set(int(v) for v in idx2id.values())
    nc = label_map.get("num_classes", 0)

    logger.info("label_map 濡쒕뱶 | num_classes=%d", nc)

    # ?? 4) ?뚯뒪???대?吏 ?ㅼ틪 ??????????????????????????????
    test_images = sorted(test_images_dir.glob("*.png")) + \
                  sorted(test_images_dir.glob("*.jpg")) + \
                  sorted(test_images_dir.glob("*.jpeg"))
    n_test_images = len(test_images)

    if n_test_images == 0:
        logger.error("?뚯뒪???대?吏媛 ?놁뒿?덈떎: %s", test_images_dir)
        sys.exit(1)

    logger.info("?뚯뒪???대?吏 %d ??諛쒓껄", n_test_images)

    # ?? 5) 諛곗튂 異붾줎 ?????????????????????????????????????????
    logger.info("異붾줎 ?쒖옉 | conf=%.3f | nms_iou=%.3f | max_det_per_image=%d | augment(TTA)=%s",
                conf, nms_iou, max_det_per_image, augment)

    logger.info("?쒖텧 媛以묒튂 ?좏깮 | selected=%s | fallback=%s", selected_weight, best_pt)
    detections = batch_predict(
        weights_path=selected_weight,
        source=test_images_dir,
        conf=conf,
        iou=nms_iou,
        max_det=max_det_per_image * 5,  # Ultralytics ?먮뒗 ?ъ쑀?덇쾶 ?꾨떖
        device=device,
        imgsz=imgsz,
        verbose=args.verbose,
        augment=augment,
    )

    # ?? 6) ?쒖텧 ???쒓컖 sanity check ???(?ㅽ뙣?대룄 怨꾩냽 吏꾪뻾) ???????
    debug_report = save_submission_debug_images(
        run_dir=run_dir,
        output_dir=submit_debug_dir,
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

    # ?? 6) ?꾩쿂由? Top-K + class?뭖ategory_id ?????????????????
    rows = postprocess_detections(
        detections,
        idx2id=idx2id,
        max_det_per_image=max_det_per_image,
    )

    # ?? 7) ?쒖텧 CSV ???????????????????????????????????????
    write_submission(rows, csv_path)
    logger.info("????쒖텧 ?щ㎎(8而щ읆) CSV ?앹꽦 ?꾨즺")

    # ?? 8) ?쒖텧 寃利?????????????????????????????????????????
    report = validate_submission(
        csv_path,
        max_det_per_image=max_det_per_image,
        valid_category_ids=valid_category_ids,
    )

    # ?? 9) 留ㅻ땲?섏뒪???????????????????????????????????????
    manifest_path = write_submission_manifest(
        report,
        run_dir=run_dir,
        conf=conf,
        n_test_images=n_test_images,
        csv_path=csv_path,
        debug_report=debug_report,
        output_path=submit_manifest_path,
    )

    if manifest_path.resolve() != root_manifest_shortcut.resolve():
        copy2(str(manifest_path), str(root_manifest_shortcut))

    # ?? 10) ?붿빟 異쒕젰 ???????????????????????????????????????
    logger.info("=" * 60)
    logger.info("STAGE 4 ?꾨즺")
    logger.info("  run_name       : %s", run_name)
    logger.info("  weight_file    : %s", selected_weight)
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
