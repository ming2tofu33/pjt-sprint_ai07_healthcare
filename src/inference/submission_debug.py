"""STAGE 4 제출 전 sanity check 시각화 유틸리티."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.visualizer import visualize_batch

logger = get_logger(__name__)


def sample_detections_for_debug(
    detections: list[dict],
    sample_size: int,
    seed: int,
) -> list[dict]:
    """디버그 시각화용 샘플 detection을 안정적으로 선택한다."""
    ordered = sorted(detections, key=lambda d: str(d.get("image_stem", "")))
    if sample_size <= 0 or not ordered:
        return []
    if sample_size >= len(ordered):
        return ordered

    rng = random.Random(seed)
    picked_indices = sorted(rng.sample(range(len(ordered)), sample_size))
    return [ordered[i] for i in picked_indices]


def save_submission_debug_images(
    *,
    run_dir: Path,
    detections: list[dict],
    idx2name: dict[int, str] | None,
    max_det_per_image: int,
    sample_size: int = 12,
    seed: int = 42,
    enabled: bool = True,
) -> dict[str, Any]:
    """제출 전 sanity check 이미지를 저장하고 결과 리포트를 반환한다."""
    output_dir = run_dir / "submission_debug"
    requested = max(0, int(sample_size))
    report: dict[str, Any] = {
        "debug_enabled": bool(enabled),
        "debug_output_dir": str(output_dir),
        "debug_sample_requested": requested,
        "debug_sample_saved": 0,
    }

    if not enabled:
        report["debug_skipped_reason"] = "disabled_by_config"
        return report

    # 재실행 시 이전 결과가 섞이지 않도록 항상 초기화한다.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = sample_detections_for_debug(detections, requested, seed)
    if not samples:
        report["debug_skipped_reason"] = "no_samples"
        return report

    try:
        saved_paths = visualize_batch(
            samples,
            output_dir,
            idx2name=idx2name,
            max_images=len(samples),
            show_score=True,
            top_k=max_det_per_image,
            highlight_overflow=True,
            show_rank=True,
        )
    except Exception as exc:  # noqa: BLE001 - 시각화 실패는 STAGE4를 중단시키지 않는다.
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("submission debug 시각화 실패: %s", reason)
        report["debug_skipped_reason"] = reason
        return report

    report["debug_sample_saved"] = len(saved_paths)
    if len(saved_paths) < len(samples):
        report["debug_skipped_reason"] = "partial_saved"

    return report

