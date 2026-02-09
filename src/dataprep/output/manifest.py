from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def write_manifest(
    *,
    records: list[dict],
    logs: dict[str, list[dict]],
    config_path: Path,
    metadata_dir: Path,
    outputs_cfg: dict,
    repo_root: Path,
) -> None:
    """
    재현성과 추적성을 위한 실행 manifest를 저장한다.

    manifest 포함 항목:
    - 생성 시각
    - git 커밋 해시(가능한 경우)
    - config 경로 및 config sha256
    - 실행 비교를 위한 행 단위 요약 통계
    """
    manifest_name = outputs_cfg.get("manifest_name", "preprocess_manifest.json")

    # config 바이트 해시를 남겨 두 실행의 설정 내용이 정확히 같은지 비교한다.
    cfg_bytes = config_path.read_bytes()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        git_hash = None

    # 실행 진단을 위한 경량 요약 통계를 집계한다.
    total_rows = len(records)
    train_rows = sum(1 for r in records if r["source"] == "train")
    external_rows = sum(1 for r in records if r["source"] == "external")
    unique_images = len({r["file_name"] for r in records})

    per_image = Counter(r["file_name"] for r in records)
    per_image_dist = Counter(per_image.values())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "config_path": str(config_path),
        "config_sha256": cfg_hash,
        "summary": {
            "total_rows": total_rows,
            "train_rows": train_rows,
            "external_rows": external_rows,
            "excluded_rows": len(logs["excluded_rows"]),
            "fixes_bbox": len(logs["fixes_bbox"]),
            "unique_images": unique_images,
            "objects_per_image_dist": dict(per_image_dist),
        },
    }
    with (metadata_dir / manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
