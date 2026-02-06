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
    manifest_name = outputs_cfg.get("manifest_name", "preprocess_manifest.json")

    cfg_bytes = config_path.read_bytes()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        git_hash = None

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
