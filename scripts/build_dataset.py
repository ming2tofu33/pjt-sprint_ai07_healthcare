from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataprep.config import load_preprocess_config
from src.dataprep.pipeline import run


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build df_clean + logs/splits from raw data using YAML config.")
    default_cfg = PROJECT_ROOT / "configs" / "preprocess.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"Path to preprocess YAML (default: {default_cfg})",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress prints")
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N files (0 to disable)",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERR] config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        config, repo_root, _ = load_preprocess_config(config_path, SCRIPT_PATH)
        run(
            config,
            config_path=config_path,
            repo_root=repo_root,
            quiet=bool(args.quiet),
            log_every=int(args.log_every),
        )
        print("[OK] preprocess completed")
        return 0
    except KeyboardInterrupt:
        print("[INT] interrupted by user (Ctrl+C).", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
