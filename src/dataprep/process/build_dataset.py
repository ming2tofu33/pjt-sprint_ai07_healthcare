from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 스크립트를 어디서 실행하든 프로젝트 루트를 import 경로에 올린다.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]  # src/dataprep/process/ → repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_preprocess_config
from src.dataprep.output.data_pipeline import run


def main(argv: list[str]) -> int:
    """YAML 설정 기반 전처리 파이프라인 진입점."""
    parser = argparse.ArgumentParser(description="YAML 설정으로 원천 데이터에서 df_clean + 로그/split을 생성합니다.")
    default_cfg = PROJECT_ROOT / "configs" / "base.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"전처리 YAML 경로 (기본값: {default_cfg})",
    )
    parser.add_argument("--quiet", action="store_true", help="진행 로그 출력 비활성화")
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="N개 파일마다 진행상황 출력 (0이면 비활성화)",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"[ERR] 설정 파일이 존재하지 않습니다: {config_path}", file=sys.stderr)
        return 2

    try:
        # 설정 로드 -> 경로 해석 -> 전체 파이프라인 실행
        config, repo_root, _ = load_preprocess_config(config_path, SCRIPT_PATH)
        run(
            config,
            config_path=config_path,
            repo_root=repo_root,
            quiet=bool(args.quiet),
            log_every=int(args.log_every),
        )
        print("[OK] 전처리 작업이 완료되었습니다.")
        return 0
    except KeyboardInterrupt:
        # 배치 실행 중 Ctrl+C 중단 시 관례적인 종료코드(130) 사용
        print("[INT] 사용자에 의해 중단되었습니다 (Ctrl+C).", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
