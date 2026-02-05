"""
src/utils.py 기능 테스트 스크립트

사용법:
    cd /home/user/webapp
    python test_utils.py
"""

import sys
from pathlib import Path

# src 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import (
    setup_project_paths,
    set_seed,
    get_default_config,
    save_config,
    create_run_manifest,
    save_json,
    print_section,
    print_dict,
)


def main():
    print_section("Stage 0: src/utils.py 기능 테스트")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정 및 검증...")
    try:
        paths = setup_project_paths(
            run_name="test_utils_demo",
            root=Path(__file__).parent,  # 프로젝트 루트
            create_dirs=True,
            check_input_exists=False,  # 데이터가 없어도 테스트 가능하게
        )
        print(f"✅ ROOT: {paths['ROOT']}")
        print(f"✅ RUN_NAME: {paths['RUN_NAME']}")
        print(f"✅ RUN_DIR: {paths['RUN_DIR']}")
        
        # 데이터 폴더 확인
        if paths['TRAIN_IMAGES'].exists():
            print(f"✅ INPUT paths OK: {list(paths['TRAIN_IMAGES'].glob('*'))[:3]}")
        else:
            print(f"⚠️  데이터 폴더 없음 (테스트용으로는 OK)")
    except Exception as e:
        print(f"❌ 경로 설정 실패: {e}")
        return
    
    # 2) Seed 고정 및 환경 정보
    print("\n[2] Seed 고정 및 환경 수집...")
    env_meta = set_seed(seed=42, deterministic=True)
    print(f"✅ Seed: {env_meta['seed']}")
    print(f"✅ Python: {env_meta['python']['version'][:20]}...")
    if env_meta['torch']:
        print(f"✅ Torch: {env_meta['torch']['torch_version']}")
        print(f"✅ CUDA: {env_meta['torch']['cuda_available']}")
    
    # 환경 메타 저장
    env_meta_path = paths["CONFIG"] / "env_meta.json"
    save_json(env_meta_path, env_meta)
    print(f"✅ 환경 메타 저장: {env_meta_path}")
    
    # 3) Config 생성 및 저장
    print("\n[3] 기본 Config 생성...")
    config = get_default_config(
        run_name=paths["RUN_NAME"],
        paths=paths,
        seed=42,
    )
    
    print("✅ Config 주요 필드:")
    print(f"  - model: {config['train']['model']['name']}")
    print(f"  - imgsz: {config['train']['model']['imgsz']}")
    print(f"  - epochs: {config['train']['hyperparams']['epochs']}")
    print(f"  - batch: {config['train']['hyperparams']['batch']}")
    
    # Config 저장
    config_path = paths["CONFIG"] / "config.json"
    save_config(config, config_path)
    print(f"✅ Config 저장: {config_path}")
    
    # 4) Manifest 생성
    print("\n[4] Run Manifest 생성...")
    manifest = create_run_manifest(paths["RUN_NAME"], paths)
    manifest_path = paths["CONFIG"] / "run_manifest.json"
    save_json(manifest_path, manifest)
    print(f"✅ Manifest 저장: {manifest_path}")
    print(f"  - Git branch: {manifest['git']['git_branch']}")
    print(f"  - Git dirty: {manifest['git']['git_dirty']}")
    
    # 5) 요약 출력
    print_section("테스트 완료")
    print("\n생성된 파일들:")
    for p in paths["CONFIG"].glob("*"):
        print(f"  - {p.relative_to(paths['ROOT'])}")
    
    print("\n✅ Stage 0 (src/utils.py) 구현 완료!")
    print(f"\n다음 단계:")
    print(f"  - Stage 1: scripts/0_splitting.py (데이터 분할)")
    print(f"  - Stage 1: scripts/1_create_coco_format.py (COCO 변환)")


if __name__ == "__main__":
    main()
