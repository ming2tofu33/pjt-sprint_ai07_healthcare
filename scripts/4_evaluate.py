#!/usr/bin/env python3
"""
모델 평가 스크립트 (Placeholder)

기능:
1. Best 체크포인트로 Val set 평가
2. mAP@[0.75:0.95] 계산
3. 클래스별 AP 계산
4. 평가 리포트 저장

사용법:
    python scripts/4_evaluate.py --run-name RUN_NAME
    
    예시:
    python scripts/4_evaluate.py --run-name exp_baseline_v1
    python scripts/4_evaluate.py --run-name exp_v1 --ckpt best
"""

import sys
import argparse
from pathlib import Path

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_config,
    save_config,
    save_json,
    print_section,
    get_data_yaml,
)


def main():
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--run-name", type=str, required=True, help="실험명")
    parser.add_argument("--config", type=str, help="Config YAML 경로 (상속 지원)")
    parser.add_argument("--ckpt", type=str, default="best", choices=["best", "last"], help="체크포인트")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    args = parser.parse_args()
    
    print_section("Stage 2-3: 모델 평가")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=False,
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    
    # 2) 체크포인트 확인
    print("\n[2] 체크포인트 확인...")
    ckpt_path = paths["CKPT"] / f"{args.ckpt}.pt"
    
    if not ckpt_path.exists():
        print(f"  ❌ 체크포인트 없음: {ckpt_path}")
        print(f"  ℹ️  먼저 scripts/3_train.py를 실행하세요.")
        sys.exit(1)
    
    print(f"  ✅ {ckpt_path.relative_to(paths['ROOT'])}")

    # 3) Config 로드
    print("\n[3] Config 로드...")
    config_path = paths["CONFIG"] / "config.json"
    if args.config:
        config = load_config(Path(args.config))
        print(f"  ✅ Config from YAML: {args.config}")
    elif config_path.exists():
        config = load_config(config_path)
        print(f"  ✅ Config: {config_path.relative_to(paths['ROOT'])}")
    else:
        from utils import get_default_config
        config = get_default_config(paths["RUN_NAME"], paths)
        print(f"  ✅ 기본 Config 사용")
    save_config(config, config_path)

    # 4) data.yaml 확인
    print("\n[4] data.yaml 확인...")
    data_yaml = get_data_yaml(paths)

    if not data_yaml.exists():
        print(f"  ❌ data.yaml 없음")
        sys.exit(1)

    print(f"  ✅ {data_yaml.relative_to(paths['ROOT'])}")

    # 5) 평가 실행
    print("\n[5] 평가 실행...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics 패키지가 설치되지 않았습니다.")
        sys.exit(1)
    
    model = YOLO(str(ckpt_path))
    
    print(f"  🚀 평가 중...")
    val_config = config.get("val", {})
    metrics = model.val(
        data=str(data_yaml),
        device=args.device,
        conf=val_config.get("conf", 0.001),
        iou=val_config.get("iou", 0.6),
        project=str(paths["RUN_DIR"]),
        name="eval",
        exist_ok=True,
        save_json=val_config.get("save_json", True),
        plots=True,
    )
    
    print(f"\n  ✅ 평가 완료!")
    
    # 6) 메트릭 추출
    print("\n[6] 메트릭 저장...")
    results = {
        "mAP_50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
        "mAP_75": float(metrics.box.map75) if hasattr(metrics.box, 'map75') else None,
        "mAP_50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else None,
    }
    
    print(f"  📊 mAP@0.5: {results['mAP_50']:.4f}" if results['mAP_50'] else "  ⚠️  mAP@0.5: N/A")
    print(f"  📊 mAP@0.75: {results['mAP_75']:.4f}" if results['mAP_75'] else "  ⚠️  mAP@0.75: N/A")
    print(f"  📊 mAP@[0.5:0.95]: {results['mAP_50_95']:.4f}" if results['mAP_50_95'] else "  ⚠️  mAP@[0.5:0.95]: N/A")
    
    # 평가 결과 저장
    eval_results_path = paths["REPORTS"] / "eval_results.json"
    save_json(eval_results_path, results)
    print(f"\n  ✅ {eval_results_path.relative_to(paths['ROOT'])}")
    
    print_section("✅ 평가 완료")
    print(f"\n다음 단계:")
    print(f"  python scripts/5_submission.py --run-name {paths['RUN_NAME']}")


if __name__ == "__main__":
    main()
