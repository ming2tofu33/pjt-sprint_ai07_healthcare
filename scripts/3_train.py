#!/usr/bin/env python3
"""
YOLO 학습 스크립트

기능:
1. Ultralytics YOLO 모델 학습
2. Config 기반 하이퍼파라미터 설정
3. 체크포인트 저장 (best.pt, last.pt)
4. 학습 로그 및 메트릭 기록

사용법:
    python scripts/3_train.py [--run-name RUN_NAME] [--model MODEL] [--epochs EPOCHS]
    
    예시:
    python scripts/3_train.py --run-name exp_baseline_v1
    python scripts/3_train.py --run-name exp_v1 --model yolov8m --epochs 100
    python scripts/3_train.py --run-name exp_v1 --resume
"""

import sys
import argparse
from pathlib import Path

# Windows 인코딩 수정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_config,
    save_config,
    set_seed,
    save_json,
    print_section,
)


def main():
    parser = argparse.ArgumentParser(description="YOLO 모델 학습")
    parser.add_argument("--run-name", type=str, help="실험명")
    parser.add_argument("--model", type=str, help="YOLO 모델 (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, help="Epoch 수")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 재개")
    parser.add_argument("--device", type=str, default="0", help="GPU device (0, 1, cpu)")
    args = parser.parse_args()
    
    print_section("Stage 2-2: YOLO 학습")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=False,
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    print(f"  ✅ CKPT: {paths['CKPT']}")
    
    # 2) Config 로드
    print("\n[2] Config 로드...")
    config_path = paths["CONFIG"] / "config.json"
    if config_path.exists():
        config = load_config(config_path)
        print(f"  ✅ Config: {config_path.relative_to(paths['ROOT'])}")
    else:
        from utils import get_default_config
        config = get_default_config(paths["RUN_NAME"], paths)
        save_config(config, config_path)
        print(f"  ✅ 기본 Config 생성: {config_path.relative_to(paths['ROOT'])}")
    
    # 3) Seed 설정
    print("\n[3] Seed 설정...")
    seed = config.get("reproducibility", {}).get("seed", 42)
    deterministic = config.get("reproducibility", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    print(f"  ✅ Seed: {seed}")
    
    # 4) data.yaml 확인
    print("\n[4] data.yaml 확인...")
    dataset_root = paths["PROC_ROOT"] / "datasets" / f"pill_od_yolo_{paths['RUN_NAME']}"
    data_yaml = dataset_root / "data.yaml"
    
    if not data_yaml.exists():
        print(f"  ❌ data.yaml 없음: {data_yaml}")
        print(f"  ℹ️  먼저 scripts/2_prepare_yolo_dataset.py를 실행하세요.")
        sys.exit(1)
    
    print(f"  ✅ {data_yaml.relative_to(paths['ROOT'])}")
    
    # 5) 학습 파라미터
    print("\n[5] 학습 파라미터 설정...")
    train_config = config["train"]
    
    # CLI 인자가 있으면 override
    model_name = args.model or train_config["model"]["name"]
    imgsz = args.imgsz or train_config["model"]["imgsz"]
    epochs = args.epochs or train_config["hyperparams"]["epochs"]
    batch = args.batch or train_config["hyperparams"]["batch"]
    
    print(f"  ✅ Model: {model_name}")
    print(f"  ✅ Image size: {imgsz}")
    print(f"  ✅ Epochs: {epochs}")
    print(f"  ✅ Batch: {batch}")
    print(f"  ✅ Device: {args.device}")
    
    # 6) YOLO 학습
    print("\n[6] YOLO 학습 시작...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics 패키지가 설치되지 않았습니다.")
        print("  ℹ️  pip install ultralytics")
        sys.exit(1)
    
    # Model 로드
    if args.resume:
        last_ckpt = paths["CKPT"] / "last.pt"
        if last_ckpt.exists():
            print(f"  ℹ️  Resume from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"  ⚠️  last.pt 없음, 새로 시작")
            model = YOLO(f"{model_name}.pt")
    else:
        model = YOLO(f"{model_name}.pt")
    
    # 학습 실행
    print(f"\n  🚀 학습 중... (이 작업은 시간이 걸립니다)")
    print(f"     - Epochs: {epochs}")
    print(f"     - Batch: {batch}")
    print(f"     - Device: {args.device}")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        project=str(paths["RUN_DIR"]),
        name="train",
        exist_ok=True,
        # Ultralytics 기본값 사용 (필요 시 추가 설정)
        patience=train_config["hyperparams"].get("patience", 50),
        save=True,
        save_period=-1,  # 마지막에만 저장
        val=True,
        plots=True,
        verbose=True,
    )
    
    print(f"\n  ✅ 학습 완료!")
    
    # 7) 체크포인트 복사
    print("\n[7] 체크포인트 정리...")
    yolo_train_dir = paths["RUN_DIR"] / "train"
    yolo_weights_dir = yolo_train_dir / "weights"
    
    if (yolo_weights_dir / "best.pt").exists():
        import shutil
        shutil.copy2(yolo_weights_dir / "best.pt", paths["CKPT"] / "best.pt")
        print(f"  ✅ best.pt → {paths['CKPT']}/best.pt")
    
    if (yolo_weights_dir / "last.pt").exists():
        import shutil
        shutil.copy2(yolo_weights_dir / "last.pt", paths["CKPT"] / "last.pt")
        print(f"  ✅ last.pt → {paths['CKPT']}/last.pt")
    
    # 8) 학습 메타 저장
    print("\n[8] 학습 메타 저장...")
    train_meta = {
        "model": model_name,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": args.device,
        "data_yaml": str(data_yaml),
        "yolo_train_dir": str(yolo_train_dir),
        "best_ckpt": str(paths["CKPT"] / "best.pt"),
        "last_ckpt": str(paths["CKPT"] / "last.pt"),
    }
    
    train_meta_path = paths["CONFIG"] / "train_meta.json"
    save_json(train_meta_path, train_meta)
    print(f"  ✅ {train_meta_path.relative_to(paths['ROOT'])}")
    
    # 9) Config 업데이트
    config["train"]["model"]["name"] = model_name
    config["train"]["model"]["imgsz"] = imgsz
    config["train"]["hyperparams"]["epochs"] = epochs
    config["train"]["hyperparams"]["batch"] = batch
    save_config(config, config_path)
    print(f"  ✅ Config 업데이트")
    
    print_section("✅ 학습 완료")
    print(f"\n체크포인트:")
    print(f"  - best.pt: {paths['CKPT']}/best.pt")
    print(f"  - last.pt: {paths['CKPT']}/last.pt")
    print(f"\n다음 단계:")
    print(f"  python scripts/4_evaluate.py --run-name {paths['RUN_NAME']}")
    print(f"  python scripts/5_submission.py --run-name {paths['RUN_NAME']}")


if __name__ == "__main__":
    main()
