#!/usr/bin/env python3
"""
YOLO 학습 스크립트

기능:
1. Ultralytics YOLO 모델 학습
2. Config 기반 하이퍼파라미터 설정 (augmentation, optimizer 등 전부 반영)
3. 체크포인트 저장 (best.pt, last.pt)
4. 학습 로그 및 메트릭 기록

사용법:
    python scripts/3_train.py [--run-name RUN_NAME] [--config CONFIG_PATH]

    예시:
    python scripts/3_train.py --run-name exp_baseline_v1
    python scripts/3_train.py --run-name exp_v1 --config configs/experiments/exp007_final.yaml
    python scripts/3_train.py --run-name exp_v1 --model yolov8m --epochs 100
    python scripts/3_train.py --run-name exp_v1 --resume

    2단계 학습 (다른 run의 체크포인트로 시작):
    python scripts/3_train.py --run-name exp020_s2 --ckpt-from runs/exp020_s1/checkpoints/best.pt --config configs/experiments/exp020_stage2.yaml
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
    get_data_yaml,
)


def main():
    parser = argparse.ArgumentParser(description="YOLO 모델 학습")
    parser.add_argument("--run-name", type=str, help="실험명")
    parser.add_argument("--config", type=str, help="Config YAML 경로 (상속 지원)")
    parser.add_argument("--model", type=str, help="YOLO 모델 (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, help="Epoch 수")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 재개")
    parser.add_argument("--ckpt-from", type=str, help="다른 run의 체크포인트 경로 (2단계 학습용, 예: runs/exp020_s1/checkpoints/best.pt)")
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

    # 2) Config 로드 (우선순위: CLI --config > 기존 config.json > default)
    print("\n[2] Config 로드...")
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
        print(f"  ✅ 기본 Config 생성")
    # 항상 config.json으로 저장 (재현성)
    save_config(config, config_path)

    # 3) Seed 설정
    print("\n[3] Seed 설정...")
    seed = config.get("reproducibility", {}).get("seed", 42)
    deterministic = config.get("reproducibility", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    print(f"  ✅ Seed: {seed}")

    # 4) data.yaml 확인
    print("\n[4] data.yaml 확인...")
    data_yaml = get_data_yaml(paths)

    if not data_yaml.exists():
        print(f"  ❌ data.yaml 없음: {data_yaml}")
        print(f"  ℹ️  먼저 scripts/2_prepare_yolo_dataset.py를 실행하세요.")
        sys.exit(1)

    print(f"  ✅ {data_yaml.relative_to(paths['ROOT'])}")

    # 5) 학습 파라미터 (flat config 구조)
    print("\n[5] 학습 파라미터 설정...")
    train_config = config["train"]

    # CLI 인자가 있으면 override
    model_name_raw = args.model or train_config.get("model_name", "yolov8s")
    # .pt 확장자 정규화 (config에 "yolov8s.pt" 또는 "yolov8s" 모두 허용)
    model_name = model_name_raw.removesuffix(".pt")
    imgsz = args.imgsz or train_config.get("imgsz", 768)
    epochs = args.epochs or train_config.get("epochs", 80)
    batch = args.batch or train_config.get("batch", 8)

    print(f"  ✅ Model: {model_name}")
    print(f"  ✅ Image size: {imgsz}")
    print(f"  ✅ Epochs: {epochs}")
    print(f"  ✅ Batch: {batch}")
    print(f"  ✅ Device: {args.device}")
    print(f"  ✅ Optimizer: {train_config.get('optimizer', 'auto')}")
    print(f"  ✅ lr0: {train_config.get('lr0', 0.001)}")
    print(f"  ✅ Mosaic: {train_config.get('mosaic', 1.0)}")
    print(f"  ✅ Mixup: {train_config.get('mixup', 0.0)}")

    # 6) YOLO 학습
    print("\n[6] YOLO 학습 시작...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics 패키지가 설치되지 않았습니다.")
        print("  ℹ️  pip install ultralytics")
        sys.exit(1)

    # Model 로드
    if args.ckpt_from:
        # 다른 run의 체크포인트로 시작 (2단계 학습용)
        ckpt_from = Path(args.ckpt_from)
        if not ckpt_from.is_absolute():
            ckpt_from = paths["ROOT"] / ckpt_from
        if ckpt_from.exists():
            print(f"  ℹ️  외부 체크포인트 로드: {ckpt_from}")
            model = YOLO(str(ckpt_from))
        else:
            print(f"  ❌ 체크포인트 없음: {ckpt_from}")
            sys.exit(1)
    elif args.resume:
        last_ckpt = paths["CKPT"] / "last.pt"
        if last_ckpt.exists():
            print(f"  ℹ️  Resume from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"  ⚠️  last.pt 없음, 새로 시작")
            model = YOLO(f"{model_name}.pt")
    else:
        model = YOLO(f"{model_name}.pt")

    # 학습 실행 - config에서 ALL 파라미터 전달
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
        # Hyperparams from config
        patience=train_config.get("patience", 50),
        optimizer=train_config.get("optimizer", "auto"),
        lr0=train_config.get("lr0", 0.001),
        lrf=train_config.get("lrf", 0.01),
        momentum=train_config.get("momentum", 0.937),
        weight_decay=train_config.get("weight_decay", 0.0005),
        warmup_epochs=train_config.get("warmup_epochs", 3.0),
        warmup_momentum=train_config.get("warmup_momentum", 0.8),
        warmup_bias_lr=train_config.get("warmup_bias_lr", 0.1),
        workers=train_config.get("workers", 4),
        # Augmentation from config
        hsv_h=train_config.get("hsv_h", 0.015),
        hsv_s=train_config.get("hsv_s", 0.7),
        hsv_v=train_config.get("hsv_v", 0.4),
        degrees=train_config.get("degrees", 0.0),
        translate=train_config.get("translate", 0.1),
        scale=train_config.get("scale", 0.5),
        shear=train_config.get("shear", 0.0),
        perspective=train_config.get("perspective", 0.0),
        flipud=train_config.get("flipud", 0.0),
        fliplr=train_config.get("fliplr", 0.5),
        mosaic=train_config.get("mosaic", 1.0),
        mixup=train_config.get("mixup", 0.0),
        copy_paste=train_config.get("copy_paste", 0.0),
        # Loss from config
        box=train_config.get("box", 7.5),
        cls=train_config.get("cls", 0.5),
        dfl=train_config.get("dfl", 1.5),
        # Misc from config
        save=train_config.get("save", True),
        save_period=train_config.get("save_period", -1),
        val=True,
        plots=train_config.get("plots", True),
        verbose=train_config.get("verbose", True),
        pretrained=train_config.get("pretrained", True),
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

    # 9) Config 업데이트 (실제 사용된 값 기록)
    config["train"]["model_name"] = model_name
    config["train"]["imgsz"] = imgsz
    config["train"]["epochs"] = epochs
    config["train"]["batch"] = batch
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
