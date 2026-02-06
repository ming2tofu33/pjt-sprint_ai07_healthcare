#!/usr/bin/env python3
"""
[DEPRECATED] Trainer Module
학습 프로세스 관리 및 실행

NOTE: 현재 파이프라인에서 사용되지 않습니다.
      실제 학습은 scripts/3_train.py에서 ultralytics.YOLO를 직접 호출합니다.
      향후 필요 시 참고용으로 보존합니다.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

from .model import YOLOModel
from .utils import (
    setup_project_paths,
    load_config,
    save_config,
    set_seed,
    create_run_manifest,
    record_result,
    print_section,
)


class Trainer:
    """
    모델 학습 관리자
    
    Args:
        run_name: 실험명
        config: 설정 딕셔너리 또는 경로
        root: 프로젝트 루트 경로
        device: 디바이스 ("0", "cpu", ...)
        verbose: 상세 출력 여부
    """
    
    def __init__(
        self,
        run_name: str,
        config: Optional[Dict] = None,
        root: Optional[Path] = None,
        device: str = "0",
        verbose: bool = True,
    ):
        self.run_name = run_name
        self.device = device
        self.verbose = verbose
        
        if root is None:
            root = Path.cwd()
        self.root = Path(root)
        
        # 경로 설정
        self.paths = setup_project_paths(
            run_name=run_name,
            root=self.root,
            create_dirs=True,
            check_input_exists=False,  # Trainer는 데이터 존재 체크 안 함
        )
        
        # Config 로드/생성
        if config is None:
            # 기존 config 로드 시도
            config_path = self.paths["CONFIG"] / "config.json"
            if config_path.exists():
                self.config = load_config(config_path)
                if self.verbose:
                    print(f"✅ Config loaded: {config_path}")
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
        elif isinstance(config, (str, Path)):
            # 파일에서 로드
            self.config = load_config(Path(config))
        else:
            # 딕셔너리 직접 사용
            self.config = config
        
        # 재현성 설정
        seed = self.config.get("reproducibility", {}).get("seed", 42)
        deterministic = self.config.get("reproducibility", {}).get("deterministic", True)
        set_seed(seed, deterministic=deterministic)
        
        # Run manifest 생성
        create_run_manifest(
            run_dir=self.paths["RUN_DIR"],
            config=self.config,
        )
        
        # 모델 초기화 (나중에 로드)
        self.model = None
        
        if self.verbose:
            print_section(f"Trainer Initialized: {run_name}")
            print(f"  Run dir: {self.paths['RUN_DIR']}")
            print(f"  Device: {device}")
    
    def prepare_model(self, model_name: Optional[str] = None):
        """
        모델 준비
        
        Args:
            model_name: 모델 이름 (None이면 config에서 가져옴)
        """
        if model_name is None:
            model_name = self.config.get("train", {}).get("model_name", "yolov8s.pt")
        
        self.model = YOLOModel(
            model_name=model_name,
            device=self.device,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print(f"✅ Model prepared: {model_name}")
    
    def train(
        self,
        data_yaml: str,
        resume: bool = False,
        **override_kwargs,
    ) -> Any:
        """
        학습 실행
        
        Args:
            data_yaml: data.yaml 경로
            resume: 중단된 학습 재개 여부
            **override_kwargs: Config override 인자
        
        Returns:
            학습 결과 객체
        """
        if self.model is None:
            self.prepare_model()
        
        # Config에서 학습 파라미터 가져오기
        train_config = self.config.get("train", {})
        
        train_params = {
            "data": data_yaml,
            "epochs": train_config.get("epochs", 80),
            "imgsz": train_config.get("imgsz", 768),
            "batch": train_config.get("batch", 8),
            "lr0": train_config.get("lr0", 0.001),
            "lrf": train_config.get("lrf", 0.01),
            "momentum": train_config.get("momentum", 0.937),
            "weight_decay": train_config.get("weight_decay", 0.0005),
            "warmup_epochs": train_config.get("warmup_epochs", 3.0),
            "warmup_momentum": train_config.get("warmup_momentum", 0.8),
            "warmup_bias_lr": train_config.get("warmup_bias_lr", 0.1),
            "box": train_config.get("box", 7.5),
            "cls": train_config.get("cls", 0.5),
            "dfl": train_config.get("dfl", 1.5),
            "hsv_h": train_config.get("hsv_h", 0.015),
            "hsv_s": train_config.get("hsv_s", 0.7),
            "hsv_v": train_config.get("hsv_v", 0.4),
            "degrees": train_config.get("degrees", 0.0),
            "translate": train_config.get("translate", 0.1),
            "scale": train_config.get("scale", 0.5),
            "shear": train_config.get("shear", 0.0),
            "perspective": train_config.get("perspective", 0.0),
            "flipud": train_config.get("flipud", 0.0),
            "fliplr": train_config.get("fliplr", 0.5),
            "mosaic": train_config.get("mosaic", 1.0),
            "mixup": train_config.get("mixup", 0.0),
            "copy_paste": train_config.get("copy_paste", 0.0),
            "patience": train_config.get("patience", 50),
            "save": train_config.get("save", True),
            "save_period": train_config.get("save_period", -1),
            "workers": train_config.get("workers", 4),
            "optimizer": train_config.get("optimizer", "auto"),
            "verbose": train_config.get("verbose", True),
            "plots": train_config.get("plots", True),
            "project": str(self.paths["RUN_DIR"].parent),  # runs/
            "name": self.run_name,
            "exist_ok": True,
            "resume": resume,
        }
        
        # Override 적용
        train_params.update(override_kwargs)
        
        if self.verbose:
            print_section("Starting Training")
            print(f"  Data: {data_yaml}")
            print(f"  Epochs: {train_params['epochs']}")
            print(f"  Image size: {train_params['imgsz']}")
            print(f"  Batch: {train_params['batch']}")
            print(f"  LR: {train_params['lr0']}")
        
        # 학습 시작 시간
        start_time = time.time()
        
        # 학습 실행
        results = self.model.train(**train_params)
        
        # 학습 종료 시간
        end_time = time.time()
        elapsed = end_time - start_time
        
        if self.verbose:
            print_section("Training Completed")
            print(f"  Elapsed: {elapsed / 60:.2f} min")
        
        # 결과 기록
        self._record_training_results(results, elapsed)
        
        return results
    
    def _record_training_results(self, results: Any, elapsed: float):
        """
        학습 결과 기록
        
        Args:
            results: 학습 결과 객체
            elapsed: 소요 시간 (초)
        """
        # 체크포인트 경로 확인
        best_ckpt = self.paths["CKPT"] / "best.pt"
        last_ckpt = self.paths["CKPT"] / "last.pt"
        
        # Ultralytics는 runs/<project>/<name>/weights/에 저장
        # 그것을 우리 경로로 복사/이동 필요 (TODO: 나중에 scripts에서 처리)
        
        result_entry = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": elapsed,
            "elapsed_min": elapsed / 60,
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
            "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else None,
            "status": "completed",
        }
        
        # 결과 CSV에 기록
        record_result(
            artifacts_dir=self.paths["ART_DIR"],
            **result_entry,
        )
        
        if self.verbose:
            print(f"✅ Results recorded to {self.paths['ART_DIR'] / 'reports'}")
    
    def evaluate(self, split: str = "val", **kwargs) -> Any:
        """
        모델 평가
        
        Args:
            split: 분할 (val/test)
            **kwargs: 추가 인자
        
        Returns:
            평가 결과 객체
        """
        if self.model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
        
        if self.verbose:
            print_section(f"Evaluating on {split}")
        
        results = self.model.validate(split=split, **kwargs)
        
        if self.verbose:
            print_section("Evaluation Completed")
        
        return results
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로 (None이면 best.pt)
        """
        if checkpoint_path is None:
            checkpoint_path = self.paths["CKPT"] / "best.pt"
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.model is None:
            self.model = YOLOModel(
                model_name=str(checkpoint_path),
                device=self.device,
                verbose=self.verbose,
            )
        else:
            self.model.load_checkpoint(str(checkpoint_path))
        
        if self.verbose:
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        학습 요약 정보 반환
        
        Returns:
            {
                "run_name": str,
                "config": dict,
                "paths": dict,
                "model_info": dict,
            }
        """
        summary = {
            "run_name": self.run_name,
            "config": self.config,
            "paths": {k: str(v) for k, v in self.paths.items()},
        }
        
        if self.model:
            summary["model_info"] = self.model.get_model_info()
        
        return summary


if __name__ == "__main__":
    # 간단한 테스트
    print("trainer.py: Training Process Manager")
    print("\nUsage:")
    print("  from src.trainer import Trainer")
    print("  trainer = Trainer(run_name='exp001', config='configs/base.yaml')")
    print("  trainer.train(data_yaml='data/datasets/exp001_yolo/data.yaml')")
