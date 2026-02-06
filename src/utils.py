"""
공통 유틸리티 모듈
- 경로 설정 및 검증
- 재현성 설정 (Seed 고정)
- 실험 관리 (Config, Manifest, Logging)
- 결과 기록 (CSV, JSONL, Markdown)
"""

import os
import sys
import json
import csv
import random
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Any, Union


# ============================================================
# 1. Path Management
# ============================================================

def setup_project_paths(
    run_name: Optional[str] = None,
    root: Optional[Path] = None,
    create_dirs: bool = True,
    check_input_exists: bool = True,
) -> Dict[str, Path]:
    """
    프로젝트 디렉터리 구조 설정 및 검증
    
    Args:
        run_name: 실험 이름 (None이면 자동 생성: exp_YYYYMMDD_HHMMSS)
        root: 프로젝트 루트 경로 (None이면 현재 디렉터리 기준으로 탐색)
        create_dirs: True면 필요한 디렉터리 자동 생성
        check_input_exists: True면 input 폴더 존재 여부 검증 (False면 경고만)
    
    Returns:
        Dict[str, Path]: 주요 경로들이 담긴 딕셔너리
            - ROOT, DATA_ROOT, RAW_ROOT, PROC_ROOT
            - INPUT (TRAIN_IMAGES, TRAIN_ANN_DIR, TEST_IMAGES)
            - WORK (DATA, RUNS, ARTIFACTS)
            - RUN_DIR, ART_DIR
            - DIRS (CKPT, LOGS, CONFIG, SUBMISSIONS, PLOTS, REPORTS, CACHE)
    """
    # 1) 프로젝트 루트 찾기
    if root is None:
        cwd = Path.cwd().resolve()
        # notebooks 폴더에서 실행되면 상위로 올라감
        if cwd.name == "notebooks":
            root = cwd.parent
        else:
            root = cwd
    else:
        root = Path(root).resolve()
    
    # 2) 데이터 구조
    data_root = root / "data"
    raw_root = data_root / "raw"
    proc_root = data_root / "processed"
    
    # 3) Input paths (필수 존재 체크용)
    input_paths = {
        "TRAIN_IMAGES": raw_root / "train_images",
        "TRAIN_ANN_DIR": raw_root / "train_annotations",
        "TEST_IMAGES": raw_root / "test_images",
    }
    
    # 4) Working directories
    work_dirs = {
        "DATA": proc_root,
        "RUNS": root / "runs",
        "ARTIFACTS": root / "artifacts",
    }
    
    # 5) RUN 이름 생성
    if run_name is None:
        run_name = os.environ.get("RUN_NAME") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_dir = work_dirs["RUNS"] / run_name
    art_dir = work_dirs["ARTIFACTS"] / run_name
    
    # 6) RUN 하위 구조
    run_subdirs = {
        "RUN_DIR": run_dir,
        "CKPT": run_dir / "checkpoints",
        "LOGS": run_dir / "logs",
        "CONFIG": run_dir / "config",
        "ART_DIR": art_dir,
        "SUBMISSIONS": art_dir / "submissions",
        "PLOTS": art_dir / "plots",
        "REPORTS": art_dir / "reports",
        "CACHE": proc_root / "cache" / run_name,
    }
    
    # 7) 디렉터리 생성
    if create_dirs:
        # Work dirs 생성 (없으면 자동 생성)
        for p in work_dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        
        # Run subdirs 생성
        for p in run_subdirs.values():
            p.mkdir(parents=True, exist_ok=True)
    
    # 8) Input paths 존재 체크
    missing = [k for k, p in input_paths.items() if not p.exists()]
    if missing:
        if check_input_exists:
            raise FileNotFoundError(
                "필수 INPUT 폴더가 없습니다:\n" +
                "\n".join([f"- {k}: {str(input_paths[k])}" for k in missing])
            )
        else:
            print("[WARN] INPUT 폴더가 없습니다 (check_input_exists=False 이므로 경고만):")
            for k in missing:
                print(f"  - {k}: {str(input_paths[k])}")
    
    # 9) 메타데이터 저장
    paths_meta = {
        "run_name": run_name,
        "root": str(root),
        "input": {k: str(v) for k, v in input_paths.items()},
        "work": {k: str(v) for k, v in work_dirs.items()},
        "dirs": {k: str(v) for k, v in run_subdirs.items()},
    }
    
    meta_path = run_subdirs["CONFIG"] / "paths_meta.json"
    save_json(meta_path, paths_meta)
    
    # 10) 통합 반환
    return {
        "ROOT": root,
        "DATA_ROOT": data_root,
        "RAW_ROOT": raw_root,
        "PROC_ROOT": proc_root,
        "RUN_NAME": run_name,
        **input_paths,
        **work_dirs,
        **run_subdirs,
    }


def get_dataset_dir(paths: Dict[str, Path]) -> Path:
    """YOLO dataset 디렉토리 경로 반환"""
    return paths["PROC_ROOT"] / "datasets" / f"pill_od_yolo_{paths['RUN_NAME']}"


def get_data_yaml(paths: Dict[str, Path]) -> Path:
    """data.yaml 경로 반환"""
    return get_dataset_dir(paths) / "data.yaml"


def count_files(folder: Path, exts: Optional[List[str]] = None) -> int:
    """폴더 내 파일 개수 세기 (재귀)"""
    if not folder.exists():
        return 0
    if exts is None:
        return sum(1 for _ in folder.rglob("*") if _.is_file())
    exts = {e.lower() for e in exts}
    return sum(1 for _ in folder.rglob("*") if _.is_file() and _.suffix.lower() in exts)


# ============================================================
# 2. Reproducibility (Seed Fixing)
# ============================================================

def set_seed(seed: int = 42, deterministic: bool = True) -> Dict[str, Any]:
    """
    재현성을 위한 Seed 고정 및 환경 정보 수집
    
    Args:
        seed: Random seed 값
        deterministic: True면 cudnn deterministic 모드 활성화 (성능 저하 가능)
    
    Returns:
        Dict: 환경 메타정보 (Python, Torch, 패키지 버전 등)
    """
    # 1) Python/OS seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    
    # 2) NumPy seed
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        np = None
    
    # 3) Torch seed + cudnn
    torch_info = {}
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        
        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
        }
    except ImportError:
        torch = None
    
    # 4) 패키지 버전 수집
    pkg_list = [
        "numpy", "pandas", "opencv-python", "albumentations",
        "pycocotools", "torch", "torchvision", "ultralytics",
        "timm", "matplotlib", "scikit-learn",
    ]
    pkg_versions = {p: get_package_version(p) for p in pkg_list}
    
    # 5) 환경 메타 구성
    env_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "deterministic": deterministic,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": pkg_versions,
        "torch": torch_info,
    }
    
    return env_meta


def get_package_version(pkg_name: str) -> Optional[str]:
    """패키지 버전 안전하게 가져오기"""
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


# ============================================================
# 3. Config Management
# ============================================================

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge: override 값을 base 위에 덮어쓰기

    Args:
        base: 기본 config dict
        override: 덮어쓸 config dict

    Returns:
        병합된 config dict
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    JSON/YAML config 파일 로드 (_base_ 상속 지원)

    YAML 파일에 _base_ 키가 있으면 해당 파일을 먼저 로드하고
    현재 파일의 값으로 deep merge합니다.

    예시:
        # experiment.yaml
        _base_: "../base.yaml"
        train:
          model_name: "yolov8m"
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # _base_ 상속 처리
        if config and "_base_" in config:
            base_rel = config.pop("_base_")
            base_path = (config_path.parent / base_rel).resolve()
            if not base_path.exists():
                raise FileNotFoundError(f"Base config not found: {base_path}")
            base_config = load_config(base_path)  # 재귀 (다단계 상속 지원)
            config = merge_configs(base_config, config)

        return config or {}
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: Path):
    """Config를 JSON/YAML로 저장"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    elif config_path.suffix == ".json":
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def get_default_config(run_name: str, paths: Dict[str, Path], seed: int = 42) -> Dict[str, Any]:
    """기본 Config 템플릿 생성"""
    return {
        "project": {
            "name": "ai07_pill_od",
            "run_name": run_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "paths": {k: str(v) for k, v in paths.items()},
        "reproducibility": {
            "seed": seed,
            "deterministic": True,
        },
        "data": {
            "format": "coco_json_multi",
            "max_objects_per_image": 4,
            "num_classes": None,  # 추후 자동 추출
            "class_whitelist": None,  # [1900, 16548, ...] 또는 None (전체 사용)
        },
        "split": {
            "strategy": "stratify_by_num_objects",
            "seed": seed,
            "ratios": {"train": 0.8, "valid": 0.2},
            "kfold": {"enabled": False, "n_splits": 5, "fold_idx": 0},
        },
        "train": {
            "framework": "ultralytics_yolo",
            # Model
            "model_name": "yolov8s",
            "imgsz": 768,
            "pretrained": True,
            # Training hyperparameters
            "epochs": 80,
            "batch": 8,
            "workers": 4,
            "device": "0",
            # Optimizer
            "optimizer": "auto",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            # Scheduler
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            # Augmentation
            "augment": True,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            # Loss weights
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            # Checkpointing
            "save": True,
            "save_period": -1,
            "patience": 50,
            # Misc
            "verbose": True,
            "plots": True,
        },
        "infer": {
            "conf_thr": 0.001,
            "nms_iou_thr": 0.5,
            "max_det_per_image": 4,
            "tta": {"enabled": False},
        },
        "postprocess": {
            "strategy": "topk_by_score",
            "topk": 4,
            "classwise_threshold": None,  # {class_id: thr} 또는 None
            "clip_boxes": True,
        },
        "submission": {
            "columns": ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"],
            "image_id_rule": "file_stem_int",
            "annotation_id_rule": "unique_row_id",
            "bbox_format": "xywh_abs",
        },
        "notes": "",
    }


# ============================================================
# 4. Logging & Experiment Tracking
# ============================================================

def save_json(path: Path, obj: Dict[str, Any]):
    """JSON 저장 (pretty print)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """JSON 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, obj: Dict[str, Any]):
    """JSONL 형식으로 append (실험 로그용)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_command(cmd: List[str]) -> Optional[str]:
    """외부 명령 실행 (Git 등)"""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        return out
    except Exception:
        return None


def get_git_info() -> Dict[str, Any]:
    """현재 Git 상태 정보 수집"""
    return {
        "git_head": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run_command(["git", "status", "--porcelain"])),
    }


def create_run_manifest(run_name: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    """실험 실행 Manifest 생성"""
    git_info = get_git_info()
    
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths_meta": str(paths.get("CONFIG", Path(".")) / "paths_meta.json"),
        "env_meta": str(paths.get("CONFIG", Path(".")) / "env_meta.json"),
        "config": str(paths.get("CONFIG", Path(".")) / "config.json"),
        "git": git_info,
    }
    
    return manifest


def init_experiment_registry(runs_dir: Path):
    """실험 레지스트리 CSV 초기화 (헤더만)"""
    reg_path = runs_dir / "_registry.csv"
    if reg_path.exists():
        return reg_path
    
    with open(reg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "created_at", "run_dir", "config_path", "git_head", "notes"])
    
    return reg_path


def register_experiment(
    runs_dir: Path,
    run_name: str,
    run_dir: Path,
    config_path: Path,
    git_head: Optional[str] = None,
    notes: str = "",
):
    """실험 레지스트리에 등록"""
    reg_path = init_experiment_registry(runs_dir)
    
    with open(reg_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            run_name,
            datetime.now().isoformat(timespec="seconds"),
            str(run_dir),
            str(config_path),
            git_head or "",
            notes,
        ])


# ============================================================
# 5. Results Recording (Metrics Tracking)
# ============================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """중첩 dict를 flat하게 변환 (예: {"a": {"b": 1}} -> {"a.b": 1})"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def safe_scalar(x: Any) -> Any:
    """CSV/JSON에 넣기 안전한 형태로 변환"""
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def init_results_table(results_csv: Path, config: Dict[str, Any]):
    """결과 테이블 초기화 (헤더 생성)"""
    if results_csv.exists():
        return
    
    base_cols = [
        "ts", "run_name", "result_name", "stage", "notes",
    ]
    
    # Config에서 자주 비교할 필드만 추출
    cfg_flat = flatten_dict(config)
    cfg_cols = [
        "reproducibility.seed",
        "split.strategy",
        "split.ratios.train",
        "split.ratios.valid",
        "train.model.name",
        "train.model.imgsz",
        "train.hyperparams.epochs",
        "train.hyperparams.batch",
        "infer.conf_thr",
        "infer.nms_iou_thr",
        "postprocess.topk",
    ]
    
    metric_cols = [
        "mAP_75_95",  # 대회 주요 지표
        "mAP_50",
        "mAP_75",
        "mean_IoU_TP",
        "precision",
        "recall",
    ]
    
    extra_cols = [
        "cfg_path", "run_dir", "ckpt_dir", "submission_path",
    ]
    
    header = base_cols + cfg_cols + metric_cols + extra_cols
    
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)


def record_result(
    results_csv: Path,
    results_jsonl: Path,
    run_name: str,
    result_name: str,
    stage: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, Path]] = None,
    notes: str = "",
    submission_path: Optional[Path] = None,
):
    """
    실험 결과 기록
    
    Args:
        results_csv: results.csv 경로
        results_jsonl: results.jsonl 경로
        run_name: 실험명
        result_name: 결과 별칭 (예: "baseline_v1")
        stage: "val" / "oof" / "public_lb" / "private_lb"
        config: 실험 config dict
        metrics: 평가 지표 dict
        paths: 경로 dict
        notes: 메모
        submission_path: 제출 파일 경로
    """
    ts = datetime.now().isoformat(timespec="seconds")
    metrics = metrics or {}
    metrics = {str(k): safe_scalar(v) for k, v in metrics.items()}
    paths = paths or {}
    
    # CSV용 row 생성
    cfg_flat = flatten_dict(config)
    
    def get_metric(key: str, default=None):
        return metrics.get(key, default)
    
    row = {
        "ts": ts,
        "run_name": run_name,
        "result_name": result_name,
        "stage": stage,
        "notes": notes,
        # Config 필드
        "reproducibility.seed": cfg_flat.get("reproducibility.seed"),
        "split.strategy": cfg_flat.get("split.strategy"),
        "split.ratios.train": cfg_flat.get("split.ratios.train"),
        "split.ratios.valid": cfg_flat.get("split.ratios.valid"),
        "train.model.name": cfg_flat.get("train.model_name", cfg_flat.get("train.model.name")),
        "train.model.imgsz": cfg_flat.get("train.imgsz", cfg_flat.get("train.model.imgsz")),
        "train.hyperparams.epochs": cfg_flat.get("train.epochs", cfg_flat.get("train.hyperparams.epochs")),
        "train.hyperparams.batch": cfg_flat.get("train.batch", cfg_flat.get("train.hyperparams.batch")),
        "infer.conf_thr": cfg_flat.get("infer.conf_thr"),
        "infer.nms_iou_thr": cfg_flat.get("infer.nms_iou_thr"),
        "postprocess.topk": cfg_flat.get("postprocess.topk"),
        # Metrics
        "mAP_75_95": get_metric("mAP_75_95", get_metric("mAP@[0.75:0.95]")),
        "mAP_50": get_metric("mAP_50", get_metric("mAP@0.5")),
        "mAP_75": get_metric("mAP_75", get_metric("mAP@0.75")),
        "mean_IoU_TP": get_metric("mean_IoU_TP"),
        "precision": get_metric("precision"),
        "recall": get_metric("recall"),
        # Paths
        "cfg_path": str(paths.get("CONFIG", "")) if paths.get("CONFIG") else "",
        "run_dir": str(paths.get("RUN_DIR", "")) if paths.get("RUN_DIR") else "",
        "ckpt_dir": str(paths.get("CKPT", "")) if paths.get("CKPT") else "",
        "submission_path": str(submission_path) if submission_path else "",
    }
    
    # CSV append
    init_results_table(results_csv, config)
    with open(results_csv, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: safe_scalar(row.get(k, "")) for k in header})
    
    # JSONL (전체 정보 보존)
    full_record = {
        "ts": ts,
        "run_name": run_name,
        "result_name": result_name,
        "stage": stage,
        "notes": notes,
        "config": config,
        "metrics": metrics,
        "paths": {k: str(v) for k, v in paths.items()} if paths else {},
        "submission_path": str(submission_path) if submission_path else "",
    }
    append_jsonl(results_jsonl, full_record)
    
    print(f"[OK] Result recorded: {result_name} ({stage}) -> {results_csv.name}")


# ============================================================
# 6. Helper Functions
# ============================================================

def print_section(title: str, width: int = 60):
    """섹션 제목 출력 (가독성)"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_dict(d: Dict[str, Any], indent: int = 0):
    """Dict를 보기 좋게 출력"""
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"- {k}:")
            print_dict(v, indent + 1)
        else:
            print("  " * indent + f"- {k}: {v}")
