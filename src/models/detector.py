"""src.models.detector -- Ultralytics YOLO 모델 래퍼.

학습/평가/추론을 config 기반으로 실행하며,
Ultralytics API 의 차이를 흡수하는 얇은 래퍼 역할을 한다.

사용 예시::

    from src.models.detector import PillDetector

    det = PillDetector.from_config(config)
    results = det.train(data_yaml=Path("data.yaml"))
    metrics = det.validate(data_yaml=Path("data.yaml"))
    preds   = det.predict(source=Path("images/"))
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]


class PillDetector:
    """Ultralytics YOLO 모델을 감싼 래퍼.

    Parameters
    ----------
    model_path : str | Path
        사전학습 가중치 경로 또는 모델 이름 (``"yolo11s.pt"`` 등).
    """

    def __init__(self, model_path: str | Path) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics 패키지가 설치되어 있지 않습니다. "
                "pip install ultralytics 로 설치하세요."
            )
        self.model_path = str(model_path)
        self.model: YOLO = YOLO(self.model_path)

    # ── 팩토리 ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "PillDetector":
        """config 의 ``model.pretrained`` 로부터 인스턴스를 생성한다."""
        model_cfg = config.get("model", {})
        pretrained = model_cfg.get("pretrained", "yolo11s.pt")
        return cls(pretrained)

    @classmethod
    def from_weights(cls, weights_path: Path) -> "PillDetector":
        """학습 완료 가중치(.pt)로부터 인스턴스를 생성한다."""
        if not weights_path.exists():
            raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {weights_path}")
        return cls(weights_path)

    # ── 학습 ────────────────────────────────────────────────

    def train(
        self,
        data_yaml: Path,
        *,
        project: str | Path = "runs",
        name: str = "train",
        config: dict | None = None,
    ) -> Any:
        """YOLO 학습을 실행한다.

        Parameters
        ----------
        data_yaml : Path
            YOLO ``data.yaml`` 경로.
        project : str | Path
            Ultralytics 가 결과를 저장할 상위 디렉터리.
        name : str
            ``project`` 하위의 실험 이름.
        config : dict, optional
            ``train`` 섹션 config. Ultralytics ``model.train()`` 에 전달.

        Returns
        -------
        ultralytics.engine.results.Results
            학습 결과 객체.
        """
        train_cfg = dict(config.get("train", {})) if config else {}

        # config 에서 직접 전달할 학습 하이퍼파라미터 추출
        kwargs: dict[str, Any] = {}

        # 핵심 파라미터
        kwargs["data"] = str(data_yaml)
        kwargs["project"] = str(project)
        kwargs["name"] = name
        kwargs["exist_ok"] = True  # 이름 충돌 시 덮어쓰기

        # train 섹션의 키를 Ultralytics 인자로 매핑
        _DIRECT_KEYS = [
            "epochs", "imgsz", "batch", "lr0", "lrf", "optimizer",
            "patience", "workers", "seed", "deterministic",
            "pretrained", "save", "save_period", "verbose", "plots",
            # 증강
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr",
            "mosaic", "mixup", "copy_paste",
            # 손실 가중치
            "box", "cls", "dfl",
        ]
        for key in _DIRECT_KEYS:
            if key in train_cfg:
                kwargs[key] = train_cfg[key]

        # device 처리 (config 또는 CLI)
        device = train_cfg.get("device")
        if device is not None:
            kwargs["device"] = device

        results = self.model.train(**kwargs)
        return results

    # ── 평가 ────────────────────────────────────────────────

    def validate(
        self,
        data_yaml: Path,
        *,
        config: dict | None = None,
    ) -> dict:
        """Ultralytics validation 을 실행하고 메트릭 dict 를 반환한다.

        Returns
        -------
        dict
            ``{mAP50, mAP50_95, precision, recall, per_class}``
        """
        eval_cfg = dict(config.get("evaluate", {})) if config else {}

        kwargs: dict[str, Any] = {"data": str(data_yaml)}
        if "conf" in eval_cfg:
            kwargs["conf"] = float(eval_cfg["conf"])
        if "iou" in eval_cfg:
            kwargs["iou"] = float(eval_cfg["iou"])
        if "device" in eval_cfg:
            kwargs["device"] = eval_cfg["device"]

        results = self.model.val(**kwargs)

        # Ultralytics results 객체에서 메트릭 추출
        metrics = _extract_metrics(results)
        return metrics

    # ── 추론 ────────────────────────────────────────────────

    def predict(
        self,
        source: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 300,
        device: Any = None,
        imgsz: int = 640,
        save: bool = False,
        verbose: bool = False,
    ) -> list:
        """배치 추론을 실행하고 결과 리스트를 반환한다.

        Returns
        -------
        list
            Ultralytics ``Results`` 객체 리스트.
        """
        kwargs: dict[str, Any] = {
            "source": str(source),
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "imgsz": imgsz,
            "save": save,
            "verbose": verbose,
        }
        if device is not None:
            kwargs["device"] = device

        return self.model.predict(**kwargs)


# ── 유틸리티 ────────────────────────────────────────────────

def _extract_metrics(results: Any) -> dict:
    """Ultralytics validation 결과에서 메트릭 dict 를 추출한다."""
    metrics: dict[str, Any] = {}

    try:
        box = results.box
        metrics["mAP50"] = float(box.map50)
        metrics["mAP50_95"] = float(box.map)
        metrics["precision"] = float(box.mp)
        metrics["recall"] = float(box.mr)

        # 클래스별 AP (가용 시)
        if hasattr(box, "ap50") and box.ap50 is not None:
            metrics["per_class_ap50"] = [float(v) for v in box.ap50]
        if hasattr(box, "ap") and box.ap is not None:
            metrics["per_class_ap50_95"] = [float(v) for v in box.ap]
    except Exception:
        # Ultralytics 버전 차이 대응
        try:
            metrics["mAP50"] = float(results.results_dict.get("metrics/mAP50(B)", 0))
            metrics["mAP50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0))
        except Exception:
            pass

    return metrics


def save_config_resolved(config: dict, run_dir: Path) -> Path:
    """병합 config 를 ``config_resolved.yaml`` 로 저장한다."""
    out = run_dir / "config_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out


def save_metrics(metrics: dict, run_dir: Path, filename: str = "metrics.json") -> Path:
    """메트릭 dict 를 JSON 으로 저장한다."""
    out = run_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out


def copy_best_weights(
    train_output_dir: Path,
    *,
    run_dir: Path,
    best_models_dir: Path,
    run_name: str,
) -> tuple[Path | None, Path | None]:
    """학습 완료 후 best.pt 를 runs/<run>/ 와 artifacts/best_models/ 에 복사한다.

    Returns
    -------
    tuple[Path | None, Path | None]
        ``(run_best_path, artifact_best_path)``
    """
    # Ultralytics 가 저장하는 best.pt 경로 탐색
    candidates = [
        train_output_dir / "weights" / "best.pt",
        train_output_dir / "best.pt",
    ]
    src_best: Path | None = None
    for c in candidates:
        if c.exists():
            src_best = c
            break

    if src_best is None:
        return None, None

    # runs/<run_name>/weights/best.pt
    run_weights = run_dir / "weights"
    run_weights.mkdir(parents=True, exist_ok=True)
    run_best = run_weights / "best.pt"

    # Ultralytics 가 이미 run_dir 에 직접 저장했을 경우 same-file 방지
    if src_best.resolve() != run_best.resolve():
        shutil.copy2(str(src_best), str(run_best))

    # last.pt 도 복사 (있으면)
    src_last = src_best.parent / "last.pt"
    dst_last = run_weights / "last.pt"
    if src_last.exists() and src_last.resolve() != dst_last.resolve():
        shutil.copy2(str(src_last), str(dst_last))

    # artifacts/best_models/<run_name>_best.pt
    best_models_dir.mkdir(parents=True, exist_ok=True)
    artifact_best = best_models_dir / f"{run_name}_best.pt"
    shutil.copy2(str(src_best), str(artifact_best))

    return run_best, artifact_best
