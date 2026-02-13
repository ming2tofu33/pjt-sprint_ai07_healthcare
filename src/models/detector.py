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
import csv
import shutil
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from src.utils.logger import get_logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


def _parse_int_list(value: Any) -> list[int]:
    """Normalize an int-list-like value into a sorted unique int list."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return sorted({int(v) for v in items})
    if isinstance(value, (list, tuple, set)):
        return sorted({int(v) for v in value})
    return [int(value)]


def _load_idx_to_category_from_data_yaml(data_yaml: Path) -> dict[int, int]:
    """Read YOLO data.yaml names and build class-index -> category_id mapping."""
    if not data_yaml.exists():
        return {}

    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    names = data.get("names")
    mapping: dict[int, int] = {}

    if isinstance(names, dict):
        for key, value in names.items():
            try:
                idx = int(key)
                cat = int(value)
            except Exception:
                continue
            mapping[idx] = cat
        return mapping

    if isinstance(names, list):
        for idx, value in enumerate(names):
            try:
                cat = int(value)
            except Exception:
                continue
            mapping[idx] = cat
        return mapping

    return {}


def _safe_float(value: Any) -> float | None:
    """다양한 숫자 타입(torch/tensor 포함)을 안전하게 float으로 변환한다."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def _is_primary_rank(trainer: Any) -> bool:
    """DDP 환경에서 메인 프로세스(rank -1 또는 0)에서만 작업하도록 판별한다."""
    rank = getattr(trainer, "rank", -1)
    return rank in (-1, 0)


def _is_primary_process(state: Any) -> bool:
    """trainer/validator 객체를 받아 메인 프로세스 여부를 판단한다."""
    rank = getattr(state, "rank", None)
    if rank is None:
        return True
    return rank in (-1, 0)


def _parse_val_batch_index(file_name: str) -> int | None:
    """`val_batch{idx}_*.jpg` 파일명에서 배치 인덱스를 추출한다."""
    match = re.search(r"val_batch(\d+)_", file_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_snapshot_epochs(
    total_epochs: int,
    percent_points: list[int],
    max_epoch_gap: int,
) -> list[int]:
    """하이브리드 규칙(퍼센트 + 최대 간격 보정)으로 스냅샷 epoch 목록을 계산한다."""
    if total_epochs <= 0:
        return []

    base_points: set[int] = {1, total_epochs}
    for point in percent_points:
        try:
            p = int(point)
        except Exception:
            continue
        p = max(0, min(100, p))
        epoch = int(round((p / 100.0) * total_epochs))
        epoch = max(1, min(total_epochs, epoch))
        base_points.add(epoch)

    seeds = sorted(base_points)
    if max_epoch_gap <= 0 or len(seeds) <= 1:
        return seeds

    expanded: list[int] = [seeds[0]]
    for target in seeds[1:]:
        cursor = expanded[-1]
        while target - cursor > max_epoch_gap:
            cursor += max_epoch_gap
            expanded.append(cursor)
        if target != expanded[-1]:
            expanded.append(target)

    return expanded


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
        self.last_train_runtime: dict[str, Any] = {}

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
        self.last_train_runtime = {}

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
            "patience", "workers", "seed", "deterministic", "close_mosaic",
            "pretrained", "save", "save_period", "verbose", "plots",
            "rect",
            # 증강
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr",
            "mosaic", "mixup", "copy_paste",
            # 손실 가중치
            "box", "cls", "dfl",
            # 기타
            "cos_lr", "label_smoothing",
        ]
        for key in _DIRECT_KEYS:
            if key in train_cfg:
                kwargs[key] = train_cfg[key]
        logger.info("train option | rect=%s", kwargs.get("rect", "<ultralytics_default>"))

        # 로그 모드: 기본은 기존 배치 로그(batch) 유지, epoch 모드는 배치 로그를 억제한다.
        log_mode = str(train_cfg.get("log_mode", "batch")).strip().lower()
        if log_mode not in {"batch", "epoch"}:
            log_mode = "batch"
        if log_mode == "epoch":
            kwargs["verbose"] = False

        # Optional class filtering:
        # - classes: class indices
        # - target_category_ids: original category IDs (mapped via data.yaml names)
        classes = _parse_int_list(train_cfg.get("classes"))
        target_category_ids = _parse_int_list(train_cfg.get("target_category_ids"))

        if classes and target_category_ids:
            raise ValueError("Use only one of train.classes or train.target_category_ids.")

        if target_category_ids:
            idx2cat = _load_idx_to_category_from_data_yaml(data_yaml)
            if not idx2cat:
                raise ValueError(
                    f"Could not resolve names mapping from data.yaml for target_category_ids: {data_yaml}"
                )
            cat2idx = {cat: idx for idx, cat in idx2cat.items()}
            mapped = sorted({cat2idx[cid] for cid in target_category_ids if cid in cat2idx})
            if not mapped:
                raise ValueError(
                    "No valid classes mapped from train.target_category_ids against data.yaml names."
                )
            kwargs["classes"] = mapped
        elif classes:
            kwargs["classes"] = classes

        # device 처리 (config 또는 CLI)
        device = train_cfg.get("device")
        if device is not None:
            kwargs["device"] = device

        # 스냅샷 설정
        raw_snapshot_cfg = train_cfg.get("debug_snapshots", {})
        snapshot_cfg = raw_snapshot_cfg if isinstance(raw_snapshot_cfg, dict) else {}
        snapshot_enabled = bool(snapshot_cfg.get("enabled", False))
        snapshot_scope = str(snapshot_cfg.get("scope", "val_only"))
        snapshot_points_raw = snapshot_cfg.get("percent_points", [10, 30, 50, 70, 90, 100])
        if isinstance(snapshot_points_raw, list):
            snapshot_points: list[int] = []
            for value in snapshot_points_raw:
                try:
                    snapshot_points.append(int(value))
                except Exception:
                    continue
            if not snapshot_points:
                snapshot_points = [10, 30, 50, 70, 90, 100]
        else:
            snapshot_points = [10, 30, 50, 70, 90, 100]
        try:
            snapshot_max_gap = int(snapshot_cfg.get("max_epoch_gap", 20))
        except Exception:
            snapshot_max_gap = 20
        snapshot_output_dirname = str(snapshot_cfg.get("output_dirname", "snapshots")).strip() or "snapshots"
        snapshot_index_filename = str(snapshot_cfg.get("index_filename", "snapshot_index.csv")).strip() or "snapshot_index.csv"
        total_epochs = int(train_cfg.get("epochs", kwargs.get("epochs", 100)))
        if total_epochs <= 0:
            total_epochs = 100
        snapshot_targets = _build_snapshot_epochs(total_epochs, snapshot_points, snapshot_max_gap) if snapshot_enabled else []
        snapshot_target_set = set(snapshot_targets)
        snapshot_state: dict[str, Any] = {
            "enabled": snapshot_enabled,
            "scope": snapshot_scope,
            "percent_points": snapshot_points,
            "max_epoch_gap": snapshot_max_gap,
            "scheduled_epochs": snapshot_targets,
            "saved_epochs": [],
            "saved_file_count": 0,
            "output_dir": "",
            "index_file": "",
            "train_artifacts_dir": "",
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "val_plots_forced_epochs": [],
        }

        def _resolve_snapshot_dir(save_dir: Path) -> Path:
            base_dir = save_dir if save_dir.name.lower() == "train" else (save_dir / "train")
            return base_dir / snapshot_output_dirname

        def _write_snapshot_manifest(snapshot_dir: Path, trainer_epochs: int) -> None:
            snapshot_manifest = {
                "enabled": snapshot_state["enabled"],
                "scope": snapshot_state["scope"],
                "percent_points": snapshot_state["percent_points"],
                "max_epoch_gap": snapshot_state["max_epoch_gap"],
                "scheduled_epochs": snapshot_state["scheduled_epochs"],
                "saved_epochs": snapshot_state["saved_epochs"],
                "saved_file_count": snapshot_state["saved_file_count"],
                "index_file": snapshot_state["index_file"],
                "output_dir": str(snapshot_dir),
                "total_epochs": int(trainer_epochs),
                "val_plots_forced_epochs": snapshot_state["val_plots_forced_epochs"],
            }
            manifest_path = snapshot_dir / "snapshot_manifest.json"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot_manifest, f, ensure_ascii=False, indent=2)

        def _on_fit_epoch_log(trainer: Any) -> None:
            if not _is_primary_rank(trainer):
                return
            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            trainer_epochs = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
            save_dir_attr = getattr(trainer, "save_dir", None)
            if save_dir_attr is not None:
                snapshot_state["train_artifacts_dir"] = str(Path(save_dir_attr))

            train_loss = getattr(trainer, "tloss", None)
            box_loss = cls_loss = dfl_loss = None
            if train_loss is not None:
                try:
                    values = train_loss.tolist() if hasattr(train_loss, "tolist") else list(train_loss)
                except Exception:
                    values = []
                if len(values) >= 3:
                    box_loss = _safe_float(values[0])
                    cls_loss = _safe_float(values[1])
                    dfl_loss = _safe_float(values[2])

            metrics_dict = getattr(trainer, "metrics", {}) or {}
            precision = _safe_float(metrics_dict.get("metrics/precision(B)"))
            recall = _safe_float(metrics_dict.get("metrics/recall(B)"))
            map50 = _safe_float(metrics_dict.get("metrics/mAP50(B)"))
            map50_95 = _safe_float(metrics_dict.get("metrics/mAP50-95(B)"))

            def _fmt(value: float | None) -> str:
                return "-" if value is None else f"{value:.4f}"

            logger.info(
                "train epoch %d/%d | box=%s cls=%s dfl=%s | P=%s R=%s mAP50=%s mAP50-95=%s",
                current_epoch,
                trainer_epochs,
                _fmt(box_loss),
                _fmt(cls_loss),
                _fmt(dfl_loss),
                _fmt(precision),
                _fmt(recall),
                _fmt(map50),
                _fmt(map50_95),
            )

        def _save_snapshot_from_dir(source_dir: Path, *, current_epoch: int, trainer_epochs: int) -> int:
            if current_epoch not in snapshot_target_set:
                return 0
            if not source_dir.exists():
                return 0

            progress = int(round((current_epoch / max(1, trainer_epochs)) * 100))
            progress = max(0, min(100, progress))

            snapshot_dir = _resolve_snapshot_dir(source_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            index_path = snapshot_dir / snapshot_index_filename
            snapshot_state["output_dir"] = str(snapshot_dir)
            snapshot_state["index_file"] = str(index_path)
            snapshot_state["train_artifacts_dir"] = str(source_dir)

            source_files = sorted(source_dir.glob("val_batch*_labels.jpg")) + sorted(source_dir.glob("val_batch*_pred.jpg"))
            if not source_files:
                return 0

            write_header = not index_path.exists()
            files_written = 0
            with index_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "progress", "batch_idx", "kind", "source_file", "snapshot_file"],
                )
                if write_header:
                    writer.writeheader()

                for src in source_files:
                    file_name = src.name
                    if file_name.endswith("_labels.jpg"):
                        kind = "labels"
                    elif file_name.endswith("_pred.jpg"):
                        kind = "pred"
                    else:
                        continue

                    batch_idx = _parse_val_batch_index(file_name)
                    if batch_idx is None:
                        continue

                    dst_name = f"e{current_epoch:04d}_p{progress:03d}_val_b{batch_idx}_{kind}.jpg"
                    dst = snapshot_dir / dst_name
                    if dst.exists():
                        continue
                    shutil.copy2(str(src), str(dst))
                    files_written += 1

                    writer.writerow(
                        {
                            "epoch": current_epoch,
                            "progress": progress,
                            "batch_idx": batch_idx,
                            "kind": kind,
                            "source_file": str(src),
                            "snapshot_file": str(dst),
                        }
                    )

            if files_written > 0:
                saved_epochs = snapshot_state["saved_epochs"]
                if current_epoch not in saved_epochs:
                    saved_epochs.append(current_epoch)
                    saved_epochs.sort()
                snapshot_state["saved_file_count"] = int(snapshot_state["saved_file_count"]) + files_written
                _write_snapshot_manifest(snapshot_dir, trainer_epochs)

            return files_written

        def _on_train_epoch_end_snapshot(trainer: Any) -> None:
            if not snapshot_enabled or not _is_primary_rank(trainer):
                return
            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            trainer_epochs = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
            snapshot_state["current_epoch"] = current_epoch
            snapshot_state["total_epochs"] = trainer_epochs
            save_dir_attr = getattr(trainer, "save_dir", None)
            if save_dir_attr is not None:
                snapshot_state["train_artifacts_dir"] = str(Path(save_dir_attr))

        def _on_val_start_snapshot(validator: Any) -> None:
            if not snapshot_enabled or not _is_primary_process(validator):
                return
            current_epoch = int(snapshot_state.get("current_epoch", 0))
            if current_epoch not in snapshot_target_set:
                return
            try:
                validator.args.plots = True
            except Exception:
                return
            forced_epochs = snapshot_state["val_plots_forced_epochs"]
            if current_epoch not in forced_epochs:
                forced_epochs.append(current_epoch)
                forced_epochs.sort()

        def _on_val_end_snapshot(validator: Any) -> None:
            if not snapshot_enabled or not _is_primary_process(validator):
                return
            current_epoch = int(snapshot_state.get("current_epoch", 0))
            trainer_epochs = int(snapshot_state.get("total_epochs", total_epochs))
            if current_epoch not in snapshot_target_set:
                return
            save_dir_attr = getattr(validator, "save_dir", None)
            if save_dir_attr is None:
                return
            save_dir = Path(save_dir_attr)
            _save_snapshot_from_dir(save_dir, current_epoch=current_epoch, trainer_epochs=trainer_epochs)

        def _on_train_end(trainer: Any) -> None:
            if not _is_primary_rank(trainer):
                return
            save_dir_attr = getattr(trainer, "save_dir", None)
            if save_dir_attr is not None:
                save_dir = Path(save_dir_attr)
                snapshot_state["train_artifacts_dir"] = str(save_dir)
                if snapshot_enabled:
                    snapshot_dir = _resolve_snapshot_dir(save_dir)
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    if not snapshot_state["output_dir"]:
                        snapshot_state["output_dir"] = str(snapshot_dir)
                    if not snapshot_state["index_file"]:
                        snapshot_state["index_file"] = str(snapshot_dir / snapshot_index_filename)
                    trainer_epochs = int(getattr(trainer, "epochs", total_epochs) or total_epochs)
                    _write_snapshot_manifest(snapshot_dir, trainer_epochs)

        if log_mode == "epoch":
            self.model.add_callback("on_fit_epoch_end", _on_fit_epoch_log)
        if snapshot_enabled:
            self.model.add_callback("on_train_epoch_end", _on_train_epoch_end_snapshot)
            self.model.add_callback("on_val_start", _on_val_start_snapshot)
            self.model.add_callback("on_val_end", _on_val_end_snapshot)
        self.model.add_callback("on_train_end", _on_train_end)

        results = self.model.train(**kwargs)

        default_train_dir = Path(str(project)) / str(name)
        train_artifacts_dir = snapshot_state["train_artifacts_dir"] or str(default_train_dir)
        self.last_train_runtime = {
            "log_mode": log_mode,
            "train_artifacts_dir": train_artifacts_dir,
            "debug_snapshots": {
                "enabled": snapshot_state["enabled"],
                "scope": snapshot_state["scope"],
                "percent_points": snapshot_state["percent_points"],
                "max_epoch_gap": snapshot_state["max_epoch_gap"],
                "scheduled_epochs": snapshot_state["scheduled_epochs"],
                "saved_epochs": snapshot_state["saved_epochs"],
                "saved_file_count": snapshot_state["saved_file_count"],
                "val_plots_forced_epochs": snapshot_state["val_plots_forced_epochs"],
                "output_dir": snapshot_state["output_dir"],
                "index_file": snapshot_state["index_file"],
            },
        }

        return results

    # ── 평가 ────────────────────────────────────────────────

    def validate(
        self,
        data_yaml: Path,
        *,
        config: dict | None = None,
        eval_overrides: dict[str, Any] | None = None,
        project: str | Path | None = None,
        name: str | None = None,
        exist_ok: bool | None = None,
    ) -> dict:
        """Ultralytics validation 을 실행하고 메트릭 dict 를 반환한다.

        필요 시 ``project/name/exist_ok`` 를 전달해 평가 산출물 저장 경로를
        명시적으로 제어할 수 있다.

        Returns
        -------
        dict
            ``{mAP50, mAP50_95, mAP75_95, precision, recall, per_class}``

            ``mAP75_95`` 는 대회 평가 지표 mAP@[0.75:0.95] 이다.
        """
        eval_cfg = dict(config.get("evaluate", {})) if config else {}
        if eval_overrides:
            eval_cfg.update(eval_overrides)

        kwargs: dict[str, Any] = {"data": str(data_yaml)}
        if "conf" in eval_cfg:
            kwargs["conf"] = float(eval_cfg["conf"])
        # NOTE: Ultralytics val() 의 iou 파라미터는 NMS IoU threshold 이다.
        # mAP 계산 IoU 범위(0.50~0.95)와는 무관하다.
        if "nms_iou" in eval_cfg:
            kwargs["iou"] = float(eval_cfg["nms_iou"])
        if "device" in eval_cfg:
            kwargs["device"] = eval_cfg["device"]
        if "augment" in eval_cfg:
            kwargs["augment"] = bool(eval_cfg["augment"])
        if project is not None:
            kwargs["project"] = str(project)
        if name is not None:
            kwargs["name"] = name
        if exist_ok is not None:
            kwargs["exist_ok"] = bool(exist_ok)

        results = self.model.val(**kwargs)

        # Ultralytics results 객체에서 메트릭 추출
        metrics = extract_metrics(results)
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
        augment: bool = False,
    ) -> list:
        """배치 추론을 실행하고 결과 리스트를 반환한다.

        Parameters
        ----------
        augment : bool
            True 이면 TTA(Test-Time Augmentation)를 적용한다.

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
        if augment:
            kwargs["augment"] = True

        return self.model.predict(**kwargs)


# ── 유틸리티 ────────────────────────────────────────────────

def extract_metrics(results: Any, *, results_csv_path: Path | None = None) -> dict:
    """Ultralytics 결과 객체에서 메트릭을 추출한다.

    기본 경로는 ``results.box``이며, 실패 시 ``results.results_dict``와
    (선택적으로) ``results.csv``를 순차 fallback으로 사용한다.
    """
    metrics: dict[str, Any] = {}

    try:
        box = results.box
        metrics["mAP50"] = float(box.map50)
        metrics["mAP50_95"] = float(box.map)
        metrics["precision"] = float(box.mp)
        metrics["recall"] = float(box.mr)

        # all_ap: shape (nc, 10), index 5~9 => IoU 0.75~0.95
        all_ap = None
        if hasattr(box, "all_ap") and box.all_ap is not None:
            all_ap = np.array(box.all_ap)
        elif hasattr(box, "ap_class_index") and hasattr(box, "ap"):
            ap_arr = np.array(box.ap) if box.ap is not None else None
            if ap_arr is not None and ap_arr.ndim == 2 and ap_arr.shape[1] == 10:
                all_ap = ap_arr

        if all_ap is not None and all_ap.ndim == 2 and all_ap.shape[1] >= 10:
            metrics["mAP75_95"] = float(all_ap[:, 5:].mean())
            metrics["per_class_mAP75_95"] = [float(v) for v in all_ap[:, 5:].mean(axis=1)]
        else:
            metrics["mAP75_95"] = None
            metrics["metric_source"] = "results.box_without_all_ap"

        if hasattr(box, "ap50") and box.ap50 is not None:
            metrics["per_class_ap50"] = [float(v) for v in box.ap50]
        if hasattr(box, "ap") and box.ap is not None:
            ap_arr = np.array(box.ap)
            if ap_arr.ndim == 1:
                metrics["per_class_ap50_95"] = [float(v) for v in ap_arr]
            elif ap_arr.ndim == 2:
                metrics["per_class_ap50_95"] = [float(v) for v in ap_arr.mean(axis=1)]

        metrics["metric_source"] = metrics.get("metric_source", "results.box")
        return metrics
    except Exception:
        pass

    try:
        rd = results.results_dict
        metrics["mAP50"] = float(rd.get("metrics/mAP50(B)", 0))
        metrics["mAP50_95"] = float(rd.get("metrics/mAP50-95(B)", 0))
        metrics["precision"] = float(rd.get("metrics/precision(B)", 0))
        metrics["recall"] = float(rd.get("metrics/recall(B)", 0))
        metrics["mAP75_95"] = None
        metrics["metric_source"] = "results.results_dict"
    except Exception:
        pass

    if results_csv_path is not None and results_csv_path.exists():
        try:
            with results_csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                for col in last:
                    key = col.strip()
                    if "mAP50(B)" in key and "95" not in key:
                        metrics["mAP50"] = float(last[col])
                    elif "mAP50-95(B)" in key:
                        metrics["mAP50_95"] = float(last[col])
                    elif "precision(B)" in key:
                        metrics["precision"] = float(last[col])
                    elif "recall(B)" in key:
                        metrics["recall"] = float(last[col])
                metrics.setdefault("mAP75_95", None)
                metrics["metric_source"] = "results.csv"
        except Exception:
            pass

    return metrics


def _extract_metrics(results: Any) -> dict:
    """Backward-compatible alias for internal callers."""
    return extract_metrics(results)


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
