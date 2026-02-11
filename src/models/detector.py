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

import os
import json
import csv
import shutil
import re
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def build_hybrid_snapshot_epochs(
    total_epochs: int,
    percent_points: list[int] | tuple[int, ...],
    max_epoch_gap: int,
) -> list[int]:
    """Build a sorted unique epoch list from percent points + fixed max gap.

    Rules:
    1) convert percent points into epoch numbers
    2) if interval between selected epochs is larger than max_epoch_gap,
       fill intermediate epochs by max_epoch_gap
    3) always include epoch 1 and last epoch
    """
    if total_epochs <= 0:
        return []

    normalized_points: list[int] = []
    for p in percent_points:
        try:
            val = int(p)
        except Exception:
            continue
        if 1 <= val <= 100:
            normalized_points.append(val)
    if not normalized_points:
        normalized_points = [10, 30, 50, 70, 90, 100]

    selected: set[int] = {1, total_epochs}
    for p in normalized_points:
        epoch = int(round(total_epochs * (p / 100.0)))
        epoch = min(max(epoch, 1), total_epochs)
        selected.add(epoch)

    out = sorted(selected)
    gap = int(max_epoch_gap) if int(max_epoch_gap) > 0 else 20
    dense: list[int] = []
    for i, current in enumerate(out):
        dense.append(current)
        if i == len(out) - 1:
            break
        nxt = out[i + 1]
        probe = current + gap
        while probe < nxt:
            dense.append(probe)
            probe += gap

    return sorted(set(dense))


def _parse_val_snapshot_name(filename: str) -> tuple[int, str] | None:
    """Parse val batch file name and return (batch_idx, kind)."""
    m = re.match(r"^val_batch(\d+)_(labels|pred)\.jpg$", filename)
    if m is None:
        return None
    return int(m.group(1)), str(m.group(2))


def _ensure_snapshot_index_header(index_path: Path) -> None:
    if index_path.exists():
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "progress_pct",
                "batch_idx",
                "kind",
                "source_file",
                "snapshot_file",
            ],
        )
        writer.writeheader()


def _load_existing_snapshot_files(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    try:
        with index_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {str(r.get("snapshot_file", "")).strip() for r in rows if r.get("snapshot_file")}
    except Exception:
        return set()


def _append_snapshot_index_row(index_path: Path, row: dict[str, Any]) -> None:
    _ensure_snapshot_index_header(index_path)
    with index_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "progress_pct",
                "batch_idx",
                "kind",
                "source_file",
                "snapshot_file",
            ],
        )
        writer.writerow(row)


def _write_snapshot_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
        log_mode = str(train_cfg.get("log_mode", "epoch")).strip().lower()
        use_epoch_log = log_mode == "epoch"

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
        if use_epoch_log:
            # 배치 단위 tqdm 출력을 끄고 에폭 단위 요약 로그를 사용한다.
            kwargs["verbose"] = False

        # device 처리 (config 또는 CLI)
        device = train_cfg.get("device")
        if device is not None:
            kwargs["device"] = device

        debug_snap_cfg = train_cfg.get("debug_snapshots", {})
        if not isinstance(debug_snap_cfg, dict):
            debug_snap_cfg = {}

        snapshot_enabled = bool(debug_snap_cfg.get("enabled", True))
        snapshot_scope = str(debug_snap_cfg.get("scope", "val_only")).strip().lower() or "val_only"
        snapshot_percent_points_raw = debug_snap_cfg.get("percent_points", [10, 30, 50, 70, 90, 100])
        if not isinstance(snapshot_percent_points_raw, (list, tuple)):
            snapshot_percent_points_raw = [10, 30, 50, 70, 90, 100]
        snapshot_percent_points: list[int] = []
        for item in snapshot_percent_points_raw:
            try:
                snapshot_percent_points.append(int(item))
            except Exception:
                continue
        try:
            snapshot_max_epoch_gap = int(debug_snap_cfg.get("max_epoch_gap", 20))
        except Exception:
            snapshot_max_epoch_gap = 20
        snapshot_output_dirname = str(debug_snap_cfg.get("output_dirname", "snapshots")).strip() or "snapshots"
        snapshot_index_filename = str(debug_snap_cfg.get("index_filename", "snapshot_index.csv")).strip() or "snapshot_index.csv"

        snapshot_runtime: dict[str, Any] = {
            "initialized": False,
            "saved_epochs": set(),
            "saved_files": 0,
            "known_snapshot_files": set(),
            "target_epochs": [],
            "target_epochs_set": set(),
            "snapshot_dir": None,
            "index_path": None,
            "manifest_path": None,
            "scope": snapshot_scope,
            "percent_points": snapshot_percent_points,
            "max_epoch_gap": snapshot_max_epoch_gap,
            "output_dirname": snapshot_output_dirname,
            "index_filename": snapshot_index_filename,
            "enabled": snapshot_enabled,
        }

        def _fmt_metric(value: Any) -> str:
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.4f}"
            except Exception:
                return "N/A"

        def _epoch_progress(trainer: Any) -> None:
            total = int(getattr(trainer, "epochs", train_cfg.get("epochs", 0)) or 0)
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            if total <= 0:
                return
            ratio = min(max(epoch / total, 0.0), 1.0)
            width = 20
            filled = int(ratio * width)
            bar = ("#" * filled) + ("-" * (width - filled))
            metrics = getattr(trainer, "metrics", {}) or {}
            msg = (
                f"[TRAIN] {epoch:>3}/{total:<3} {ratio*100:5.1f}% |{bar}| "
                f"mAP50={_fmt_metric(metrics.get('metrics/mAP50(B)'))} "
                f"mAP50-95={_fmt_metric(metrics.get('metrics/mAP50-95(B)'))} "
                f"P={_fmt_metric(metrics.get('metrics/precision(B)'))} "
                f"R={_fmt_metric(metrics.get('metrics/recall(B)'))}"
            )
            print(msg, flush=True)

        def _capture_debug_snapshots(trainer: Any) -> None:
            if not snapshot_runtime.get("enabled", False):
                return
            if str(snapshot_runtime.get("scope", "val_only")) != "val_only":
                return

            rank_raw = getattr(trainer, "rank", os.getenv("RANK", "-1"))
            try:
                rank = int(rank_raw)
            except Exception:
                rank = -1
            if rank not in (-1, 0):
                return

            total = int(getattr(trainer, "epochs", train_cfg.get("epochs", 0)) or 0)
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            if total <= 0 or epoch <= 0:
                return

            save_dir = Path(getattr(trainer, "save_dir", Path(project) / name))

            if not snapshot_runtime["initialized"]:
                target_epochs = build_hybrid_snapshot_epochs(
                    total,
                    snapshot_runtime.get("percent_points", [10, 30, 50, 70, 90, 100]),
                    int(snapshot_runtime.get("max_epoch_gap", 20)),
                )
                snapshot_dir = save_dir / str(snapshot_runtime.get("output_dirname", "snapshots"))
                index_path = snapshot_dir / str(snapshot_runtime.get("index_filename", "snapshot_index.csv"))
                manifest_path = snapshot_dir / "snapshot_manifest.json"

                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_runtime["target_epochs"] = target_epochs
                snapshot_runtime["target_epochs_set"] = set(target_epochs)
                snapshot_runtime["snapshot_dir"] = snapshot_dir
                snapshot_runtime["index_path"] = index_path
                snapshot_runtime["manifest_path"] = manifest_path
                snapshot_runtime["known_snapshot_files"] = _load_existing_snapshot_files(index_path)
                snapshot_runtime["saved_epochs"] = set()
                snapshot_runtime["saved_files"] = 0
                snapshot_runtime["initialized"] = True

            target_epochs_set = snapshot_runtime.get("target_epochs_set", set())
            if epoch not in target_epochs_set:
                return

            snapshot_dir = snapshot_runtime["snapshot_dir"]
            index_path = snapshot_runtime["index_path"]
            manifest_path = snapshot_runtime["manifest_path"]
            known_snapshot_files = snapshot_runtime["known_snapshot_files"]

            progress_pct = int(round((epoch / total) * 100))
            source_files = sorted(save_dir.glob("val_batch*_labels.jpg")) + sorted(save_dir.glob("val_batch*_pred.jpg"))
            saved_now = 0
            for src in source_files:
                parsed = _parse_val_snapshot_name(src.name)
                if parsed is None:
                    continue
                batch_idx, kind = parsed
                snapshot_name = f"e{epoch:04d}_p{progress_pct:03d}_val_b{batch_idx}_{kind}.jpg"
                if snapshot_name in known_snapshot_files:
                    continue

                dst = snapshot_dir / snapshot_name
                shutil.copy2(str(src), str(dst))
                known_snapshot_files.add(snapshot_name)
                _append_snapshot_index_row(
                    index_path,
                    {
                        "epoch": epoch,
                        "progress_pct": progress_pct,
                        "batch_idx": batch_idx,
                        "kind": kind,
                        "source_file": src.name,
                        "snapshot_file": snapshot_name,
                    },
                )
                saved_now += 1

            if saved_now > 0:
                saved_epochs = snapshot_runtime["saved_epochs"]
                saved_epochs.add(epoch)
                snapshot_runtime["saved_files"] = int(snapshot_runtime["saved_files"]) + saved_now
                logger.info(
                    "debug snapshots 저장 | epoch=%d/%d | files=%d | dir=%s",
                    epoch,
                    total,
                    saved_now,
                    snapshot_dir,
                )

            _write_snapshot_manifest(
                manifest_path,
                {
                    "enabled": bool(snapshot_runtime.get("enabled", False)),
                    "scope": str(snapshot_runtime.get("scope", "val_only")),
                    "percent_points": list(snapshot_runtime.get("percent_points", [])),
                    "max_epoch_gap": int(snapshot_runtime.get("max_epoch_gap", 20)),
                    "target_epochs": list(snapshot_runtime.get("target_epochs", [])),
                    "saved_epochs": sorted(list(snapshot_runtime.get("saved_epochs", set()))),
                    "saved_files": int(snapshot_runtime.get("saved_files", 0)),
                    "snapshot_dir": str(snapshot_dir),
                    "index_file": str(index_path),
                },
            )

        prev_yolo_verbose = os.getenv("YOLO_VERBOSE")
        prev_ultra_verbose: Optional[bool] = None
        try:
            if use_epoch_log:
                # Ultralytics 내부 tqdm 비활성화(VERBOSE=False)로 배치 로그 폭주를 막는다.
                os.environ["YOLO_VERBOSE"] = "False"
                try:
                    import ultralytics.utils as ultra_utils

                    prev_ultra_verbose = bool(getattr(ultra_utils, "VERBOSE", True))
                    ultra_utils.VERBOSE = False
                except Exception:
                    prev_ultra_verbose = None
                self.model.add_callback("on_fit_epoch_end", _epoch_progress)
            if snapshot_enabled:
                self.model.add_callback("on_fit_epoch_end", _capture_debug_snapshots)

            results = self.model.train(**kwargs)
            return results
        finally:
            if use_epoch_log:
                if prev_ultra_verbose is not None:
                    try:
                        import ultralytics.utils as ultra_utils

                        ultra_utils.VERBOSE = prev_ultra_verbose
                    except Exception:
                        pass
                if prev_yolo_verbose is None:
                    os.environ.pop("YOLO_VERBOSE", None)
                else:
                    os.environ["YOLO_VERBOSE"] = prev_yolo_verbose

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
