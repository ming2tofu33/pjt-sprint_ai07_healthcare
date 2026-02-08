#!/usr/bin/env python3
"""
Train YOLO with Ultralytics Python API.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


DEFAULT_DATA = "data/processed/yolo/data.yaml"
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_PROJECT = "runs/detect"
DEFAULT_DEVICE = "0"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_abs_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return root / path


def to_repo_display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def choose_run_name(name: str, epochs: int, batch: int, imgsz: int) -> str:
    if name.strip().lower() == "auto":
        return f"e{epochs}_b{batch}_img{imgsz}"
    return name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def resolve_model_source(model_arg: str, root: Path) -> str:
    model_path = to_abs_path(model_arg, root)
    if model_path.exists():
        return str(model_path)

    if "/" in model_arg or "\\" in model_arg:
        raise FileNotFoundError(f"[ERR] model file not found: {model_path}")

    # Model alias (e.g. yolov8s.pt) can be handled by Ultralytics.
    return model_arg


def run_training(
    data: str = DEFAULT_DATA,
    model: str = DEFAULT_MODEL,
    epochs: int = 80,
    batch: int = 16,
    imgsz: int = 640,
    device: str = DEFAULT_DEVICE,
    name: str = "auto",
    project: str = DEFAULT_PROJECT,
    seed: int = 42,
    workers: int = 8,
    patience: int = 25,
    resume: bool = False,
) -> dict[str, Any]:
    root = repo_root()
    data_path = to_abs_path(data, root)
    if not data_path.exists():
        raise FileNotFoundError(f"[ERR] data.yaml not found: {data_path}")

    project_path = to_abs_path(project, root)
    project_path.mkdir(parents=True, exist_ok=True)

    run_name = choose_run_name(name, epochs, batch, imgsz)
    model_source = resolve_model_source(model, root)

    model_display = model_source
    source_as_path = Path(model_source)
    if source_as_path.exists():
        model_display = to_repo_display(source_as_path, root)

    print("[INFO] Training config")
    print(f"  - epochs: {epochs}")
    print(f"  - batch: {batch}")
    print(f"  - imgsz: {imgsz}")
    print(f"  - device: {device}")
    print(f"  - name: {run_name}")
    print(f"  - data: {to_repo_display(data_path, root)}")
    print(f"  - model: {model_display}")
    print(f"  - resume: {resume}")

    if resume:
        model_path = Path(model_source)
        if not model_path.exists():
            raise ValueError(
                "[ERR] --resume requires --model to be an existing checkpoint path "
                "(e.g. runs/detect/<run>/weights/last.pt)"
            )
        if model_path.suffix.lower() != ".pt":
            raise ValueError("[ERR] --resume model must be a .pt checkpoint file")
        if model_path.name != "last.pt":
            print(
                "[WARN] --resume usually expects last.pt for full optimizer-state "
                "resume; proceeding with the provided checkpoint."
            )

    seed_everything(seed)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "[ERR] ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    try:
        yolo = YOLO(model_source)
    except Exception as exc:
        raise RuntimeError(f"[ERR] failed to load model: {model_source}\n{exc}") from exc

    try:
        if resume:
            # Ultralytics restores training args from the checkpoint metadata.
            yolo.train(resume=True)
        else:
            yolo.train(
                data=str(data_path),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=str(device),
                name=run_name,
                project=str(project_path),
                seed=seed,
                workers=workers,
                patience=patience,
            )
    except Exception as exc:
        raise RuntimeError(f"[ERR] training failed.\n{exc}") from exc

    save_dir = getattr(getattr(yolo, "trainer", None), "save_dir", None)
    if save_dir is None:
        save_dir = project_path / run_name
    else:
        save_dir = Path(save_dir)

    best_pt = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"

    if best_pt.exists():
        print(f"[OK] best.pt: {to_repo_display(best_pt, root)}")
    else:
        print(
            "[WARN] Training finished, but best.pt was not found: "
            f"{to_repo_display(best_pt, root)}"
        )

    return {
        "name": run_name,
        "save_dir": save_dir,
        "best_pt": best_pt,
        "results_csv": results_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO using Python API")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--name",
        type=str,
        default="auto",
        help="If 'auto', use e{epochs}_b{batch}_img{imgsz}",
    )
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training from --model checkpoint "
        "(recommended: */weights/last.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_training(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
            project=args.project,
            seed=args.seed,
            workers=args.workers,
            patience=args.patience,
            resume=args.resume,
        )
    except Exception as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
