#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/detect/e80_b16_img6402/weights/best.pt"
DEFAULT_TEST_DIR = REPO_ROOT / "data/raw/test_images"
DEFAULT_CLASS_MAP = REPO_ROOT / "data/metadata/class_map.csv"
DEFAULT_OUT = REPO_ROOT / "submit.csv"


def load_class_map(path: Path) -> dict[int, int]:
    df = pd.read_csv(path, usecols=["class_index", "category_id"])
    return {
        int(row.class_index): int(row.category_id)
        for row in df.itertuples(index=False)
    }


def clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def build_submit_csv(
    weights: Path,
    test_dir: Path,
    class_map_csv: Path,
    out_csv: Path,
) -> None:
    class_map = load_class_map(class_map_csv)
    image_paths = sorted(test_dir.glob("*.png"), key=lambda p: int(p.stem))
    model = YOLO(str(weights))

    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    rows: list[dict[str, float | int]] = []
    annotation_id = 1
    zero_prediction_images = 0

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=896,
        conf=0.15,
        iou=0.6,
        batch=1,
        half=True,
        device="0",
        verbose=False,
        stream=True,
    )

    for result in results:
        image_id = int(Path(result.path).stem)
        image_h, image_w = result.orig_shape

        kept = 0
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_indices = boxes.cls.cpu().numpy().astype(int)

            for idx in scores.argsort()[::-1][:4]:
                class_index = int(class_indices[idx])
                if class_index not in class_map:
                    raise ValueError(
                        f"class_index {class_index} not found in {class_map_csv}"
                    )

                x1, y1, x2, y2 = xyxy[idx]
                x1 = clip(float(x1), 0.0, float(image_w))
                y1 = clip(float(y1), 0.0, float(image_h))
                x2 = clip(float(x2), 0.0, float(image_w))
                y2 = clip(float(y2), 0.0, float(image_h))

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0.0 or bbox_h <= 0.0:
                    continue

                rows.append(
                    {
                        "annotation_id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_map[class_index],
                        "bbox_x": x1,
                        "bbox_y": y1,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "score": float(scores[idx]),
                    }
                )
                annotation_id += 1
                kept += 1

        if kept == 0:
            zero_prediction_images += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"processed_images={len(image_paths)}")
    print(f"generated_rows={len(rows)}")
    print(f"zero_prediction_images={zero_prediction_images}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submit.csv from YOLO best.pt")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best.pt")
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR), help="Directory containing numeric .png test images")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="CSV with class_index,category_id")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submit_csv(
        weights=Path(args.weights),
        test_dir=Path(args.test_dir),
        class_map_csv=Path(args.class_map),
        out_csv=Path(args.out),
    )


if __name__ == "__main__":
    main()
