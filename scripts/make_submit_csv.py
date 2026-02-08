#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


FIELDS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def load_class_map(path: Path) -> dict[int, int]:
    df = pd.read_csv(path, usecols=["class_index", "category_id"])
    return {int(r.class_index): int(r.category_id) for r in df.itertuples(index=False)}


def resolve_output_path(out_arg: str, conf: float, iou: float, imgsz: int, min_conf: float) -> Path:
    auto_name = f"submit_conf{conf:.2f}_iou{iou:.2f}_img{imgsz}_min{min_conf:.2f}.csv"
    raw = out_arg.strip()
    if raw == "":
        return Path.cwd() / auto_name

    out_path = Path(raw)
    if out_path.suffix.lower() == ".csv":
        return out_path
    return out_path / auto_name


def run_predict(
    weights: Path,
    test_dir: Path,
    class_map: dict[int, int],
    imgsz: int,
    conf: float,
    iou: float,
    min_conf: float,
    topk: int,
) -> tuple[list[dict[str, float | int]], int, int]:
    model = YOLO(str(weights))
    image_ids = sorted(int(p.stem) for p in test_dir.glob("*.png"))
    per_image_count = {image_id: 0 for image_id in image_ids}

    rows: list[dict[str, float | int]] = []
    annotation_id = 1

    results = model.predict(
        source=str(test_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="0",
        batch=1,
        half=True,
        verbose=False,
        stream=True,
    )

    for result in results:
        image_id = int(Path(result.path).stem)
        image_h, image_w = result.orig_shape

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_indices = boxes.cls.cpu().numpy().astype(int)

        kept = 0
        for idx in scores.argsort()[::-1]:
            if kept >= topk:
                break

            score = float(scores[idx])
            if score < min_conf:
                continue

            class_index = int(class_indices[idx])
            if class_index not in class_map:
                raise ValueError(f"class_index {class_index} not found in class_map")

            x1, y1, x2, y2 = xyxy[idx]
            x1 = max(0.0, min(float(x1), float(image_w)))
            y1 = max(0.0, min(float(y1), float(image_h)))
            x2 = max(0.0, min(float(x2), float(image_w)))
            y2 = max(0.0, min(float(y2), float(image_h)))

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
                    "score": score,
                }
            )
            annotation_id += 1
            kept += 1

        per_image_count[image_id] = kept

    processed_images = len(image_ids)
    zero_prediction_images = sum(1 for v in per_image_count.values() if v == 0)
    return rows, processed_images, zero_prediction_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submit CSV from YOLO best.pt")
    parser.add_argument("--weights", required=True, help="best.pt path")
    parser.add_argument("--test-dir", required=True, help="test_images directory (numeric .png)")
    parser.add_argument("--class-map", required=True, help="class_map.csv path (class_index, category_id)")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--min-conf", type=float, default=0.15)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    class_map = load_class_map(Path(args.class_map))
    rows, processed_images, zero_prediction_images = run_predict(
        weights=Path(args.weights),
        test_dir=Path(args.test_dir),
        class_map=class_map,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        min_conf=args.min_conf,
        topk=args.topk,
    )

    out_path = resolve_output_path(
        out_arg=args.out,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    avg_boxes_per_image = (len(rows) / processed_images) if processed_images > 0 else 0.0
    print(f"processed_images={processed_images}")
    print(f"generated_rows={len(rows)}")
    print(f"zero_prediction_images={zero_prediction_images}")
    print(f"avg_boxes_per_image={avg_boxes_per_image:.6f}")
    print(f"saved_csv={out_path.resolve()}")


if __name__ == "__main__":
    main()

# python scripts/make_submit_csv.py --weights runs/detect/e80_b16_img6402/weights/best.pt --test-dir data/raw/test_images --class-map data/metadata/class_map.csv
# python scripts/make_submit_csv.py --weights runs/detect/e80_b16_img6402/weights/best.pt --test-dir data/raw/test_images --class-map data/metadata/class_map.csv --imgsz 640
# python scripts/make_submit_csv.py --weights runs/detect/e80_b16_img6402/weights/best.pt --test-dir data/raw/test_images --class-map data/metadata/class_map.csv --conf 0.12
