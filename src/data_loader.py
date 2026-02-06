#!/usr/bin/env python3
"""
Data Loader Module
YOLO 데이터셋 로딩 및 전처리
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class COCODataset(Dataset):
    """
    COCO 포맷 데이터셋 로더
    
    Args:
        coco_json_path: COCO JSON 파일 경로
        image_root: 이미지 루트 디렉토리
        split_ids: 사용할 이미지 ID 리스트 (None이면 전체)
        transform: 이미지 변환 함수
    """
    
    def __init__(
        self,
        coco_json_path: Path,
        image_root: Path,
        split_ids: Optional[List[int]] = None,
        transform=None,
    ):
        self.coco_json_path = Path(coco_json_path)
        self.image_root = Path(image_root)
        self.transform = transform
        
        # COCO 데이터 로드
        with open(self.coco_json_path, "r") as f:
            self.coco_data = json.load(f)
        
        # 이미지 필터링
        if split_ids is not None:
            split_ids_set = set(split_ids)
            self.images = [
                img for img in self.coco_data["images"]
                if img["id"] in split_ids_set
            ]
        else:
            self.images = self.coco_data["images"]
        
        # 이미지 ID → 인덱스 매핑
        self.image_id_to_idx = {img["id"]: idx for idx, img in enumerate(self.images)}
        
        # Annotation 빌드 (image_id별로 그룹화)
        self.annotations_by_image = self._build_annotations()
        
        # Category 정보
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
    
    def _build_annotations(self) -> Dict[int, List[Dict]]:
        """이미지별 annotation 그룹화"""
        anns_by_img = {img["id"]: [] for img in self.images}
        
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id in anns_by_img:
                anns_by_img[img_id].append(ann)
        
        return anns_by_img
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Dict]:
        """
        Returns:
            image: PIL Image 또는 Tensor (transform 적용 후)
            target: {
                "image_id": int,
                "boxes": [[x, y, w, h], ...],  # COCO format (xywh)
                "labels": [class_id, ...],
                "area": [area, ...],
                "iscrowd": [0/1, ...],
            }
        """
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        
        # 이미지 로드
        img_path = self.image_root / file_name
        image = Image.open(img_path).convert("RGB")
        
        # Annotation 가져오기
        anns = self.annotations_by_image[img_id]
        
        boxes = []
        labels = []
        areas = []
        iscrowds = []
        
        for ann in anns:
            boxes.append(ann["bbox"])  # [x, y, w, h]
            labels.append(ann["category_id"])
            areas.append(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
            iscrowds.append(ann.get("iscrowd", 0))
        
        target = {
            "image_id": img_id,
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowds,
        }
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def get_image_info(self, idx: int) -> Dict:
        """이미지 메타데이터 반환"""
        return self.images[idx]
    
    def get_category_info(self, cat_id: int) -> Dict:
        """카테고리 정보 반환"""
        return self.categories.get(cat_id, {})


class YOLODatasetWrapper:
    """
    YOLO 데이터셋 래퍼 (Ultralytics 호환)
    
    Args:
        data_yaml_path: data.yaml 파일 경로
    """
    
    def __init__(self, data_yaml_path: Path):
        self.data_yaml_path = Path(data_yaml_path)
        
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.data_yaml_path}")
        
        # data.yaml 파싱
        import yaml
        with open(self.data_yaml_path, "r") as f:
            self.data_config = yaml.safe_load(f)
        
        self.dataset_root = self.data_yaml_path.parent
        self.train_path = self.dataset_root / self.data_config["train"]
        self.val_path = self.dataset_root / self.data_config["val"]
        self.nc = self.data_config["nc"]
        self.names = self.data_config["names"]
    
    def get_train_path(self) -> Path:
        """Train 이미지 경로"""
        return self.train_path
    
    def get_val_path(self) -> Path:
        """Val 이미지 경로"""
        return self.val_path
    
    def get_num_classes(self) -> int:
        """클래스 개수"""
        return self.nc
    
    def get_class_names(self) -> List[str]:
        """클래스 이름 리스트"""
        return self.names
    
    def get_data_yaml_path(self) -> str:
        """data.yaml 경로 (Ultralytics 학습에 사용)"""
        return str(self.data_yaml_path)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn=None,
) -> DataLoader:
    """
    PyTorch DataLoader 생성
    
    Args:
        dataset: Dataset 객체
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: Worker 개수
        collate_fn: Collate 함수 (기본: None)
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def load_split_ids(split_file: Path) -> List[int]:
    """
    Split ID 파일 로드 (train_ids.txt, valid_ids.txt)
    
    Args:
        split_file: Split ID 파일 경로 (.txt)
    
    Returns:
        이미지 ID 리스트
    """
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    
    # 파싱 (숫자 또는 문자열)
    parsed_ids = []
    for id_str in ids:
        try:
            parsed_ids.append(int(id_str))
        except ValueError:
            # 숫자가 아니면 문자열 그대로
            parsed_ids.append(id_str)
    
    return parsed_ids


def load_label_map(label_map_path: Path) -> Dict:
    """
    Label map JSON 로드
    
    Args:
        label_map_path: label_map_full.json 또는 label_map_whitelist.json
    
    Returns:
        {
            "id2idx": {category_id: idx},
            "idx2id": {idx: category_id},
            "names": [name1, name2, ...]
        }
    """
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    
    return label_map


# ============================================================
# Collate Functions (Optional)
# ============================================================

def coco_collate_fn(batch):
    """
    COCO 데이터셋용 collate function
    배치 내 이미지 크기가 다를 수 있음
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# Utility Functions
# ============================================================

def get_dataset_stats(dataset: COCODataset) -> Dict:
    """
    데이터셋 통계 계산
    
    Returns:
        {
            "num_images": int,
            "num_annotations": int,
            "num_classes": int,
            "avg_objects_per_image": float,
            "class_distribution": {class_id: count},
        }
    """
    num_images = len(dataset)
    num_annotations = sum(len(anns) for anns in dataset.annotations_by_image.values())
    num_classes = len(dataset.categories)
    
    # 클래스 분포
    class_dist = {}
    for anns in dataset.annotations_by_image.values():
        for ann in anns:
            cat_id = ann["category_id"]
            class_dist[cat_id] = class_dist.get(cat_id, 0) + 1
    
    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "num_classes": num_classes,
        "avg_objects_per_image": num_annotations / num_images if num_images > 0 else 0,
        "class_distribution": class_dist,
    }


if __name__ == "__main__":
    # 간단한 테스트
    print("data_loader.py: COCO Dataset and YOLO Wrapper")
    print("Usage:")
    print("  from src.data_loader import COCODataset, YOLODatasetWrapper")
    print("  dataset = COCODataset(coco_json, image_root, split_ids)")
    print("  yolo_wrapper = YOLODatasetWrapper(data_yaml_path)")
