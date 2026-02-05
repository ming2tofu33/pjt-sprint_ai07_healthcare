#!/usr/bin/env python3
"""
Inference Module
추론 및 결과 처리
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import json
import numpy as np
import pandas as pd

from .model import YOLOModel
from .utils import print_section


class Inferencer:
    """
    추론 실행 및 결과 처리
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스 ("0", "cpu", ...)
        verbose: 상세 출력 여부
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "0",
        verbose: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.verbose = verbose
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # 모델 로드
        self.model = YOLOModel(
            model_name=str(self.checkpoint_path),
            device=device,
            verbose=verbose,
        )
        
        if self.verbose:
            print_section("Inferencer Initialized")
            print(f"  Checkpoint: {self.checkpoint_path}")
            print(f"  Device: {device}")
    
    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        save: bool = False,
        **kwargs,
    ) -> Any:
        """
        추론 실행
        
        Args:
            source: 이미지 경로 (파일/폴더/URL)
            conf: Confidence threshold
            iou: NMS IoU threshold
            max_det: 최대 검출 개수
            save: 결과 저장 여부
            **kwargs: 추가 인자
        
        Returns:
            추론 결과 리스트 (Ultralytics Results 객체)
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            save=save,
            **kwargs,
        )
        
        return results
    
    def predict_and_filter_top_k(
        self,
        source: str,
        top_k: int = 4,
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs,
    ) -> Any:
        """
        추론 후 Top-K 필터링
        
        Args:
            source: 이미지 경로
            top_k: 상위 K개 선택
            conf: Confidence threshold
            iou: NMS IoU threshold
            **kwargs: 추가 인자
        
        Returns:
            추론 결과 (Top-K 필터링됨)
        """
        # max_det을 top_k로 설정 (YOLO가 자동으로 top_k만 반환)
        results = self.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=top_k,
            **kwargs,
        )
        
        return results
    
    def results_to_dataframe(self, results: List) -> pd.DataFrame:
        """
        추론 결과를 DataFrame으로 변환
        
        Args:
            results: model.predict() 결과 리스트
        
        Returns:
            DataFrame with columns: [image_id, bbox_x, bbox_y, bbox_w, bbox_h, class_id, score]
        """
        rows = []
        
        for img_idx, result in enumerate(results):
            image_path = Path(result.path)
            
            # 이미지 ID 추출 (파일명 기반)
            try:
                image_id = int(image_path.stem)
            except ValueError:
                # 숫자가 아니면 인덱스 사용
                image_id = img_idx
            
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box_idx in range(len(boxes)):
                cls = int(boxes.cls[box_idx].item())
                score = float(boxes.conf[box_idx].item())
                xyxy = boxes.xyxy[box_idx].cpu().numpy()
                
                # xyxy → xywh
                x1, y1, x2, y2 = xyxy
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                rows.append({
                    "image_id": image_id,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "class_id": cls,
                    "score": score,
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def create_submission_csv(
        self,
        results: List,
        output_path: Path,
        idx2id_map: Optional[Dict[int, int]] = None,
        top_k: int = 4,
    ) -> Path:
        """
        Kaggle 제출 파일 생성 (submission.csv)
        
        Args:
            results: model.predict() 결과 리스트
            output_path: 출력 CSV 파일 경로
            idx2id_map: YOLO 인덱스 → 원본 category_id 매핑 (필수!)
            top_k: 이미지당 최대 객체 개수
        
        Returns:
            저장된 파일 경로
        """
        if idx2id_map is None:
            raise ValueError(
                "idx2id_map is required for submission! "
                "Load from label_map_full.json or label_map_whitelist.json"
            )
        
        rows = []
        annotation_id = 1
        
        for img_idx, result in enumerate(results):
            image_path = Path(result.path)
            
            # 이미지 ID 추출
            try:
                image_id = int(image_path.stem)
            except ValueError:
                image_id = img_idx
            
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Top-K 선택 (score 높은 순)
            scores = boxes.conf.cpu().numpy()
            top_indices = scores.argsort()[::-1][:top_k]
            
            for idx in top_indices:
                cls_idx = int(boxes.cls[idx].item())  # YOLO 인덱스
                score = float(boxes.conf[idx].item())
                xyxy = boxes.xyxy[idx].cpu().numpy()
                
                # YOLO 인덱스 → 원본 category_id 변환
                if cls_idx not in idx2id_map:
                    if self.verbose:
                        print(f"  ⚠️  Unknown class index: {cls_idx} (이미지 {image_id})")
                    continue
                category_id = idx2id_map[cls_idx]
                
                # xyxy → xywh
                x1, y1, x2, y2 = xyxy
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                rows.append({
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # 원본 category_id
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "score": score,
                })
                annotation_id += 1
        
        df = pd.DataFrame(rows)
        
        # 빈 결과 처리
        if df.empty:
            df = pd.DataFrame(columns=[
                "annotation_id", "image_id", "category_id",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
            ])
        
        # 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print_section("Submission CSV Created")
            print(f"  Output: {output_path}")
            print(f"  Total objects: {len(df)}")
            print(f"  Unique images: {df['image_id'].nunique() if not df.empty else 0}")
        
        return output_path
    
    def validate_submission_csv(self, csv_path: Path) -> Dict[str, Any]:
        """
        제출 파일 검증
        
        Args:
            csv_path: submission.csv 경로
        
        Returns:
            검증 결과 딕셔너리
        """
        df = pd.read_csv(csv_path)
        
        required_cols = [
            "annotation_id", "image_id", "category_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
        ]
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }
        
        # 1. 필수 컬럼 체크
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            validation["valid"] = False
            validation["errors"].append(f"Missing columns: {missing_cols}")
        
        # 2. NaN 체크
        if df.isnull().any().any():
            validation["valid"] = False
            validation["errors"].append("NaN values found")
        
        # 3. 음수 bbox 체크
        if (df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]] < 0).any().any():
            validation["warnings"].append("Negative bbox values found")
        
        # 4. 통계
        validation["stats"] = {
            "total_objects": len(df),
            "unique_images": df["image_id"].nunique() if not df.empty else 0,
            "avg_objects_per_image": len(df) / df["image_id"].nunique() if not df.empty else 0,
            "unique_categories": df["category_id"].nunique() if not df.empty else 0,
        }
        
        if self.verbose:
            print_section("Submission Validation")
            print(f"  Valid: {validation['valid']}")
            if validation["errors"]:
                print(f"  Errors: {validation['errors']}")
            if validation["warnings"]:
                print(f"  Warnings: {validation['warnings']}")
            print(f"  Stats: {validation['stats']}")
        
        return validation
    
    def visualize_predictions(
        self,
        results: List,
        output_dir: Path,
        max_images: int = 10,
    ):
        """
        예측 결과 시각화 (TODO: 구현 필요)
        
        Args:
            results: model.predict() 결과
            output_dir: 출력 디렉토리
            max_images: 최대 시각화 이미지 개수
        """
        # TODO: matplotlib/cv2로 시각화
        if self.verbose:
            print("⚠️  visualize_predictions() not implemented yet")


def load_inferencer(checkpoint_path: str, device: str = "0", verbose: bool = True) -> Inferencer:
    """
    Inferencer 생성 헬퍼 함수
    
    Args:
        checkpoint_path: 체크포인트 경로
        device: 디바이스
        verbose: 상세 출력
    
    Returns:
        Inferencer 인스턴스
    """
    return Inferencer(checkpoint_path=checkpoint_path, device=device, verbose=verbose)


if __name__ == "__main__":
    # 간단한 테스트
    print("inference.py: Inference and Result Processing")
    print("\nUsage:")
    print("  from src.inference import Inferencer")
    print("  inferencer = Inferencer('runs/exp001/checkpoints/best.pt')")
    print("  results = inferencer.predict('data/raw/test_images/')")
    print("  inferencer.create_submission_csv(results, 'submission.csv')")
