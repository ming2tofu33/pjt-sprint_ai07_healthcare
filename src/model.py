#!/usr/bin/env python3
"""
Model Module
YOLO 모델 래퍼 및 관리
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import json

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. Install with: pip install ultralytics")


class YOLOModel:
    """
    Ultralytics YOLO 모델 래퍼
    
    Args:
        model_name: 모델 이름 (yolov8n.pt, yolov8s.pt, ...) 또는 체크포인트 경로
        device: 디바이스 ("0", "cpu", ...)
        verbose: 상세 출력 여부
    """
    
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        device: str = "0",
        verbose: bool = True,
    ):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        
        # 모델 로드
        self.model = YOLO(model_name)
        
        if self.verbose:
            print(f"✅ Model loaded: {model_name}")
            print(f"   Device: {device}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 80,
        imgsz: int = 768,
        batch: int = 8,
        project: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        모델 학습
        
        Args:
            data_yaml: data.yaml 경로
            epochs: 에폭 수
            imgsz: 이미지 크기
            batch: 배치 크기
            project: 프로젝트 디렉토리
            name: 실험명
            **kwargs: 추가 Ultralytics 인자
        
        Returns:
            학습 결과 객체
        """
        if self.verbose:
            print(f"\n🚀 Starting training...")
            print(f"   Data: {data_yaml}")
            print(f"   Epochs: {epochs}")
            print(f"   Image size: {imgsz}")
            print(f"   Batch: {batch}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            verbose=self.verbose,
            **kwargs,
        )
        
        if self.verbose:
            print(f"✅ Training completed!")
        
        return results
    
    def validate(
        self,
        data_yaml: Optional[str] = None,
        split: str = "val",
        **kwargs,
    ) -> Any:
        """
        모델 검증
        
        Args:
            data_yaml: data.yaml 경로 (None이면 학습 시 사용한 것)
            split: 분할 (val/test)
            **kwargs: 추가 Ultralytics 인자
        
        Returns:
            검증 결과 객체
        """
        if self.verbose:
            print(f"\n🔍 Starting validation...")
            if data_yaml:
                print(f"   Data: {data_yaml}")
            print(f"   Split: {split}")
        
        results = self.model.val(
            data=data_yaml,
            split=split,
            device=self.device,
            verbose=self.verbose,
            **kwargs,
        )
        
        if self.verbose:
            print(f"✅ Validation completed!")
        
        return results
    
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
            **kwargs: 추가 Ultralytics 인자
        
        Returns:
            추론 결과 리스트
        """
        if self.verbose:
            print(f"\n🔮 Starting inference...")
            print(f"   Source: {source}")
            print(f"   Conf: {conf}, IoU: {iou}, Max det: {max_det}")
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device,
            save=save,
            verbose=self.verbose,
            **kwargs,
        )
        
        if self.verbose:
            print(f"✅ Inference completed! ({len(results)} images)")
        
        return results
    
    def export(
        self,
        format: str = "onnx",
        **kwargs,
    ) -> str:
        """
        모델 내보내기
        
        Args:
            format: 포맷 (onnx, torchscript, tflite, ...)
            **kwargs: 추가 Ultralytics 인자
        
        Returns:
            내보낸 파일 경로
        """
        if self.verbose:
            print(f"\n📦 Exporting model to {format}...")
        
        path = self.model.export(format=format, **kwargs)
        
        if self.verbose:
            print(f"✅ Export completed: {path}")
        
        return path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로 (.pt)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = YOLO(str(checkpoint_path))
        self.model_name = str(checkpoint_path)
        
        if self.verbose:
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            {
                "model_name": str,
                "device": str,
                "num_params": int (if available),
                "model_type": str,
            }
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "YOLO",
        }
        
        # Parameter 개수 (가능하면)
        try:
            num_params = sum(p.numel() for p in self.model.model.parameters())
            info["num_params"] = num_params
        except:
            info["num_params"] = None
        
        return info
    
    def save_predictions_to_coco(
        self,
        results: List,
        output_path: Path,
        image_id_offset: int = 0,
    ) -> Path:
        """
        추론 결과를 COCO JSON 형식으로 저장
        
        Args:
            results: model.predict() 결과 리스트
            output_path: 출력 파일 경로
            image_id_offset: 이미지 ID 오프셋
        
        Returns:
            저장된 파일 경로
        """
        annotations = []
        annotation_id = 1
        
        for img_idx, result in enumerate(results):
            image_id = img_idx + image_id_offset
            
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
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "score": score,
                    "area": w * h,
                    "iscrowd": 0,
                })
                annotation_id += 1
        
        coco_output = {
            "annotations": annotations,
            "categories": [],  # TODO: 필요 시 추가
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        
        if self.verbose:
            print(f"✅ Predictions saved to COCO format: {output_path}")
        
        return output_path


def load_model(checkpoint_path: str, device: str = "0", verbose: bool = True) -> YOLOModel:
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스
        verbose: 상세 출력 여부
    
    Returns:
        YOLOModel 인스턴스
    """
    model = YOLOModel(model_name=checkpoint_path, device=device, verbose=verbose)
    return model


def get_available_models() -> List[str]:
    """
    사용 가능한 YOLO 모델 리스트
    
    Returns:
        모델 이름 리스트
    """
    return [
        "yolov8n.pt",   # Nano
        "yolov8s.pt",   # Small
        "yolov8m.pt",   # Medium
        "yolov8l.pt",   # Large
        "yolov8x.pt",   # Extra Large
    ]


if __name__ == "__main__":
    # 간단한 테스트
    print("model.py: YOLO Model Wrapper")
    print("\nAvailable models:")
    for model in get_available_models():
        print(f"  - {model}")
    print("\nUsage:")
    print("  from src.model import YOLOModel")
    print("  model = YOLOModel('yolov8s.pt')")
    print("  model.train(data_yaml='data.yaml', epochs=80)")
