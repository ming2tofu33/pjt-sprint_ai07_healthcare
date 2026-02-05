#!/usr/bin/env python3
"""
전체 파이프라인 통합 실행 스크립트

기능:
1. Config 파일 기반 전체 파이프라인 실행
2. 각 단계별 에러 핸들링 및 로깅
3. 중간 단계부터 재개 가능
4. CLI 인자로 유연한 실험 설정

사용법:
    # 전체 파이프라인 실행 (YAML config 기반)
    python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml
    
    # 특정 단계만 실행
    python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --stages 1,2,3
    
    # CLI 인자로 설정 override
    python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --epochs 100 --batch 16
    
    # 실험명 지정
    python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --run-name exp_custom_v1
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_yaml_with_inheritance,
    save_config,
    ensure_dependencies,
    check_data_exists,
    print_section,
    get_project_defaults,
)


class PipelineRunner:
    """파이프라인 실행 매니저"""
    
    STAGES = {
        1: {"name": "COCO Format 생성", "script": "1_create_coco_format.py"},
        2: {"name": "Train/Val Split", "script": "0_splitting.py"},
        3: {"name": "YOLO Dataset 준비", "script": "2_prepare_yolo_dataset.py"},
        4: {"name": "모델 학습", "script": "3_train.py"},
        5: {"name": "모델 평가", "script": "4_evaluate.py"},
        6: {"name": "제출 파일 생성", "script": "5_submission.py"},
    }
    
    def __init__(self, config_path: Path, run_name: Optional[str] = None, 
                 root: Optional[Path] = None, args: Optional[argparse.Namespace] = None):
        self.config_path = Path(config_path).resolve()  # 절대 경로로 변환
        self.root = (root or Path(__file__).parent.parent).resolve()
        self.args = args
        
        # Config 로드 (YAML 상속 지원)
        print_section(f"Pipeline Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 상대 경로 표시 (에러 방지)
        try:
            config_rel = self.config_path.relative_to(self.root)
        except ValueError:
            config_rel = self.config_path
        
        print(f"\n[INFO] Config 로드: {config_rel}")
        
        try:
            self.config = load_yaml_with_inheritance(self.config_path, self.root)
        except Exception as e:
            print(f"[ERROR] Config 로드 실패: {e}")
            sys.exit(1)
        
        # Run name 결정
        if run_name:
            self.run_name = run_name
        elif "experiment" in self.config and "name" in self.config["experiment"]:
            exp_id = self.config["experiment"].get("id", "exp")
            exp_name = self.config["experiment"]["name"]
            self.run_name = f"{exp_id}_{exp_name}"
        else:
            self.run_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"[INFO] Run Name: {self.run_name}")
        
        # 경로 설정
        self.paths = setup_project_paths(
            run_name=self.run_name,
            root=self.root,
            create_dirs=True,
            check_input_exists=False,  # 나중에 체크
        )
        
        # Config를 JSON으로 저장 (실행 시점 snapshot)
        config_json_path = self.paths["CONFIG"] / "config.json"
        save_config(self.config, config_json_path)
        
        try:
            config_rel = config_json_path.relative_to(self.root)
        except ValueError:
            config_rel = config_json_path
        
        print(f"[INFO] Config snapshot 저장: {config_rel}")
    
    def check_prerequisites(self):
        """사전 조건 체크"""
        print_section("사전 조건 체크")
        
        # 1) 의존성 체크
        print("\n[1] 필수 패키지 체크...")
        ensure_dependencies(exit_on_missing=False)
        
        # 2) 데이터 체크
        print("\n[2] 데이터 존재 여부 체크...")
        data_status = check_data_exists(self.paths)
        
        all_exists = all(data_status.values())
        for key, exists in data_status.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {key}: {self.paths[key]}")
        
        if not all_exists:
            print("\n[WARNING] 일부 데이터 디렉토리가 없습니다.")
            print("[INFO] 로컬에 데이터가 있다면 계속 진행됩니다.")
        
        return all_exists
    
    def run_stage(self, stage_num: int, extra_args: Optional[List[str]] = None) -> bool:
        """
        특정 단계 실행
        
        Args:
            stage_num: 실행할 단계 번호 (1~6)
            extra_args: 추가 CLI 인자
        
        Returns:
            성공 여부
        """
        if stage_num not in self.STAGES:
            print(f"[ERROR] 잘못된 단계 번호: {stage_num}")
            return False
        
        stage = self.STAGES[stage_num]
        script_path = self.root / "scripts" / stage["script"]
        
        if not script_path.exists():
            print(f"[ERROR] 스크립트 없음: {script_path}")
            return False
        
        print_section(f"Stage {stage_num}: {stage['name']}")
        print(f"\n[INFO] 실행: {stage['script']}")
        
        # 기본 인자
        cmd = [sys.executable, str(script_path), "--run-name", self.run_name]
        
        # 추가 인자
        if extra_args:
            cmd.extend(extra_args)
        
        # CLI override 인자 추가 (args에서)
        if self.args:
            if stage_num == 4:  # Train
                if self.args.model:
                    cmd.extend(["--model", self.args.model])
                if self.args.epochs:
                    cmd.extend(["--epochs", str(self.args.epochs)])
                if self.args.batch:
                    cmd.extend(["--batch", str(self.args.batch)])
                if self.args.imgsz:
                    cmd.extend(["--imgsz", str(self.args.imgsz)])
                if self.args.device:
                    cmd.extend(["--device", self.args.device])
            
            elif stage_num == 6:  # Submission
                if self.args.conf:
                    cmd.extend(["--conf", str(self.args.conf)])
                if self.args.device:
                    cmd.extend(["--device", self.args.device])
        
        print(f"[CMD] {' '.join(cmd)}\n")
        
        # 실행
        try:
            result = subprocess.run(cmd, check=True, cwd=str(self.root))
            print(f"\n[SUCCESS] Stage {stage_num} 완료\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Stage {stage_num} 실패: {e}\n")
            return False
        except KeyboardInterrupt:
            print(f"\n[INFO] 사용자 중단\n")
            return False
    
    def run_pipeline(self, stages: Optional[List[int]] = None) -> Dict[int, bool]:
        """
        파이프라인 실행
        
        Args:
            stages: 실행할 단계 리스트 (None이면 전체)
        
        Returns:
            {stage_num: success} dict
        """
        if stages is None:
            stages = sorted(self.STAGES.keys())
        
        results = {}
        
        for stage_num in stages:
            success = self.run_stage(stage_num)
            results[stage_num] = success
            
            if not success:
                print(f"\n[ERROR] Stage {stage_num} 실패로 파이프라인 중단")
                break
        
        # 결과 요약
        print_section("파이프라인 실행 결과")
        for stage_num, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            stage_name = self.STAGES[stage_num]["name"]
            print(f"  {status} | Stage {stage_num}: {stage_name}")
        
        # 성공한 마지막 단계 출력
        if results:
            last_success = max([s for s, ok in results.items() if ok], default=0)
            print(f"\n[INFO] 마지막 성공 단계: {last_success}")
            
            if last_success == 6:
                submission_path = self.paths["SUBMISSIONS"] / "submission.csv"
                if submission_path.exists():
                    print(f"\n🎉 전체 파이프라인 완료!")
                    print(f"   제출 파일: {submission_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare AI Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 파이프라인 실행
  python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml
  
  # 특정 단계만 실행
  python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --stages 1,2,3
  
  # CLI 인자로 설정 override
  python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --epochs 100 --batch 16
        """
    )
    
    # 필수 인자
    parser.add_argument("--config", type=str, required=True, help="Config 파일 경로 (YAML)")
    
    # 선택 인자
    parser.add_argument("--run-name", type=str, help="실험명 (지정하지 않으면 config에서 추출)")
    parser.add_argument("--stages", type=str, help="실행할 단계 (쉼표 구분, 예: 1,2,3 또는 전체는 생략)")
    parser.add_argument("--skip-check", action="store_true", help="사전 조건 체크 건너뛰기")
    
    # Train 관련 override
    parser.add_argument("--model", type=str, help="모델명 (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, help="Epoch 수")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="GPU device (0, 1, cpu)")
    
    # Inference 관련 override
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Config 파일 존재 확인
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config 파일 없음: {config_path}")
        sys.exit(1)
    
    # PipelineRunner 생성
    try:
        runner = PipelineRunner(config_path, run_name=args.run_name, args=args)
    except Exception as e:
        print(f"[ERROR] PipelineRunner 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 사전 조건 체크
    if not args.skip_check:
        runner.check_prerequisites()
        input("\n계속하려면 Enter를 누르세요 (Ctrl+C로 취소)...")
    
    # 실행할 단계 파싱
    stages = None
    if args.stages:
        try:
            stages = [int(s.strip()) for s in args.stages.split(",")]
        except ValueError:
            print(f"[ERROR] 잘못된 --stages 형식: {args.stages}")
            print("올바른 예: --stages 1,2,3")
            sys.exit(1)
    
    # 파이프라인 실행
    results = runner.run_pipeline(stages=stages)
    
    # 종료 코드
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
