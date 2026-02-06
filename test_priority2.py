#!/usr/bin/env python3
"""
Priority 2: 여러 confidence threshold 테스트
다양한 conf 값으로 submission 생성 후 비교
"""

import subprocess
import sys
from pathlib import Path

# 테스트할 confidence threshold 값들
CONF_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_priority2.py <run_name>")
        print("Example: python test_priority2.py test_exp")
        sys.exit(1)
    
    run_name = sys.argv[1]
    
    print("=" * 60)
    print(f"Priority 2: Confidence Threshold 테스트")
    print(f"RUN_NAME: {run_name}")
    print("=" * 60)
    
    print("\n테스트할 confidence 값:")
    for conf in CONF_THRESHOLDS:
        print(f"  - {conf}")
    
    print("\n" + "=" * 60)
    
    for conf in CONF_THRESHOLDS:
        print(f"\n[{conf}] 제출 파일 생성 중...")
        
        # submission 파일명
        output_name = f"submission_conf{conf:.2f}.csv"
        
        # scripts/5_submission.py 실행
        cmd = [
            sys.executable,
            "scripts/5_submission.py",
            "--run-name", run_name,
            "--conf", str(conf),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5분 타임아웃
            )
            
            if result.returncode == 0:
                print(f"  ✅ 성공: {output_name}")
                
                # submission.csv 복사
                src = Path(f"artifacts/{run_name}/submissions/submission.csv")
                dst = Path(f"artifacts/{run_name}/submissions/{output_name}")
                
                if src.exists():
                    import shutil
                    shutil.copy(src, dst)
                    print(f"  📁 저장: {dst}")
                
                # 통계 출력 (stdout에서 추출)
                for line in result.stdout.split('\n'):
                    if "총 예측 객체" in line or "이미지당 평균" in line:
                        print(f"     {line.strip()}")
            else:
                print(f"  ❌ 실패: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  타임아웃")
        except Exception as e:
            print(f"  ❌ 에러: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    
    print("\n생성된 파일:")
    submissions_dir = Path(f"artifacts/{run_name}/submissions")
    if submissions_dir.exists():
        for f in sorted(submissions_dir.glob("submission_conf*.csv")):
            print(f"  - {f.name}")
    
    print("\n다음 단계:")
    print("  1. 각 submission 파일을 Kaggle에 제출")
    print("  2. mAP 점수 비교")
    print("  3. 최적 confidence 값 선택")
    print("\n권장 전략:")
    print("  - mAP@0.75~0.95 → 높은 conf (0.35~0.50) 권장")
    print("  - Precision 중시 → conf 높이기")
    print("  - Recall 중시 → conf 낮추기")


if __name__ == "__main__":
    main()
