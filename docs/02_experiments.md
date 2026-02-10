# 02 Experiments

실험 운영 규칙 문서입니다. 목표는 재현성, 비교 가능성, 제출 안정성입니다.

## 1) 설정 관리 원칙

- 공통 기본값: `configs/base.yaml`
- 실험별 변경: `configs/experiments/*.yaml`
- 코드보다 YAML override를 우선합니다.
- 실행 시 병합 결과를 `runs/<run_name>/config_resolved.yaml`로 저장합니다.

## 2) 실험명 규칙

권장 패턴:

- `exp_YYYYMMDD_HHMMSS_<tag>`
- 예: `exp_20260210_213000_11s_lr_tune`

규칙:

- 날짜/시간 포함
- 모델/핵심 변경점 태그 포함
- 같은 이름 재사용 금지

## 3) 레지스트리 기록

- 파일: `runs/_registry.csv`
- 최소 기록 항목:
  - `run_name`
  - `config_path`
  - `best_metric` (예: mAP50-95)
  - `submission_path`
  - `git_commit` (가능 시)

실험 완료 후 반드시 레지스트리를 갱신합니다.

## 4) 결과물 저장 규칙

- 학습 산출물: `runs/<run_name>/`
- 제출 CSV: `artifacts/submissions/`
- 우수 모델: `artifacts/best_models/`
- 시각화/오답 샘플: `runs/<run_name>/vis/`

## 5) 제출 운영 (하루 제한)

- 팀 정책으로 1일 제출 횟수를 제한합니다.
- 권장: 최종 후보 1~2개만 제출
- 제출 전 체크리스트:
  - 로컬 평가 지표 확인
  - `submission.py` 검증 통과
  - 파일명에 run_name + conf 포함

예:

- `kaggle_exp_v1_conf0.15.csv`

## 6) 실험 비교 체크포인트

동일 조건 비교를 위해 아래를 고정/명시합니다.

- 데이터 split seed
- 입력 해상도(imgsz)
- conf / nms / max_det
- augmentation 주요 파라미터
- 학습 epoch, batch, lr

## 7) 실패 대응

- 데이터 이슈는 `data/metadata/audit_*.csv`부터 확인
- 설정 이슈는 `config_resolved.yaml` 우선 확인
- 성능 급락 시 최근 변경 YAML diff를 먼저 확인
