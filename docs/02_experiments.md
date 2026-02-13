# 02 Experiments

실험 운영 규칙 문서입니다. 목표는 재현성, 비교 가능성, 제출 안정성입니다.

## 1) 설정 관리 원칙

- 공통 기본값: `configs/base.yaml`
  - `train.competition_select` 설정을 통해 대회용 지표(mAP75_95) 기반 모델 재선정 기능을 제어할 수 있습니다.
  - `paths.artifact_layout` 기본값은 `compact`이며, `legacy`는 실험 YAML에서 명시적으로 opt-in 합니다.
  - `train.log_mode` 기본값은 `batch`, `train.debug_snapshots.enabled` 기본값은 `true`입니다.
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
- 현재 레지스트리 컬럼:
  - `run_name`
  - `created_at`
  - `model`
  - `epochs`
  - `imgsz`
  - `best_map50`
  - `best_map50_95`
  - `best_map75_95` (대회 지표: mAP75_95)
  - `weights_path`
  - `config_path`
  - `notes`

실험 완료 후 반드시 레지스트리를 갱신합니다.

## 4) 결과물 저장 규칙

- 학습 산출물: `runs/<run_name>/train/` (`compact` 기본), `runs/<run_name>/` (`legacy` 옵션)
- 대회용 선정 가중치: `runs/<run_name>/weights/competition_best.pt`
- 대회용 선정 리포트: `runs/<run_name>/competition_best.json`
- 제출 CSV: `artifacts/submissions/`
- 우수 모델: `artifacts/best_models/`
- 시각화/오답 샘플: `runs/<run_name>/vis/`

`competition_best.pt`는 STAGE 2에서 `best.pt`/`last.pt`를 mAP75_95 기준으로 재평가해 선택한 파일입니다.

레이아웃 모드별 경로:

| 모드 | 학습 산출물 | 제출 디버그 | 제출 매니페스트 |
|---|---|---|---|
| `legacy` | `runs/<run_name>/` | `runs/<run_name>/submission_debug/` | `runs/<run_name>/submission_manifest.json` |
| `compact` | `runs/<run_name>/train/` | `runs/<run_name>/submit/debug/` | `runs/<run_name>/submit/submission_manifest.json` (+ 루트 shortcut) |

`train.debug_snapshots.enabled=true`일 때는 val 스냅샷이 `runs/<run_name>/train/snapshots/`에 저장됩니다.

현재 1·2·3·5·6 개선 작업은 `competition_select` 동작(#4)을 변경하지 않는 것을 원칙으로 합니다.

## 5) 제출 운영 (하루 제한)

- 팀 정책으로 1일 제출 횟수(10회)를 제한합니다.
- 권장: 최종 후보 인당 1~2개 제출
- 제출 전 체크리스트:
  - 로컬 평가 지표 확인
  - `submission.py` 검증 통과
  - 이미지당 최대 박스 수 확인 (기본 4, `submission.max_det_per_image` 또는 `--topk`로 조정 가능)
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
