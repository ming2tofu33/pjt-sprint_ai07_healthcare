# 실험 가이드 (Experiments Guide)

이 문서는 프로젝트의 **계층적 설정 시스템(Hierarchical Configuration System)**을 사용하여 실험을 설계, 실행 및 분석하는 방법을 상세히 기술합니다.

---

## 1. 개요 (Overview)

본 프로젝트는 실험 설정의 **재사용성**과 **유지보수성**을 높이기 위해 YAML 기반의 상속 구조를 채택했습니다.
모든 실험은 공통 설정(`base.yaml`)을 상속받으며, 변경이 필요한 파라미터만 오버라이드하여 관리합니다. 이를 통해 중복을 제거하고 실험 간 차이점을 명확히 파악할 수 있습니다.

---

## 2. 설정 시스템 상세 (Configuration System)

### 2.1 상속 구조 (Inheritance)
실험 설정 파일(`configs/experiments/*.yaml`)은 `_base_` 키를 통해 부모 설정을 참조합니다. 로더는 부모 설정을 먼저 로드한 뒤, 자식 설정의 값으로 **Deep Merge(재귀적 병합)**를 수행합니다.

**예시:**
```yaml
# configs/experiments/my_experiment.yaml
_base_: "../base.yaml"  # 1. base.yaml 로드

model:
  architecture: "yolov8s"  # 2. base의 architecture 값을 덮어씀

train:
  epochs: 50               # 3. train 섹션 내의 epochs 값만 변경 (나머지 train 설정은 유지)
```

### 2.2 경로 정규화 (Path Normalization)
설정 파일 내의 모든 경로는 **상대 경로**로 작성하는 것을 권장합니다. 시스템이 실행 시점에 프로젝트 루트(Repo Root)를 기준으로 **절대 경로로 자동 변환**합니다. 따라서 서버마다 작업 디렉터리가 달라져도 설정 파일을 수정할 필요가 없습니다.

---

## 3. 실험 실행 (Workflows)

### 3.1 단일 실험 (Standard Pipeline)
하나의 실험 설정을 사용하여 전체 파이프라인(전처리→학습→평가→제출)을 실행합니다. 기본적으로 로그 출력이 억제된 **Quiet 모드**로 동작하여 터미널이 깔끔하게 유지됩니다.

**명령어 포맷:**
```bash
bash scripts/run_pipeline.sh --run-name <실험명> --config <설정파일경로> [옵션]
```

**주요 옵션:**
*   `--verbose`: 상세 로그를 출력합니다. (예: 추론 시 이미지별 결과 출력)
*   `--device`: 사용할 GPU 디바이스 (예: `0`, `1`, `cpu`)
*   `--start`: 시작할 STAGE 번호 (0~4)
*   `--stop`: 종료할 STAGE 번호 (0~4)

**사용 예시:**
```bash
# 전체 파이프라인 실행 (Quiet 모드)
bash scripts/run_pipeline.sh --run-name exp_v1 --config configs/experiments/baseline.yaml

# 상세 로그와 함께 실행
bash scripts/run_pipeline.sh --run-name exp_v1 --config configs/experiments/baseline.yaml --verbose

# GPU 1번 사용, 학습(Stage 2)부터 시작
bash scripts/run_pipeline.sh --run-name exp_v1 --config configs/experiments/baseline.yaml --device 1 --start 2
```

### 3.2 매트릭스 실험 (Matrix Run)
여러 하이퍼파라미터 조합을 비교하거나, 변경 사항이 시스템에 미치는 영향을 빠르게 검증할 때 사용합니다. `run_matrix_smoke.sh` 스크립트는 **Seed Run** 최적화를 사용합니다.

*   **Seed Run 최적화:**
    *   데이터 전처리(STAGE 0~1)를 한 번만 수행하여 공용 캐시(Seed)를 생성합니다.
    *   개별 실험 케이스(Case)들은 이 Seed를 복제(Link/Copy)하여 학습(STAGE 2)부터 즉시 시작합니다.

**명령어 포맷:**
```bash
bash scripts/run_matrix_smoke.sh --mode <quick|balanced|broad> [옵션]
```

**모드 설명:**
*   `quick`: 3개 케이스. 최소한의 기능 작동 여부 확인. (약 5~10분 소요)
*   `balanced`: 6개 케이스. 주요 하이퍼파라미터 변화 감지.
*   `broad`: 10개 이상. 다양한 넓은 케이스 포함.

---

## 4. 결과 분석 및 산출물 (Analysis & Outputs)

### 4.1 디렉터리 구조
실험이 완료되면 다음과 같은 경로에 산출물이 생성됩니다.

| 구분 | 경로 (기본값) | 내용 |
| :--- | :--- | :--- |
| **로그/체크포인트** | `runs/<run_name>/` | 학습 로그(`train.log`), 모델 가중치(`weights/*.pt`), 학습 곡선 그래프 등 |
| **제출 파일** | `artifacts/submissions/` | `submission.csv` (Dacon 제출 양식) |
| **베스트 모델** | `artifacts/best_models/` | 성능이 가장 좋은 모델 파일 (`*_best.pt`) |
| **매트릭스 결과** | `playground/matrix/<prefix>/` | 매트릭스 실험별 임시 설정 파일 및 통합 요약서 |

### 4.2 매트릭스 요약서 (Summary)
매트릭스 실험 후 `playground/matrix/<prefix>/summary.tsv` 파일이 생성됩니다. 이 파일은 탭(Tab)으로 구분되어 있으며 엑셀이나 구글 시트에 붙여넣어 보기 좋습니다.

**주요 컬럼:**
*   `status`: PASS / FAIL
*   `elapsed_sec`: 소요 시간 (초)
*   `mAP50_95`: 주요 평가 지표 (높을수록 좋음)

---

## 5. 트러블슈팅 (Troubleshooting)

### 5.1 자주 발생하는 오류
1.  **`KeyError: 'paths'` 또는 경로 누락 에러**
    *   **원인:** 설정 파일 상속 과정에서 `paths` 섹션이 누락되었거나 `_base_` 경로가 잘못되었습니다.
    *   **해결:** YAML 파일 최상단의 `_base_` 경로가 올바른지 확인하고, `base.yaml`의 `paths` 키가 제대로 상속되었는지 확인합니다.

2.  **`Circular inheritance detected`**
    *   **원인:** A가 B를 상속하고, B가 다시 A를 상속하는 순환 참조가 발생했습니다.
    *   **해결:** 상속 구조를 단방향(Tree 구조)으로 수정하세요.

### 5.2 디스크 공간 관리
매트릭스 실험은 많은 디스크 공간을 소비할 수 있습니다.
*   `playground/matrix/` 디렉터리는 실험 후 삭제해도 안전합니다. (단, 필요한 로그나 설정은 백업하세요.)
*   `data/processed/cache/` 내의 오래된 실험 폴더(`exp_*`)를 주기적으로 정리하세요.

---

## 6. 작업 모범 사례 (Best Practices)

1.  **`base.yaml` 수정 주의:** `base.yaml`을 수정하면 모든 실험에 영향을 미칩니다. 공통적인 변경 사항이 아니라면 개별 실험 YAML에서 오버라이드하세요.
2.  **실험명 규칙:** 실험명(`--run-name`)에 날짜나 특징을 포함하면 관리가 쉽습니다. (예: `260210_yolov8s_lr001`)
3.  **코드 vs 설정:** 모델 구조나 학습 로직을 하드코딩하기보다, 가능한 한 YAML 설정으로 제어할 수 있도록 코드를 작성하세요.

---

## 7. 중단 복구 (Resume)

학습(STAGE 2) 중단 시에는 아래 방식으로 이어서 실행할 수 있습니다.

### 7.1 파이프라인에서 명시적으로 재개
```bash
bash scripts/run_pipeline.sh --run-name exp_v1 --config configs/experiments/baseline.yaml --start 2 --resume-train
```

### 7.2 파이프라인에서 자동 재개
`last.pt`가 있으면 재개하고, 없으면 처음부터 학습을 시작합니다.

```bash
bash scripts/run_pipeline.sh --run-name exp_v1 --config configs/experiments/baseline.yaml --start 2 --auto-resume-train
```

### 7.3 학습 스크립트 직접 실행
```bash
# 강제 재개
python scripts/2_train.py --run-name exp_v1 --config configs/experiments/baseline.yaml --resume

# 자동 재개
python scripts/2_train.py --run-name exp_v1 --config configs/experiments/baseline.yaml --auto-resume
```

> 주의: 기본 동작은 안전하게 "새 학습"입니다. 자동 복구 옵션을 명시했을 때만 동작합니다.