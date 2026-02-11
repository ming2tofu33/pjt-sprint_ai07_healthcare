# 팀 공용 실행 가이드 (`feat/JK`)

## 1. 문서 목적
이 문서는 `feat/JK` 브랜치 기준으로 팀원들이 같은 방식으로 데이터 분할, 전처리, 학습, 제출 CSV 생성을 재현할 수 있도록 정리한 실행 가이드입니다.  
이번 반영의 핵심은 아래 3가지입니다.

1. STAGE 0에서 `exclude_4444_208.txt` 목록을 자동 반영해 특정 이미지들을 공통 제외
2. STAGE 2/4에서 기존 로컬 커맨드 스타일을 최대한 유지할 수 있도록 CLI 호환 옵션 추가
3. 상대 경로 기반으로 구성해서 팀원 컴퓨터 환경에서도 동일하게 실행 가능

---

## 2. 이번 브랜치에서 팀원이 알아야 할 변경 핵심
### 2-1. 4444 제외 목록 자동 반영
- 파일: `configs/resources/exclude_4444_208.txt`
- 연결: `configs/base.yaml`의 `manual_overrides.exclude_file_names_file`
- 동작: `scripts/0_split_data.py` 실행 시 목록 파일을 읽어 `manual_overrides.exclude_file_names`에 병합합니다.

즉, 팀원이 별도로 코드 수정 없이 STAGE 0만 정상 실행하면 4444 제외가 공통으로 반영됩니다.

### 2-2. 학습/제출 스크립트의 실사용 CLI 호환
- `scripts/2_train.py`: `--data-yaml`, `--model`, `--project`, `--epochs`, `--imgsz`, `--batch`, `--lr0`, `--lrf`, `--mosaic`, `--close-mosaic`, `--mixup`, `--copy-paste`, `--box`, `--cls`, `--dfl`, `--classes`, `--target-category-ids` 등 지원
- `scripts/4_submission.py`: `--weights`, `--test-dir`, `--class-map`, `--imgsz`, `--iou`, `--min-conf`, `--topk`, `--out`, `--keep-category-ids`, `--keep-category-ids-file` 등 지원

기존의 단순 파이프라인 실행도 가능하고, 필요하면 실험 파라미터를 CLI로 빠르게 오버라이드할 수 있습니다.

### 2-3. 팀 환경 호환성
- 경로는 상대 경로 기준으로 동작합니다.
- 단, 데이터 루트 구조(`data/raw/...`)는 팀원 로컬에서도 동일해야 합니다.
- `configs/base.yaml`의 `external_data.enabled: true` 상태이므로 외부 데이터 경로가 없는 팀원은 해당 경로 준비가 필요합니다.

---

## 3. 실행 전 체크리스트
아래 7개를 먼저 확인하세요.

1. Python 가상환경 및 필수 패키지 설치 완료 (`ultralytics`, `pandas`, `pyyaml` 등)
2. 저장소 루트에서 명령 실행 중인지 확인
3. `data/raw/train_images`, `data/raw/train_annotations`, `data/raw/test_images` 존재 확인
4. 외부 데이터 사용 시 `data/raw/external/combined/images`, `data/raw/external/combined/annotations` 존재 확인
5. `configs/resources/exclude_4444_208.txt` 파일 존재 확인
6. GPU 학습 시 CUDA/드라이버 정상 확인
7. 이전 실행 결과와 충돌 방지를 위해 `run_name` 규칙 통일

권장 확인 명령:

```powershell
python -V
git branch --show-current
Get-ChildItem data/raw
Get-ChildItem configs/resources
```

---

## 4. 표준 실행 순서 (STAGE 0 -> 4)
아래는 팀 공용으로 가장 안전한 기본 순서입니다.

### STAGE 0: split + 정제 + COCO/label_map 생성
```powershell
python scripts/0_split_data.py --run-name exp_team_001 --config configs/base.yaml --verbose
```

산출물(대표):
- `data/processed/cache/exp_team_001/df_clean.csv`
- `data/processed/cache/exp_team_001/label_map_full.json`
- `data/metadata/splits.csv`

### STAGE 1: YOLO 포맷 데이터셋 생성
```powershell
python scripts/1_preprocess.py --run-name exp_team_001 --config configs/base.yaml --verbose
```

산출물(대표):
- `data/processed/datasets/pill_od_yolo_exp_team_001/data.yaml`
- `data/metadata/class_map.csv`

### STAGE 2: 학습
```powershell
python scripts/2_train.py --run-name exp_team_001 --config configs/base.yaml --device 0
```

산출물(대표):
- `runs/exp_team_001/weights/best.pt`
- `runs/exp_team_001/weights/competition_best.pt` (설정 사용 시)
- `runs/exp_team_001/results.csv`

### STAGE 3: 평가
```powershell
python scripts/3_evaluate.py --run-name exp_team_001 --config configs/base.yaml --device 0
```

### STAGE 4: 제출 CSV 생성
```powershell
python scripts/4_submission.py --run-name exp_team_001 --config configs/base.yaml --device 0
```

산출물(대표):
- `artifacts/submissions/exp_team_001_conf0.25.csv` (기본값 기준)

---

## 5. 실전에서 자주 쓰는 오버라이드 예시
### 5-1. 특정 data.yaml로 바로 학습
```powershell
python scripts/2_train.py `
  --run-name y8_custom `
  --config configs/base.yaml `
  --data-yaml data/processed/yolo/exclude_4444/data_excluded_4444.yaml `
  --model yolo11s.pt `
  --epochs 60 --imgsz 1024 --batch 8 `
  --lr0 0.0018 --lrf 0.05 `
  --mosaic 0.05 --close-mosaic 35 `
  --mixup 0.0 --copy-paste 0.0 `
  --box 9.0 --cls 0.5 --dfl 2.0 `
  --device 0
```

### 5-2. 특정 가중치로 제출 CSV 생성
```powershell
python scripts/4_submission.py `
  --run-name y8_custom `
  --config configs/base.yaml `
  --weights runs/detect/y8m1024_non74_resplit_e60/weights/best.pt `
  --test-dir data/raw/test_images `
  --class-map data/metadata/class_map.csv `
  --imgsz 1024 --conf 0.10 --iou 0.60 --min-conf 0.15 --topk 4 `
  --out runs/analysis/submit_y8_custom.csv `
  --device 0
```

### 5-3. 제출 시 category 필터 적용 (예: 74개만 유지)
```powershell
python scripts/4_submission.py `
  --run-name y8_custom `
  --config configs/base.yaml `
  --weights runs/detect/y8m1024_non74_resplit_e60/weights/best.pt `
  --class-map data/metadata/class_map.csv `
  --keep-category-ids-file runs/analysis/target74_category_ids.txt `
  --out runs/analysis/submit_y8_custom_keep74.csv
```

---

## 6. 결과 재현성 관련 주의사항
`results.csv` 숫자가 동일하게 나오려면 아래가 모두 같아야 합니다.

1. 동일한 입력 데이터셋 리스트(train/val txt 포함)
2. 동일한 모델 시작점(같은 pretrained/weights)
3. 동일한 하이퍼파라미터(lr, imgsz, batch, aug, loss gain 등)
4. 동일한 seed 및 라이브러리 버전
5. 동일한 디바이스/드라이버 조건

특히 팀에서 흔히 놓치는 포인트:
- `base.yaml` 기본값과 CLI 오버라이드가 섞이면 실제 적용값이 달라질 수 있음
- 외부 데이터 경로가 팀원별로 다르면 STAGE 0 결과가 달라짐
- 제출 CSV 평균 score는 로컬 검증 mAP와 직접 일치하지 않을 수 있음(후처리 영향)

---

## 7. 트러블슈팅
### 문제 1) STAGE 0에서 exclude 파일 못 찾음
- 증상: `exclude_file_names_file not found` 에러
- 조치: `configs/resources/exclude_4444_208.txt` 파일 존재 및 경로 확인

### 문제 2) STAGE 1에서 이미지 누락 임계치 초과
- 증상: critical missing 에러
- 조치: `data/raw/...` 경로 구조 확인, 외부 데이터 경로 존재 여부 확인

### 문제 3) STAGE 2에서 data.yaml 없음
- 조치: STAGE 1 먼저 실행하거나 `--data-yaml`로 직접 지정

### 문제 4) 제출 CSV가 기대보다 낮음
- 조치:
  1. `--conf`, `--iou`, `--min-conf`, `--topk` 확인
  2. `--class-map` 매핑 파일 확인
  3. `keep-category-ids` 필터가 과도하게 적용됐는지 확인

---

## 8. 팀 운영 권장 규칙
1. 공용 브랜치에서는 데이터/실험 산출물(`runs`, `submit*.csv`, `tmp_*`) 커밋 금지
2. 코드 반영 시 실행 명령 1세트(STAGE 0~4 또는 최소 STAGE 2/4)를 PR 본문에 명시
3. `run_name` 규칙 통일(예: `y8m1024_xxx_e60`)
4. 동일 실험 비교 시 변경점은 1~2개만 바꿔서 원인 추적 가능하게 유지

---

## 9. 빠른 시작용 최소 명령 세트
```powershell
python scripts/0_split_data.py --run-name exp_quick --config configs/base.yaml
python scripts/1_preprocess.py --run-name exp_quick --config configs/base.yaml
python scripts/2_train.py --run-name exp_quick --config configs/base.yaml --device 0
python scripts/4_submission.py --run-name exp_quick --config configs/base.yaml --device 0
```

이 4개가 정상 완료되면 팀원 환경에서 기본 파이프라인은 정상 동작하는 상태입니다.

