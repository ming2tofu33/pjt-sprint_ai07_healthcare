# Priority 2: Confidence Threshold 가이드

> 목표: **mAP@[0.75:0.95]** 구간 성능 개선 (BBox 정밀도 중심)

---

## 1. 개요

mAP@[0.75:0.95]는 **BBox 정확도**에 민감합니다.  
Confidence threshold(conf)를 조정하면 **Precision/Recall 균형**이 바뀌며, 최적값은 모델·데이터에 따라 달라집니다.

---

## 2. 기본 원칙 (Trade-off)

- **conf ↑** → Precision ↑ / Recall ↓ / FP 감소
- **conf ↓** → Recall ↑ / Precision ↓ / FP 증가
- 최적 conf는 **Public LB 기준으로 실험적으로 결정**하는 것이 가장 안전합니다.

---

## 3. 설정 방법

### 3-1. base.yaml에서 기본값 변경

```yaml
# configs/base.yaml
infer:
  conf_thr: 0.35
```

### 3-2. 실험 YAML에서 override

```yaml
# configs/experiments/exp006_high_conf.yaml
_base_: "../base.yaml"

infer:
  conf_thr: 0.50
```

### 3-3. CLI로 즉시 변경

```bash
python scripts/5_submission.py --run-name exp001 --conf 0.25
python scripts/5_submission.py --run-name exp001 --conf 0.35
python scripts/5_submission.py --run-name exp001 --conf 0.50
```

---

## 4. 추천 범위

- **기본 시작값**: `0.25 ~ 0.35` (현재 base.yaml 기본값: `0.35`)
- **과검출(FP) 많음** → conf를 올리기 (0.40~0.50)
- **미검출(FN) 많음** → conf를 내리기 (0.20~0.30)

---

## 5. conf 스윕 예시 (권장)

> `5_submission.py`는 항상 `submission.csv`를 덮어씁니다.  
> **conf 값을 구분해서 저장하려면 복사/이름 변경**이 필요합니다.

### Bash
```bash
RUN=exp001
for conf in 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
  python scripts/5_submission.py --run-name $RUN --conf $conf
  cp artifacts/$RUN/submissions/submission.csv artifacts/$RUN/submissions/submission_conf$conf.csv
  echo "saved: submission_conf$conf.csv"
done
```

### PowerShell
```powershell
$RUN = "exp001"
$CONFS = 0.20,0.25,0.30,0.35,0.40,0.45,0.50
foreach ($conf in $CONFS) {
  python scripts/5_submission.py --run-name $RUN --conf $conf
  Copy-Item "artifacts/$RUN/submissions/submission.csv" "artifacts/$RUN/submissions/submission_conf$conf.csv" -Force
  Write-Host "saved: submission_conf$conf.csv"
}
```

> 이후 Kaggle에 각각 제출하여 Public LB 점수를 비교합니다.

---

## 6. 제출 파일 체크 포인트

- `category_id`는 **원본 COCO ID**인지 확인 (0~55 아님)
- `bbox_*`는 **절대 픽셀 좌표 xywh**인지 확인 (정규화 아님)
- `image_id`는 파일명 stem의 정수 (`0.png` → `0`)

---

## 7. 요약

- **conf는 mAP@[0.75:0.95]에 직접 영향**
- 기본값 0.35에서 시작해 **0.20~0.50 범위**로 스윕 권장
- 최종 기준은 **Public LB 성능**

---

**문서 업데이트**: 2026-02-06
