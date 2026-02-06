# Priority 2: 신뢰도 임계값 조정 가이드

## 📌 개요

mAP@0.75~0.95는 **정확한 bbox**가 중요합니다. 신뢰도 임계값을 조정하여 False Positive를 줄이고 정확도를 높일 수 있습니다.

---

## 🎯 전략

### mAP@0.75~0.95의 특징
- IoU 0.75~0.95에서 평가 (엄격한 기준)
- bbox가 정확해야 점수를 받음
- False Positive가 많으면 점수 하락

### Confidence Threshold 효과

| conf | Precision | Recall | mAP | 특징 |
|------|-----------|--------|-----|------|
| 0.25 | 낮음 | 높음 | ? | 많은 객체 검출, FP 증가 |
| 0.35 | 중간 | 중간 | ? | 균형잡힌 설정 |
| 0.50 | 높음 | 낮음 | ? | 정확한 객체만, FP 감소 |

---

## 🔧 사용법

### 1. Base Config 수정됨
```yaml
# configs/base.yaml
infer:
  conf_thr: 0.35  # 0.25 → 0.35 (기본값 상향)
```

### 2. 실험별 Config
```yaml
# configs/experiments/exp006_high_conf.yaml
infer:
  conf_thr: 0.5   # 높은 신뢰도
```

### 3. CLI로 동적 조정
```bash
# 기본값 (0.35)
python scripts/5_submission.py --run-name test_exp

# 특정 값 지정
python scripts/5_submission.py --run-name test_exp --conf 0.40

# 높은 신뢰도
python scripts/5_submission.py --run-name test_exp --conf 0.50
```

---

## 🧪 여러 임계값 테스트

### test_priority2.py 사용
```bash
# 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 자동 테스트
python test_priority2.py test_exp
```

**생성 파일**:
```
artifacts/test_exp/submissions/
├── submission_conf0.25.csv
├── submission_conf0.30.csv
├── submission_conf0.35.csv
├── submission_conf0.40.csv
├── submission_conf0.45.csv
└── submission_conf0.50.csv
```

**다음 단계**:
1. 각 파일을 Kaggle에 제출
2. mAP 점수 비교
3. 최적 conf 값 선택

---

## 📊 예상 결과

### Confidence별 특징

#### conf=0.25 (낮음)
- 많은 객체 검출
- False Positive 증가
- Recall ↑, Precision ↓
- mAP 0.75~0.95에서 불리

#### conf=0.35 (중간)
- 균형잡힌 설정
- FP/FN 적절히 조절
- **권장 시작점**

#### conf=0.50 (높음)
- 정확한 객체만 검출
- False Positive 감소
- Recall ↓, Precision ↑
- mAP 0.75~0.95에서 유리 (bbox 정확할 때)

---

## 🎓 최적화 팁

### 1. 점진적 조정
```bash
# Baseline (0.25)
python scripts/5_submission.py --run-name exp001 --conf 0.25

# 조금 올리기 (0.30)
python scripts/5_submission.py --run-name exp001 --conf 0.30

# 더 올리기 (0.35)
python scripts/5_submission.py --run-name exp001 --conf 0.35
```

### 2. 모델 품질에 따라 조정
- **모델이 좋음** (학습 잘됨) → conf 높여도 OK (0.4~0.5)
- **모델이 약함** (학습 부족) → conf 낮게 유지 (0.25~0.3)

### 3. Kaggle 피드백 활용
- Public LB에서 점수 확인
- conf 값 조정 후 재제출
- 최적값 찾기

---

## 📈 통계 확인

### submission.csv 통계
```bash
# 객체 개수 확인
wc -l artifacts/test_exp/submissions/submission_conf0.35.csv

# 이미지당 평균 객체
# (총 객체 - 1) / 842 = ?

# conf별 비교
for conf in 0.25 0.30 0.35 0.40 0.45 0.50; do
  echo "conf=$conf:"
  wc -l artifacts/test_exp/submissions/submission_conf$conf.csv
done
```

---

## ⚖️ Trade-off 이해

### Confidence ↑ 효과
✅ Precision 증가 (정확도 향상)  
✅ False Positive 감소  
✅ mAP@0.75~0.95 유리 (bbox 정확할 때)  

❌ Recall 감소 (누락 증가)  
❌ 일부 진짜 객체도 누락 가능  
❌ 총 검출 객체 개수 감소  

### 언제 conf 올릴까?
- Baseline 점수가 낮을 때
- False Positive가 많을 때
- bbox가 정확한 편일 때
- Precision > Recall 전략

### 언제 conf 낮출까?
- 객체 누락이 많을 때
- Recall이 너무 낮을 때
- 모델이 약할 때
- Recall > Precision 전략

---

## 🚀 빠른 시작

```bash
# 1. 기본값으로 제출
python scripts/5_submission.py --run-name test_exp

# 2. 여러 값 테스트
python test_priority2.py test_exp

# 3. Kaggle 제출 후 점수 확인

# 4. 최적값 선택
# 예: 0.35가 가장 좋았다면
python scripts/5_submission.py --run-name exp001 --conf 0.35
```

---

**작성**: 2026-02-05  
**담당**: @DM  
**상태**: Priority 2 완료 ✅
