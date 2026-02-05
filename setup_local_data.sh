#!/bin/bash
# 로컬 환경 데이터 설정 스크립트

echo "🔍 데이터 위치 설정 도우미"
echo "======================================"
echo ""

# 1. 현재 데이터 구조 확인
echo "현재 data/raw/ 구조:"
ls -la data/raw/ 2>/dev/null || echo "  ❌ data/raw/ 디렉토리가 없습니다."
echo ""

# 2. 데이터 위치 입력 받기
echo "📌 데이터가 저장된 경로를 입력하세요:"
echo "   예: /mnt/aidrive/pill_data"
echo "   또는 ./my_data"
read -p "데이터 경로: " DATA_PATH

if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 경로가 존재하지 않습니다: $DATA_PATH"
    exit 1
fi

echo ""
echo "🔗 심볼릭 링크 생성 중..."

# 3. 심볼릭 링크 생성
mkdir -p data/raw

# train_images
if [ -d "$DATA_PATH/train_images" ]; then
    ln -sf "$DATA_PATH/train_images" data/raw/train_images
    echo "  ✅ train_images 링크 생성"
else
    echo "  ⚠️  train_images 폴더를 찾을 수 없습니다"
fi

# train_annotations
if [ -d "$DATA_PATH/train_annotations" ]; then
    ln -sf "$DATA_PATH/train_annotations" data/raw/train_annotations
    echo "  ✅ train_annotations 링크 생성"
else
    echo "  ⚠️  train_annotations 폴더를 찾을 수 없습니다"
fi

# test_images
if [ -d "$DATA_PATH/test_images" ]; then
    ln -sf "$DATA_PATH/test_images" data/raw/test_images
    echo "  ✅ test_images 링크 생성"
else
    echo "  ⚠️  test_images 폴더를 찾을 수 없습니다"
fi

echo ""
echo "✅ 설정 완료! 데이터 구조 확인:"
ls -la data/raw/

echo ""
echo "📊 데이터 개수 확인:"
echo "  Train images:      $(find data/raw/train_images -type f 2>/dev/null | wc -l)"
echo "  Train annotations: $(find data/raw/train_annotations -name "*.json" 2>/dev/null | wc -l)"
echo "  Test images:       $(find data/raw/test_images -type f 2>/dev/null | wc -l)"
