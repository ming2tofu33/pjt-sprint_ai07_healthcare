# ============================================================
# run_pipeline.sh: STAGE 0~4 오케스트레이션
#
# Usage:
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml
#
# Options:
#   --run-name           run 이름 (필수)
#   --config             실험 config YAML 경로 (필수)
#   --device             GPU 디바이스 (기본: config 값 사용)
#   --skip               건너뛸 STAGE 번호 (콤마 구분, 예: 0,1)
#   --start              시작 STAGE 번호 (기본: 0)
#   --stop               종료 STAGE 번호 (기본: 4, 해당 STAGE 포함)
#   --conf               STAGE 4 confidence threshold 오버라이드
#   --resume-train       STAGE 2를 --resume 모드로 강제 실행
#   --auto-resume-train  STAGE 2에서 last.pt가 있으면 자동 재개
#   --verbose            모든 STAGE 상세 로그 출력 (기본: STAGE 0만 상세 로그)
#
# Examples:
#   # 전체 파이프라인 실행
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml
#
#   # 상세 로그와 함께 실행
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml --verbose
#
#   # STAGE 2부터 재실행
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml --start 2
#
#   # STAGE 2 학습 재개 (명시적)
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml --start 2 --resume-train
#
#   # STAGE 2 자동 재개 (checkpoint 없으면 새로 시작)
#   bash scripts/run_pipeline.sh --run-name exp01 --config configs/experiments/baseline.yaml --start 2 --auto-resume-train
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

RUN_NAME=""
CONFIG=""
DEVICE=""
SKIP=""
START=0
STOP=4
CONF=""
VERBOSE_ALL=false
RESUME_TRAIN=""
AUTO_RESUME_TRAIN=""
TRAIN_DATA_YAML_OVERRIDE=""
TRAIN_DATA_YAML=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)  RUN_NAME="$2"; shift 2 ;;
        --config)    CONFIG="$2"; shift 2 ;;
        --train-data-yaml) TRAIN_DATA_YAML_OVERRIDE="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --skip)      SKIP="$2"; shift 2 ;;
        --start)     START="$2"; shift 2 ;;
        --stop)      STOP="$2"; shift 2 ;;
        --conf)      CONF="$2"; shift 2 ;;
        --resume-train) RESUME_TRAIN="--resume"; shift ;;
        --auto-resume-train) AUTO_RESUME_TRAIN="--auto-resume"; shift ;;
        --verbose)   VERBOSE_ALL=true; shift ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] 알 수 없는 옵션: $1${NC}" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo -e "${RED}[ERROR] --run-name 이 필요합니다.${NC}" >&2
    exit 1
fi
if [[ -z "$CONFIG" ]]; then
    echo -e "${RED}[ERROR] --config 가 필요합니다.${NC}" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo -e "${RED}[ERROR] config 파일이 존재하지 않습니다: $CONFIG${NC}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

declare -A SKIP_MAP
if [[ -n "$SKIP" ]]; then
    IFS=',' read -ra SKIP_STAGES <<< "$SKIP"
    for s in "${SKIP_STAGES[@]}"; do
        SKIP_MAP["$s"]=1
    done
fi

should_run() {
    local stage=$1
    if (( stage < START || stage > STOP )); then
        return 1
    fi
    if [[ -n "${SKIP_MAP[$stage]+_}" ]]; then
        return 1
    fi
    return 0
}

run_stage() {
    local stage_num=$1
    local stage_name=$2
    shift 2
    local cmd=("$@")

    if ! should_run "$stage_num"; then
        echo -e "${YELLOW}[SKIP] STAGE $stage_num: $stage_name${NC}"
        return 0
    fi

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  STAGE $stage_num: $stage_name${NC}"
    echo -e "${BLUE}========================================${NC}"

    local start_time
    start_time=$(date +%s)

    if "${cmd[@]}"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        echo -e "${GREEN}[OK] STAGE $stage_num 완료 (${elapsed}s)${NC}"
    else
        local exit_code=$?
        echo -e "${RED}[FAIL] STAGE $stage_num 실패 (exit code: $exit_code)${NC}" >&2
        exit $exit_code
    fi
}

COMMON_ARGS=(--run-name "$RUN_NAME" --config "$CONFIG")
DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi
VERBOSE_ARG=()
if [[ "$VERBOSE_ALL" == true ]]; then
    VERBOSE_ARG=(--verbose)
fi

# 기본값: STAGE 0은 verbose 활성화(데이터 파이프라인 진행상황 확인용)
STAGE0_VERBOSE_ARG=(--verbose)

TRAIN_RESUME_ARG=()
if [[ -n "$RESUME_TRAIN" ]]; then
    TRAIN_RESUME_ARG=(--resume)
elif [[ -n "$AUTO_RESUME_TRAIN" ]]; then
    TRAIN_RESUME_ARG=(--auto-resume)
fi

if ! should_run 2 && [[ -n "$RESUME_TRAIN" || -n "$AUTO_RESUME_TRAIN" ]]; then
    echo -e "${YELLOW}[WARN] STAGE 2가 실행되지 않아 --resume-train/--auto-resume-train 옵션은 무시됩니다.${NC}"
fi

PIPELINE_START=$(date +%s)

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Healthcare Pill OD - Full Pipeline${NC}"
echo -e "${GREEN}  run_name : $RUN_NAME${NC}"
echo -e "${GREEN}  config   : $CONFIG${NC}"
echo -e "${GREEN}  stages   : $START -> $STOP${NC}"
if [[ -n "$SKIP" ]]; then
    echo -e "${GREEN}  skip     : $SKIP${NC}"
fi
if [[ -n "$DEVICE" ]]; then
    echo -e "${GREEN}  device   : $DEVICE${NC}"
fi
if [[ -n "$RESUME_TRAIN" ]]; then
    echo -e "${GREEN}  stage2_resume : explicit${NC}"
elif [[ -n "$AUTO_RESUME_TRAIN" ]]; then
    echo -e "${GREEN}  stage2_resume : auto${NC}"
fi
echo -e "${GREEN}============================================================${NC}"

run_stage 0 "데이터 정제 + 분할" \
    python scripts/0_split_data.py "${COMMON_ARGS[@]}" "${STAGE0_VERBOSE_ARG[@]}"

run_stage 1 "YOLO 데이터셋 변환" \
    python scripts/1_preprocess.py "${COMMON_ARGS[@]}" "${VERBOSE_ARG[@]}"

# AB / 학습용 data.yaml 결정 로직
if [[ -n "$TRAIN_DATA_YAML_OVERRIDE" ]]; then
    TRAIN_DATA_YAML="$TRAIN_DATA_YAML_OVERRIDE"
    echo -e "${GREEN}  Custom Data YAML used: $TRAIN_DATA_YAML${NC}"
else
    #AB / 기존처럼 파이프라인이 만든 경로 사용
    TRAIN_DATA_YAML="data/processed/datasets/pill_odyolo${RUN_NAME}/data.yaml"
fi

# AB / 파이썬 스크립트에 넘겨줄 인자 조립
TRAIN_DATA_YAML_ARG=(--data-yaml "$TRAIN_DATA_YAML")

run_stage 2 "모델 학습" \
    python scripts/2_train.py "${COMMON_ARGS[@]}" "${DEVICE_ARG[@]}" "${TRAIN_RESUME_ARG[@]}" "${TRAIN_DATA_YAML_ARG[@]}" "${VERBOSE_ARG[@]}"

run_stage 3 "모델 평가" \
    python scripts/3_evaluate.py "${COMMON_ARGS[@]}" "${DEVICE_ARG[@]}" "${VERBOSE_ARG[@]}"

CONF_ARG=()
if [[ -n "$CONF" ]]; then
    CONF_ARG=(--conf "$CONF")
fi
run_stage 4 "제출 파일 생성" \
    python scripts/4_submission.py "${COMMON_ARGS[@]}" "${DEVICE_ARG[@]}" "${CONF_ARG[@]}" "${VERBOSE_ARG[@]}"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  파이프라인 완료!${NC}"
echo -e "${GREEN}  run_name     : $RUN_NAME${NC}"
echo -e "${GREEN}  총 소요 시간 : ${TOTAL_ELAPSED}s${NC}"
echo -e "${GREEN}============================================================${NC}"
