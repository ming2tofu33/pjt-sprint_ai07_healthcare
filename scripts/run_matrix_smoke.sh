# ============================================================
# run_matrix_smoke.sh
# - 프로젝트용 매트릭스 스모크 테스트를 시간 제한 안에서 폭넓게 실행한다.
# - STAGE 0~1은 한 번만(기준 실행) 수행하고, 산출물을 각 케이스에서 재사용한다.
# - 각 케이스는 가벼운 설정 덮어쓰기로 STAGE 2~4를 실행한다.
# ============================================================
set -euo pipefail

PREFIX="matrix_$(date +%Y%m%d_%H%M%S)"
BASE_CONFIG="configs/experiments/smoke_test.yaml"
MODE="balanced"            # 실행 모드: quick | balanced | broad
BASE_EPOCHS=1
DEVICE="cpu"
START_STAGE=2
STOP_STAGE=4
MAX_CASES=0                # 0이면 전체 케이스 실행
QUIET=false
KEEP_TEMP=false
DRY_RUN=false

usage() {
    cat <<'EOF'
사용법:
  bash scripts/run_matrix_smoke.sh [options]

옵션:
  --prefix <name>         실행 이름 prefix (기본: matrix_YYYYmmdd_HHMMSS)
  --base-config <path>    기준 실험 config (기본: configs/experiments/smoke_test.yaml)
  --mode <name>           quick | balanced | broad (기본: balanced)
  --epochs <int>          케이스별 기본 epoch (기본: 1)
  --device <value>        STAGE 2~4 실행 디바이스 (기본: cpu)
  --start-stage <int>     케이스 실행 시작 stage (2~4, 기본: 2)
  --stop-stage <int>      케이스 실행 종료 stage (2~4, 기본: 4)
  --max-cases <int>       앞에서 N개 케이스만 실행 (0=전체, 기본: 0)
  --quiet                 파이프라인 스크립트에 --quiet 전달
  --keep-temp             playground/matrix/<prefix> 산출물 유지
  --dry-run               config만 생성하고 실행 없이 계획만 출력
  -h, --help              도움말 출력

예시:
  bash scripts/run_matrix_smoke.sh
  bash scripts/run_matrix_smoke.sh --mode broad --epochs 1 --device 0
  bash scripts/run_matrix_smoke.sh --mode quick --max-cases 2 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --base-config) BASE_CONFIG="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --epochs) BASE_EPOCHS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --start-stage) START_STAGE="$2"; shift 2 ;;
        --stop-stage) STOP_STAGE="$2"; shift 2 ;;
        --max-cases) MAX_CASES="$2"; shift 2 ;;
        --quiet) QUIET=true; shift ;;
        --keep-temp) KEEP_TEMP=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERR] 알 수 없는 옵션: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! "$MODE" =~ ^(quick|balanced|broad)$ ]]; then
    echo "[ERR] --mode 값은 quick, balanced, broad 중 하나여야 합니다." >&2
    exit 1
fi
if ! [[ "$BASE_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] --epochs 는 양의 정수여야 합니다: $BASE_EPOCHS" >&2
    exit 1
fi
if ! [[ "$START_STAGE" =~ ^[0-9]+$ ]] || ! [[ "$STOP_STAGE" =~ ^[0-9]+$ ]]; then
    echo "[ERR] --start-stage/--stop-stage 는 정수여야 합니다." >&2
    exit 1
fi
if (( START_STAGE < 2 || START_STAGE > 4 )); then
    echo "[ERR] --start-stage 는 [2, 4] 범위여야 합니다." >&2
    exit 1
fi
if (( STOP_STAGE < START_STAGE || STOP_STAGE > 4 )); then
    echo "[ERR] --stop-stage 는 [start-stage, 4] 범위여야 합니다." >&2
    exit 1
fi
if ! [[ "$MAX_CASES" =~ ^[0-9]+$ ]]; then
    echo "[ERR] --max-cases 는 0 이상의 정수여야 합니다: $MAX_CASES" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "[ERR] 기준 config 파일을 찾을 수 없습니다: $BASE_CONFIG" >&2
    exit 1
fi
if [[ ! -f "scripts/run_pipeline.sh" ]]; then
    echo "[ERR] scripts/run_pipeline.sh 파일을 찾을 수 없습니다." >&2
    exit 1
fi

QUIET_ARG=()
if [[ "$QUIET" == true ]]; then
    QUIET_ARG=(--quiet)
fi

DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi

TMP_ROOT="playground/matrix/$PREFIX"
CFG_GEN_DIR="$TMP_ROOT/configs"
LOG_DIR="$TMP_ROOT/logs"
mkdir -p "$CFG_GEN_DIR" "$LOG_DIR"
SUMMARY_TSV="$TMP_ROOT/summary.tsv"
echo -e "case\trun_name\tstatus\telapsed_sec\tmAP50\tmAP50_95\tmAP75_95\tconfig" > "$SUMMARY_TSV"

declare -a CASES
build_cases() {
    CASES=(
        "11s_base|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|1.0|0.010|0.60|0.25|0.50"
        "n26_fast|yolo26n|yolo26n.pt|512|16|${BASE_EPOCHS}|1.0|0.010|0.60|0.25|0.50"
        "11s_low_conf|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|1.0|0.010|0.60|0.10|0.50"
    )

    if [[ "$MODE" == "balanced" || "$MODE" == "broad" ]]; then
        CASES+=(
            "11s_high_conf|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|1.0|0.010|0.60|0.40|0.50"
            "11s_strict_nms|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|1.0|0.010|0.70|0.25|0.60"
            "v8s_fast|yolov8s|yolov8s.pt|512|8|${BASE_EPOCHS}|1.0|0.010|0.60|0.25|0.50"
        )
    fi

    if [[ "$MODE" == "broad" ]]; then
        CASES+=(
            "11s_loose_nms|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|1.0|0.010|0.45|0.20|0.40"
            "11s_no_mosaic|yolo11s|yolo11s.pt|640|8|${BASE_EPOCHS}|0.0|0.010|0.60|0.25|0.50"
            "11s_small_img|yolo11s|yolo11s.pt|448|16|${BASE_EPOCHS}|1.0|0.010|0.60|0.25|0.50"
            "n26_low_lr|yolo26n|yolo26n.pt|512|16|${BASE_EPOCHS}|1.0|0.003|0.60|0.25|0.50"
        )
    fi
}

write_case_config() {
    local out="$1"
    local arch="$2"
    local pretrained="$3"
    local imgsz="$4"
    local batch="$5"
    local epochs="$6"
    local mosaic="$7"
    local lr0="$8"
    local eval_nms="$9"
    local sub_conf="${10}"
    local sub_nms="${11}"

    PYTHONUTF8=1 python - "$BASE_CONFIG" "$out" \
        "$arch" "$pretrained" "$imgsz" "$batch" "$epochs" "$mosaic" "$lr0" "$eval_nms" "$sub_conf" "$sub_nms" <<'PY'
import sys
from pathlib import Path

import yaml

from src.utils.config_loader import load_experiment_config

(
    base_config,
    out_path,
    arch,
    pretrained,
    imgsz,
    batch,
    epochs,
    mosaic,
    lr0,
    eval_nms,
    sub_conf,
    sub_nms,
) = sys.argv[1:]

cfg, _repo_root = load_experiment_config(
    Path(base_config).resolve(),
    Path("scripts/run_pipeline.sh").resolve(),
)
paths = cfg.get("paths")
if not isinstance(paths, dict):
    print(f"[ERR] config에 paths 섹션이 없습니다: {base_config}", file=sys.stderr)
    sys.exit(2)
required_keys = ("datasets_dir", "processed_dir", "runs_dir")
missing = [k for k in required_keys if not isinstance(paths.get(k), str) or not str(paths.get(k)).strip()]
if missing:
    print(f"[ERR] config paths 누락: {', '.join(missing)} ({base_config})", file=sys.stderr)
    sys.exit(2)

cfg.setdefault("model", {})
cfg["model"]["architecture"] = arch
cfg["model"]["pretrained"] = pretrained

train = cfg.setdefault("train", {})
train["epochs"] = int(epochs)
train["imgsz"] = int(imgsz)
train["batch"] = int(batch)
train["workers"] = 0
train["patience"] = int(epochs)
train["plots"] = False
train["verbose"] = False
train["mosaic"] = float(mosaic)
train["lr0"] = float(lr0)

evaluate = cfg.setdefault("evaluate", {})
evaluate["nms_iou"] = float(eval_nms)

submission = cfg.setdefault("submission", {})
submission["conf"] = float(sub_conf)
submission["nms_iou"] = float(sub_nms)
submission.setdefault("max_det_per_image", 4)

out = Path(out_path).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

clone_seed_outputs() {
    local seed_run="$1"
    local target_run="$2"

    PYTHONUTF8=1 python - "$BASE_CONFIG" "$seed_run" "$target_run" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from src.utils.config_loader import load_experiment_config

base_config, seed_run, target_run = sys.argv[1:]
cfg, repo_root = load_experiment_config(
    Path(base_config).resolve(),
    Path("scripts/run_pipeline.sh").resolve(),
)
paths = cfg.get("paths")
if not isinstance(paths, dict):
    print(f"[ERR] config에 paths 섹션이 없습니다: {base_config}", file=sys.stderr)
    sys.exit(2)
required_keys = ("datasets_dir", "processed_dir")
missing = [k for k in required_keys if not isinstance(paths.get(k), str) or not str(paths.get(k)).strip()]
if missing:
    print(f"[ERR] config paths 누락: {', '.join(missing)} ({base_config})", file=sys.stderr)
    sys.exit(2)
prefix = cfg.get("yolo_convert", {}).get("dataset_prefix", "pill_od_yolo")

datasets_dir = Path(paths.get("datasets_dir", "data/processed/datasets"))
if not datasets_dir.is_absolute():
    datasets_dir = (repo_root / datasets_dir).resolve()

processed_dir = Path(paths.get("processed_dir", "data/processed/cache"))
if not processed_dir.is_absolute():
    processed_dir = (repo_root / processed_dir).resolve()

seed_dataset = datasets_dir / f"{prefix}_{seed_run}"
seed_cache = processed_dir / seed_run
target_dataset = datasets_dir / f"{prefix}_{target_run}"
target_cache = processed_dir / target_run

for required in (seed_dataset, seed_cache):
    if not required.exists():
        print(f"[ERR] 기준 실행 산출물이 없습니다: {required}", file=sys.stderr)
        sys.exit(2)

def clone_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except Exception:
        shutil.copytree(src, dst)

clone_tree(seed_dataset, target_dataset)
clone_tree(seed_cache, target_cache)
PY
}

collect_case_metrics() {
    local run_name="$1"
    PYTHONUTF8=1 python - "$BASE_CONFIG" "$run_name" <<'PY'
import json
import sys
from pathlib import Path

from src.utils.config_loader import load_experiment_config

base_config, run_name = sys.argv[1:]
cfg, repo_root = load_experiment_config(
    Path(base_config).resolve(),
    Path("scripts/run_pipeline.sh").resolve(),
)
paths = cfg.get("paths")
if not isinstance(paths, dict):
    print(f"[ERR] config에 paths 섹션이 없습니다: {base_config}", file=sys.stderr)
    sys.exit(2)
if not isinstance(paths.get("runs_dir"), str) or not str(paths.get("runs_dir")).strip():
    print(f"[ERR] config paths 누락: runs_dir ({base_config})", file=sys.stderr)
    sys.exit(2)
runs_dir = Path(paths.get("runs_dir", "runs"))
if not runs_dir.is_absolute():
    runs_dir = (repo_root / runs_dir).resolve()

metrics_path = runs_dir / run_name / "metrics.json"
if not metrics_path.exists():
    print("NA\tNA\tNA")
    sys.exit(0)

try:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    print("NA\tNA\tNA")
    sys.exit(0)

def pick(*keys):
    for key in keys:
        if key in data and isinstance(data[key], (int, float)):
            return f"{float(data[key]):.4f}"
    return "NA"

m50 = pick("eval_mAP50", "mAP50")
m5095 = pick("eval_mAP50_95", "mAP50_95")
m7595 = pick("eval_mAP75_95", "mAP75_95")
print(f"{m50}\t{m5095}\t{m7595}")
PY
}

run_case() {
    local case_name="$1"
    local cfg_path="$2"
    local run_name="${PREFIX}_${case_name}"
    local stage_log="$LOG_DIR/${case_name}.log"

    clone_seed_outputs "$SEED_RUN" "$run_name"

    local start_ts
    start_ts="$(date +%s)"
    set +e
    bash scripts/run_pipeline.sh \
        --run-name "$run_name" \
        --config "$cfg_path" \
        --start "$START_STAGE" \
        --stop "$STOP_STAGE" \
        "${DEVICE_ARG[@]}" \
        "${QUIET_ARG[@]}" \
        >"$stage_log" 2>&1
    local code=$?
    set -e
    local elapsed=$(( "$(date +%s)" - start_ts ))

    if [[ $code -eq 0 ]]; then
        local metrics
        metrics="$(collect_case_metrics "$run_name")"
        local m50 m5095 m7595
        IFS=$'\t' read -r m50 m5095 m7595 <<< "$metrics"
        echo -e "${case_name}\t${run_name}\tPASS\t${elapsed}\t${m50}\t${m5095}\t${m7595}\t${cfg_path}" >> "$SUMMARY_TSV"
        echo "[OK] ${case_name} (${elapsed}s) mAP75_95=${m7595}"
    else
        echo -e "${case_name}\t${run_name}\tFAIL(${code})\t${elapsed}\tNA\tNA\tNA\t${cfg_path}" >> "$SUMMARY_TSV"
        echo "[FAIL] ${case_name} (exit=${code}, ${elapsed}s) log=${stage_log}" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

build_cases
if (( MAX_CASES > 0 && MAX_CASES < ${#CASES[@]} )); then
    CASES=( "${CASES[@]:0:$MAX_CASES}" )
fi

echo "============================================================"
echo "매트릭스 스모크 실행"
echo "  prefix      : $PREFIX"
echo "  mode        : $MODE (${#CASES[@]}개 케이스)"
echo "  base_config : $BASE_CONFIG"
echo "  epochs      : $BASE_EPOCHS"
echo "  device      : $DEVICE"
echo "  stages      : $START_STAGE -> $STOP_STAGE (케이스별)"
echo "  dry_run     : $DRY_RUN"
echo "============================================================"

SEED_RUN="${PREFIX}_seed"
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "[1/3] seed 산출물 생성 (STAGE 0~1): $SEED_RUN"
    bash scripts/run_pipeline.sh \
        --run-name "$SEED_RUN" \
        --config "$BASE_CONFIG" \
        --start 0 \
        --stop 1 \
        "${QUIET_ARG[@]}"
else
    echo ""
    echo "[1/3] DRY-RUN: seed 단계 생략"
fi

echo ""
echo "[2/3] 케이스 config 생성"
for row in "${CASES[@]}"; do
    IFS='|' read -r case_name arch pre imgsz batch epochs mosaic lr0 eval_nms sub_conf sub_nms <<< "$row"
    cfg_path="$CFG_GEN_DIR/${case_name}.yaml"
    write_case_config "$cfg_path" "$arch" "$pre" "$imgsz" "$batch" "$epochs" "$mosaic" "$lr0" "$eval_nms" "$sub_conf" "$sub_nms"
    echo "  - $cfg_path"
done

echo ""
echo "[3/3] 매트릭스 실행"
FAIL_COUNT=0
if [[ "$DRY_RUN" == true ]]; then
    for row in "${CASES[@]}"; do
        IFS='|' read -r case_name _ <<< "$row"
        run_name="${PREFIX}_${case_name}"
        cfg_path="$CFG_GEN_DIR/${case_name}.yaml"
        echo -e "${case_name}\t${run_name}\tDRY_RUN\t0\tNA\tNA\tNA\t${cfg_path}" >> "$SUMMARY_TSV"
    done
else
    for row in "${CASES[@]}"; do
        IFS='|' read -r case_name _ <<< "$row"
        cfg_path="$CFG_GEN_DIR/${case_name}.yaml"
        run_case "$case_name" "$cfg_path"
    done
fi

echo ""
echo "==================== 요약 ===================="
cat "$SUMMARY_TSV"
echo "================================================="
echo "요약 파일 : $SUMMARY_TSV"
echo "로그 폴더 : $LOG_DIR"

if [[ "$KEEP_TEMP" != true ]]; then
    echo ""
    echo "[NOTE] 임시 매트릭스 설정은 $TMP_ROOT 경로에 유지됩니다."
    echo "       같은 prefix를 사용하면 생성된 config/log를 다시 확인할 수 있습니다."
fi

if [[ "$DRY_RUN" == false && $FAIL_COUNT -gt 0 ]]; then
    echo "[FAIL] 실패한 케이스 수: $FAIL_COUNT" >&2
    exit 1
fi

echo "[OK] 매트릭스 스모크 실행이 완료되었습니다."
exit 0
