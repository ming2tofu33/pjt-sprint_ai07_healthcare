from __future__ import annotations

import json
import os
import random               # AB / 추가
import math                 # AB / 추가
from pathlib import Path
from time import perf_counter
from typing import Iterable, Tuple
import albumentations as A  # AB / 추가
import cv2                  # AB / 추가
from tqdm import tqdm       # AB / 추가

from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger
from src.dataprep.process.dedup import (
    add_train_dedup_key,
    dedup_exact_records,
    dedup_iou_records,
    filter_expected4_actual3,
    filter_external_bbox_category_conflicts,
    filter_external_copy_json_records,
    filter_external_images_with_any_bad_bbox,
    filter_object_count,
    should_keep_external_record_after_dedup,
)
from src.dataprep.output.export import write_outputs
from src.dataprep.setup.io_utils import parse_one_json, scan_image_files, scan_json_files
from src.dataprep.output.manifest import write_manifest
from src.dataprep.process.normalize import extract_category_lookup_id, normalize_record
from src.dataprep.process.quality_audit import run_aux_detector_audit, run_pixel_overlap_audit
from src.dataprep.process.split import add_group_id, make_group_split, write_splits

logger = get_logger(__name__)


def build_train_mapping(
    train_json_paths: Iterable[Path],
) -> Tuple[dict[str, int], dict[str, str], list[dict[str, str]]]:
    """
    Train annotation에서 매핑 테이블을 구축한다.

    목적:
    - `mapping_key`(기본 `dl_idx`)를 기준으로 canonical category_id/category_name을 수집
    - 같은 key가 서로 다른 category_id를 가리키는 불일치 케이스를 `suspect_rows`로 기록

    반환:
    - id_map: mapping_key -> canonical category_id
    - name_map: mapping_key -> category name
    - suspect_rows: 동일 mapping_key 충돌 이력
    """
    id_map: dict[str, int] = {}
    name_map: dict[str, str] = {}
    suspect_rows: list[dict[str, str]] = []

    for p in train_json_paths:
        data, err = parse_one_json(p)
        if err or data is None:
            continue
        images = data.get("images") or []
        annotations = data.get("annotations") or []
        categories = data.get("categories") or []
        if len(images) != 1 or len(annotations) != 1 or len(categories) != 1:
            continue

        img0 = images[0]
        ann0 = annotations[0]
        cat0 = categories[0]

        lookup_key = extract_category_lookup_id(img0.get("drug_N"))
        if lookup_key is None:
            lookup_key = extract_category_lookup_id(img0.get("dl_mapping_code"))
        if lookup_key is None:
            continue

        cat_id = ann0.get("category_id")
        if not isinstance(cat_id, int):
            try:
                cat_id = int(str(cat_id))
            except Exception:
                continue

        prev = id_map.get(lookup_key)
        if prev is not None and prev != cat_id:
            suspect_rows.append(
                {
                    "source_json": str(p),
                    "dl_idx": lookup_key,
                    "prev_category_id": str(prev),
                    "new_category_id": str(cat_id),
                    "reason": "inconsistent_category_id",
                }
            )
            continue
        id_map[lookup_key] = cat_id

        name = cat0.get("name")
        if isinstance(name, str) and name.strip():
            name_map.setdefault(lookup_key, name)

    return id_map, name_map, suspect_rows


def build_df_clean(
    config: dict, config_path: Path, *, repo_root: Path, quiet: bool = False, log_every: int = 500
) -> dict:
    """
    전체 소스(train + external)를 읽어 정규화/정제한 `df_clean` 레코드 목록을 만든다.

    이 함수가 담당하는 범위:
    - JSON 파싱/정규화
    - train 기반 category mapping 구축
    - external 병합 및 dedup
    - 품질 필터(규칙 + 이미지 감사) 적용
    - split 준비용 group_id 부여

    실제 파일 쓰기(export/splits/manifest)는 `run()`에서 수행한다.
    """
    # 0단계) 경로/출력 디렉터리 준비
    base_dir = repo_root
    paths_cfg = config.get("paths", {})
    outputs_cfg = config.get("outputs", {})

    train_images_dir = resolve_path(paths_cfg["train_images_dir"], base_dir)
    train_ann_dir = resolve_path(paths_cfg["train_annotations_dir"], base_dir)
    processed_dir = resolve_path(paths_cfg["processed_dir"], base_dir)
    metadata_dir = resolve_path(paths_cfg["metadata_dir"], base_dir)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Audit 로그는 "결과가 비어도 파일 스키마를 고정"할 수 있도록 이름만 먼저 준비한다.
    audit_files = config.get("audit", {}).get("files", [])
    audit_logs = {name: [] for name in audit_files}
    audit_logs.setdefault("audit_unmapped_external.csv", [])

    logs: dict[str, list[dict]] = {
        "excluded_rows": [],
        "fixes_bbox": [],
        "excluded_rows_external": [],
        "fixes_bbox_external": [],
    }

    # 1단계) Train 기반 category 매핑 테이블 구축 (external 정렬에 사용)
    train_json_paths = scan_json_files(train_ann_dir, recursive=True)
    train_map_id, train_map_name, suspect_rows = build_train_mapping(train_json_paths)
    if "audit_suspect_files.csv" in audit_logs and suspect_rows:
        audit_logs["audit_suspect_files.csv"].extend(suspect_rows)

    # 파일명(lower) -> 이미지 경로 인덱스 + 동명이인 파일(중복명) 목록
    train_image_index, train_image_dups = scan_image_files(train_images_dir, recursive=True)

    if not quiet:
        print(f"[INFO] train: {len(train_json_paths)}개 json, {len(train_image_index)}개 이미지", flush=True)

    records: list[dict] = []
    train_dedup_keys: set[tuple] = set()
    train_image_size_cache: dict[str, tuple[int, int]] = {}

    # 2단계) Train 레코드 수집/정규화
    t0 = perf_counter()
    for i, p in enumerate(train_json_paths, start=1):
        if not quiet and log_every > 0 and (i == 1 or i % log_every == 0):
            dt = perf_counter() - t0
            print(f"[INFO] train 진행률: {i}/{len(train_json_paths)} ({dt:.1f}s)", flush=True)
        data, err = parse_one_json(p)
        if err or data is None:
            logs["excluded_rows"].append(
                {"source": "train", "file_name": "", "source_json": str(p), "reason_code": err or "parse_failed", "detail": ""}
            )
            if "audit_missing_labels.csv" in audit_logs:
                audit_logs["audit_missing_labels.csv"].append({"source_json": str(p), "reason": err})
            continue

        record, _ = normalize_record(
            source="train",
            data=data,
            source_json=p,
            image_index=train_image_index,
            image_duplicates=train_image_dups,
            image_size_cache=train_image_size_cache,
            config=config,
            mapping_id=None,
            mapping_name=None,
            logs=logs,
            audit=audit_logs,
        )
        if record is None:
            continue
        records.append(record)

        # external과의 중복 비교를 위해 train dedup key를 누적한다.
        dedup_key_fields = config.get("dedup", {}).get("exact", {}).get("key", [])
        add_train_dedup_key(record, dedup_key_fields, train_dedup_keys)

    # 3단계) External 레코드 수집/정규화 (옵션)
    external_cfg = config.get("external_data", {})
    if external_cfg.get("enabled", False):
        banned_patterns = external_cfg.get("banned_patterns", [])
        sources = external_cfg.get("sources", [])

        ingest_cfg = external_cfg.get("ingest", {})
        normalize_and_copy = bool(ingest_cfg.get("normalize_and_copy", False))
        out_ext_images = resolve_path(ingest_cfg.get("output_images_dir", "data/processed/external_normalized/images"), base_dir)
        out_ext_anns = resolve_path(ingest_cfg.get("output_annotations_dir", "data/processed/external_normalized/annotations"), base_dir)
        overwrite_outputs = bool(ingest_cfg.get("overwrite_outputs", False))

        dedup_cfg = external_cfg.get("dedup", {})
        dedup_against_train = bool(dedup_cfg.get("dedup_against_train", False))

        for src in sources:
            name = src.get("name", "external")
            img_dir = resolve_path(src["images_dir"], base_dir)
            ann_dir = resolve_path(src["annotations_dir"], base_dir)
            recursive = bool(src.get("recursive", True))

            # 금지 패턴이 경로/파일명에 있으면 즉시 중단한다.
            # (혼입 금지 데이터셋, 개인 데이터 등 안전장치)
            for root in (img_dir, ann_dir):
                for dirpath, _, filenames in os.walk(root):
                    if any(pat.lower() in dirpath.lower() for pat in banned_patterns):
                        raise RuntimeError(f"경로에서 금지 패턴이 발견되었습니다: {dirpath}")
                    for fn in filenames:
                        full = os.path.join(dirpath, fn)
                        if any(pat.lower() in full.lower() for pat in banned_patterns):
                            raise RuntimeError(f"파일에서 금지 패턴이 발견되었습니다: {full}")

            ext_image_index, ext_image_dups = scan_image_files(img_dir, recursive=recursive)
            ext_json_paths = scan_json_files(ann_dir, recursive=recursive)

            if not quiet:
                print(
                    f"[INFO] external[{name}]: {len(ext_json_paths)}개 json, {len(ext_image_index)}개 이미지",
                    flush=True,
                )

            ext_image_size_cache: dict[str, tuple[int, int]] = {}
            t1 = perf_counter()
            for j, p in enumerate(ext_json_paths, start=1):
                if not quiet and log_every > 0 and (j == 1 or j % log_every == 0):
                    dt = perf_counter() - t1
                    print(f"[INFO] external[{name}] 진행률: {j}/{len(ext_json_paths)} ({dt:.1f}s)", flush=True)
                data, err = parse_one_json(p)
                if err or data is None:
                    logs["excluded_rows"].append(
                        {
                            "source": "external",
                            "file_name": "",
                            "source_json": str(p),
                            "reason_code": err or "parse_failed",
                            "detail": "",
                        }
                    )
                    if "audit_missing_labels.csv" in audit_logs:
                        audit_logs["audit_missing_labels.csv"].append({"source_json": str(p), "reason": err})
                    continue

                record, normalized_json = normalize_record(
                    source="external",
                    data=data,
                    source_json=p,
                    image_index=ext_image_index,
                    image_duplicates=ext_image_dups,
                    image_size_cache=ext_image_size_cache,
                    config=config,
                    mapping_id=train_map_id,
                    mapping_name=train_map_name,
                    logs=logs,
                    audit=audit_logs,
                    external_cfg=external_cfg,
                )
                if record is None:
                    continue

                # train과 exact key가 동일한 external row는 설정에 따라 제거한다.
                dedup_key_fields = dedup_cfg.get("exact", {}).get("key", [])
                keep_record = should_keep_external_record_after_dedup(
                    record,
                    dedup_against_train=dedup_against_train,
                    dedup_key_fields=dedup_key_fields,
                    train_dedup_keys=train_dedup_keys,
                    logs=logs,
                    audit_logs=audit_logs,
                )
                if not keep_record:
                    continue

                records.append(record)

                # 필요 시 정규화된 external 결과(JSON/이미지)를 별도 폴더에 복제한다.
                # (재현/감사용 산출물)
                if normalize_and_copy and normalized_json is not None:
                    try:
                        rel = p.relative_to(ann_dir)
                    except Exception:
                        rel = Path(p.name)
                    out_ann_path = out_ext_anns / name / rel
                    out_img_path = out_ext_images / name / record["file_name"]

                    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
                    out_img_path.parent.mkdir(parents=True, exist_ok=True)

                    if overwrite_outputs or not out_ann_path.exists():
                        with out_ann_path.open("w", encoding="utf-8") as f:
                            json.dump(normalized_json, f, ensure_ascii=False, indent=2)
                    if overwrite_outputs or not out_img_path.exists():
                        src_img = ext_image_index.get(record["file_name"].lower())
                        if src_img is not None:
                            out_img_path.write_bytes(src_img.read_bytes())

    # 4단계) 전체(train + external) 품질/중복 필터 체인
    # 4-1) external copy.json 패턴 제거
    records = filter_external_copy_json_records(records, config, logs, audit_logs)

    # 4-2) external 동일 bbox + 다중 category 충돌 제거
    records = filter_external_bbox_category_conflicts(records, config, logs, audit_logs)

    # 4-3) external에서 bad bbox 이력이 있는 이미지는 전체 제거(옵션)
    records = filter_external_images_with_any_bad_bbox(records, config, logs, audit_logs)

    dedup_root_cfg = config.get("dedup", {})
    exact_cfg = dedup_root_cfg.get("exact", {})
    iou_cfg = dedup_root_cfg.get("iou", {})
    records = dedup_exact_records(records, exact_cfg, logs, audit_logs)
    records = dedup_iou_records(records, iou_cfg, logs, audit_logs)

    # 4-4) 이미지 단위 객체 수 규칙(min/max 또는 valid set) 필터
    records = filter_object_count(records, config, logs, audit_logs)

    # 4-5) 파일명 코드 수(예: 4)와 실제 객체 수(예: 3) 불일치 필터
    records = filter_expected4_actual3(records, config, logs, audit_logs)

    # 4-6) 픽셀 기반 bbox 정합 감사(coverage / bbox IoU)
    pixel_audit = run_pixel_overlap_audit(
        records=records,
        config=config,
        logs=logs,
        audit_logs=audit_logs,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )
    records = pixel_audit.filtered_records

    # 4-7) 보조 detector 기반 감사(max IoU)
    aux_audit = run_aux_detector_audit(
        records=records,
        config=config,
        logs=logs,
        audit_logs=audit_logs,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )
    records = aux_audit.filtered_records

    # 5단계) 누수 방지 split을 위한 group_id 부여
    split_cfg = config.get("split", {})
    add_group_id(records, split_cfg)

    # 6단계) category 매핑 테이블 출력용 row 구성
    mapping_rows = []
    for dl_idx, cat_id in train_map_id.items():
        mapping_rows.append(
            {
                "dl_idx": dl_idx,
                "canonical_category_id": cat_id,
                "name": train_map_name.get(dl_idx, ""),
            }
        )

    return {
        "records": records,
        "logs": logs,
        "audit_logs": audit_logs,
        "paths": {"processed_dir": processed_dir, "metadata_dir": metadata_dir},
        "outputs": outputs_cfg,
        "config_path": config_path,
        "repo_root": repo_root,
        "config": config,
        "mapping_rows": mapping_rows,
    }


def run(
    config: dict, *, config_path: Path, repo_root: Path, quiet: bool = False, log_every: int = 500
) -> dict:
    """
    전처리 파이프라인 실행 진입점.

    실행 순서:
    1) `build_df_clean()`으로 메모리 상 결과물(records/logs/audit) 생성
    2) `write_outputs()`로 df_clean / 제외로그 / 수정로그 / audit 파일 저장
    3) `make_group_split()` + `write_splits()`로 group 기반 train/val 분할 저장
    4) `write_manifest()`로 실행 요약(설정 해시/커밋/통계) 저장
    """
    result = build_df_clean(config, config_path, repo_root=repo_root, quiet=quiet, log_every=log_every)

    records = result["records"]
    logs = result["logs"]
    audit_logs = result["audit_logs"]
    mapping_rows = result["mapping_rows"]
    processed_dir = result["paths"]["processed_dir"]
    metadata_dir = result["paths"]["metadata_dir"]
    outputs_cfg = result["outputs"]
    base_dir = result["repo_root"]

    write_outputs(
        records=records,
        logs=logs,
        audit_logs=audit_logs,
        mapping_rows=mapping_rows,
        processed_dir=processed_dir,
        metadata_dir=metadata_dir,
        outputs_cfg=outputs_cfg,
        config=config,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )

    splits = make_group_split(records, config.get("split", {}), config.get("random_seed", 42))
    splits_name = config.get("split", {}).get("splits_name", "splits.csv")
    write_splits(metadata_dir, splits_name, splits)

    write_manifest(
        records=records,
        logs=logs,
        config_path=result["config_path"],
        metadata_dir=metadata_dir,
        outputs_cfg=outputs_cfg,
        repo_root=base_dir,
    )

    return result


# AB / 추가: 소수 클래스 기하학적 증강 함수
def augment_minority_classes(config: dict, yolo_root: Path) -> dict | None:
    """소수 클래스 오프라인 증강.

    config['imbalance_fix'] 설정에 따라 min_threshold 미만인 클래스의
    이미지를 target_count까지 증강한다. 한 이미지의 **모든** bbox를 유지한다.

    Returns
    -------
    dict | None
        증강 요약 (augmented_classes, total_generated, errors).
        imbalance_fix 설정이 없으면 None.
    """
    fix_cfg = config.get('imbalance_fix', {})
    if not fix_cfg:
        return None
    if not bool(fix_cfg.get("enabled", False)):
        logger.info("오프라인 소수 클래스 증강 비활성화 (imbalance_fix.enabled=false)")
        return None

    min_threshold = fix_cfg.get('min_threshold', 100)
    target_count = fix_cfg.get('target_count', 300)

    logger.info("소수 클래스 증강 시작 (threshold=%d, target=%d)", min_threshold, target_count)

    train_img_dir = yolo_root / "images" / "train"
    train_lbl_dir = yolo_root / "labels" / "train"

    # 기존 증강 파일 제거 [충돌/오류 방지]
    logger.debug("이전 증강 파일 정리 중...")
    for p in train_lbl_dir.glob("*_aug*.txt"):
        p.unlink()
    for p in train_img_dir.glob("*_aug*.jpg"):
        p.unlink()

    # 2. 동적 Transform 리스트 생성 (True인 것만 담기)
    augment_list = []

    # [A] 기하학적 변환 (enable 체크)
    if fix_cfg.get('geometric', {}).get('enable', True):
        augment_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    # [B] 밝기 및 대비
    br_cfg = fix_cfg.get('brightness', {})
    if br_cfg.get('enable', False):
        augment_list.append(A.RandomBrightnessContrast(
            brightness_limit=br_cfg.get('limit', 0.15),
            contrast_limit=br_cfg.get('limit', 0.15),
            p=br_cfg.get('p', 0.5)
        ))

    # [C] 선명도 조절
    sh_cfg = fix_cfg.get('sharpen', {})
    if sh_cfg.get('enable', False):
        alpha = sh_cfg.get('alpha', [0.2, 0.5])
        augment_list.append(A.Sharpen(
            alpha=(alpha[0], alpha[1]),
            lightness=(0.5, 1.0),
            p=sh_cfg.get('p', 0.5)
        ))

    # [D] 흐림 효과
    bl_cfg = fix_cfg.get('blur', {})
    if bl_cfg.get('enable', False):
        augment_list.append(A.OneOf([
            A.Blur(blur_limit=bl_cfg.get('limit', 3), p=1.0),
            A.MedianBlur(blur_limit=bl_cfg.get('limit', 3), p=1.0),
        ], p=bl_cfg.get('p', 0.2)))

    # [E] 필수 위치 조정
    augment_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5))

    # 최종 Transform 구성
    transform = A.Compose(
        augment_list,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )

    # 3. 데이터 수집 — unique 파일 기반 카운트 (B4 fix)
    all_labels = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt') and '_aug' not in f]
    class_to_files: dict[str, set] = {}
    for lbl_name in all_labels:
        with open(train_lbl_dir / lbl_name, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    class_id = parts[0]
                    class_to_files.setdefault(class_id, set()).add(lbl_name)

    # 4. 증강 실행
    error_count = 0
    total_generated = 0
    augmented_classes: list[str] = []

    for class_id, files_set in class_to_files.items():
        files = list(files_set)  # set → list (random.choice 용)
        current_count = len(files)
        if current_count >= min_threshold:
            continue

        needed = target_count - current_count
        augmented_classes.append(class_id)

        logger.info("  class %s: %d개 (< %d) → %d개 증강 예정",
                     class_id, current_count, min_threshold, needed)

        for i in tqdm(range(needed), desc=f"cls {class_id}", leave=False):
            try:
                src_lbl = random.choice(files)
                base = Path(src_lbl).stem
                # B1 fix: class_id를 파일명에 포함하여 충돌 방지
                new_name = f"{base}_cls{class_id}_aug{i}"

                # 이미지 로드
                img_path = next(
                    (train_img_dir / f"{base}{ext}"
                     for ext in ['.jpg', '.png', '.jpeg', '.JPG']
                     if (train_img_dir / f"{base}{ext}").exists()),
                    None,
                )
                if not img_path:
                    continue

                img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

                # 라벨 로드 — 모든 클래스
                bboxes, cls_labels = [], []
                with open(train_lbl_dir / src_lbl, 'r') as f:
                    for line in f:
                        p = line.split()
                        if p:
                            cls_labels.append(int(p[0]))
                            bboxes.append([float(x) for x in p[1:]])

                # 증강 적용
                augmented = transform(image=img, bboxes=bboxes, class_labels=cls_labels)

                # 결과 저장 — B2 fix: 모든 클래스 bbox 유지
                if len(augmented['bboxes']) > 0:
                    cv2.imwrite(
                        str(train_img_dir / f"{new_name}.jpg"),
                        cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR),
                    )

                    wrote_lines = 0
                    with open(train_lbl_dir / f"{new_name}.txt", 'w') as f:
                        for c, b in zip(augmented['class_labels'], augmented['bboxes']):
                            if any(math.isnan(x) for x in b):
                                continue

                            # 좌표 클램핑 (0.0~1.0)
                            xc, yc, w, h = [max(0.0, min(1.0, x)) for x in b]
                            if w > 0 and h > 0:
                                f.write(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                                wrote_lines += 1

                    # 빈 라벨 파일 방지: 유효 bbox 없으면 이미지+라벨 삭제
                    if wrote_lines == 0:
                        new_img_path = train_img_dir / f"{new_name}.jpg"
                        new_lbl_path = train_lbl_dir / f"{new_name}.txt"
                        if new_img_path.exists():
                            new_img_path.unlink()
                        if new_lbl_path.exists():
                            new_lbl_path.unlink()
                        continue

                    total_generated += 1

            except Exception as exc:  # B3 fix: 예외 로깅
                error_count += 1
                logger.warning("증강 실패 (class %s, src=%s): %s", class_id, src_lbl, exc)
                continue

    logger.info("소수 클래스 증강 완료: classes=%d, generated=%d, errors=%d",
                len(augmented_classes), total_generated, error_count)

    return {
        "augmented_classes": len(augmented_classes),
        "total_generated": total_generated,
        "errors": error_count,
    }
